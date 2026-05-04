"""
EurekaAI — Stage 4 (High School) Data Preparation
고등학교 단계: 비판적 사고 + 에세이 + Teacher 채점 품질 필터링
  - Teacher.score_response()로 품질 낮은 데이터 필터링 ← 핵심!
  - 에세이 스타일 텍스트
  - CoT 데이터 재활용 (Stage 3)
"""
import sys, json, random, logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.teacher.ollama_teacher import OllamaTeacher
from tools.collect_data import load_sharegpt_ko

Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/stage4_data_prep_{_ts}.log"
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_log_file, encoding="utf-8"),
], format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"📄 로그: {_log_file}")

STAGE_DIR  = Path("data/processed/stage4")
TRAIN_FILE = STAGE_DIR / "train_stage4.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage4.jsonl"
TARGET_EVAL = 2000
PREV_STAGES = [Path(f"data/processed/stage{i}/train_stage{i}.jsonl") for i in range(4)]

ESSAY_TOPICS_KO = [
    "환경 보호를 위해 우리가 할 수 있는 일들을 논술하세요.",
    "인터넷이 사회에 미치는 긍정적, 부정적 영향을 분석하세요.",
    "독서가 인간 발달에 미치는 영향을 서술하세요.",
    "민주주의의 장단점을 비교하여 서술하세요.",
    "과학 기술 발전이 인류에게 가져온 변화를 논하세요.",
    "청소년 스마트폰 과사용 문제와 해결 방안을 논술하세요.",
    "인공지능 시대에 필요한 역량은 무엇인가 논하세요.",
    "세계화가 문화에 미치는 영향을 논술하세요.",
]

ESSAY_TOPICS_EN = [
    "Discuss the impact of social media on modern communication.",
    "Analyze the causes and effects of climate change.",
    "Should school uniforms be mandatory? Argue both sides.",
    "How does technology affect human relationships?",
    "What are the benefits and drawbacks of globalization?",
    "Discuss the importance of critical thinking in education.",
    "How should governments balance economic growth and environmental protection?",
]

def generate_scored_qa(n_articles: int = 100, score_threshold: float = 0.6, max_workers: int = 2) -> list[dict]:
    """Teacher Q&A 생성 + 채점 — 점수 높은 것만 학습 데이터로 사용."""
    teacher = OllamaTeacher()
    if not teacher.is_available():
        logger.warning("⚠️  Ollama 미사용")
        return []

    try:
        from datasets import load_dataset
        logger.info(f"🤖 Q&A 생성+채점 (위키 {n_articles}개, 임계값 {score_threshold})...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split=f"train[200:{200+n_articles}]")
        texts = [(i, item.get("text","").strip()[:600]) for i, item in enumerate(ko_ds) if len(item.get("text","").strip()) >= 100]
    except Exception as e:
        logger.warning(f"⚠️  Wiki 로드 실패: {e}"); return []

    samples = []
    def process_one(args):
        idx, text = args
        results = []
        try:
            pairs = teacher.generate_qa_pairs(passage=text, stage=4, n=3)
            for pair in pairs:
                q = pair.get("question","").strip()
                a = pair.get("answer","").strip()
                if not (q and a): continue
                # Teacher로 품질 채점
                score = teacher.score_response(question=q, answer=a, stage=4)
                if score >= score_threshold:
                    results.append({"text": f"Q: {q}\nA: {a}", "source": "scored_qa", "stage": 4, "score": score})
        except Exception as e:
            logger.debug(f"문서 {idx} 처리 실패: {e}")
        return results

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, t): t[0] for t in texts}
        for future in as_completed(futures):
            completed += 1
            samples.extend(future.result())
            if completed % 20 == 0:
                logger.info(f"  완료: {completed}/{len(texts)} | 고품질 Q&A {len(samples)}건")

    logger.info(f"✅ 채점 Q&A: {len(samples):,}건 (임계값 {score_threshold} 이상)")
    return samples

def generate_essays(max_workers: int = 2) -> list[dict]:
    """Teacher로 에세이 생성."""
    teacher = OllamaTeacher()
    if not teacher.is_available(): return []
    all_topics = [(t, 4) for t in (ESSAY_TOPICS_KO + ESSAY_TOPICS_EN)] * 5
    random.shuffle(all_topics)
    logger.info(f"🤖 에세이 생성 ({len(all_topics)}개)...")

    def write_essay(args):
        topic, stage = args
        try:
            from core.teacher.ollama_teacher import STAGE_CONTEXTS, STAGE_NAMES_KO
            prompt = (
                f"다음 주제에 대해 정교하고 논리적인 에세이를 작성해 주세요.\n"
                f"대상: {STAGE_NAMES_KO[stage]}학생\n"
                f"수준 및 컨텍스트: {STAGE_CONTEXTS[stage]}\n\n"
                f"주제: {topic}"
            )
            resp = teacher.generate(prompt, temperature=0.7, max_tokens=1024, stage=stage)
            if resp and len(resp.content) > 100:
                return {"text": f"주제: {topic}\n\n{resp.content}", "source": "teacher_essay", "stage": 4}
        except Exception: pass
        return None

    samples = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(write_essay, all_topics):
            if result: samples.append(result)
    logger.info(f"✅ 에세이: {len(samples):,}건")
    return samples

def load_wiki(max_samples: int = 10000) -> list[dict]:
    samples = []
    try:
        from datasets import load_dataset
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:500]")
        for item in ko_ds:
            if len(samples) >= max_samples: break
            for para in item.get("text","").strip().split("\n"):
                para = para.strip()
                if 100 <= len(para) <= 600:
                    samples.append({"text": para, "source": "wiki_ko", "stage": 4})
                if len(samples) >= max_samples: break
        random.shuffle(samples)
        logger.info(f"✅ Wikipedia: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️  Wikipedia 실패: {e}")
    return samples

def load_replay(max_per_stage: int = 2000) -> list[dict]:
    samples = []
    for path in PREV_STAGES:
        if not path.exists(): continue
        lines = path.read_text().strip().split("\n")
        random.shuffle(lines)
        n = min(max_per_stage, int(len(lines) * 0.10))
        for line in lines[:n]:
            d = json.loads(line); d["source"] = f"{path.parent.name}_replay"; d["stage"] = 4
            samples.append(d)
    logger.info(f"✅ 이전 단계 리플레이: {len(samples):,}건")
    return samples

def save_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 저장: {len(samples):,}건 → {path}")

def main():
    logger.info("=" * 60)
    logger.info("   EurekaAI — Stage 4: High School Data Preparation")
    logger.info("=" * 60)
    all_samples = []
    all_samples += generate_scored_qa(n_articles=100, score_threshold=0.6)
    all_samples += generate_essays(max_workers=2)
    all_samples += load_wiki(max_samples=40000)
    
    logger.info("📥 ShareGPT_Ko (심화) 로드 중...")
    all_samples += load_sharegpt_ko(target=50000, stage_idx=4, min_len=250, max_len=1000, max_turns=8)
    
    all_samples += load_replay(max_per_stage=2000)

    seen = set()
    deduped = [s for s in all_samples if (k := s["text"][:80]) not in seen and not seen.add(k)]
    random.shuffle(deduped)

    from collections import Counter
    total = len(deduped)
    logger.info(f"\n📊 데이터 구성 ({total:,}건):")
    for src, cnt in Counter(s["source"] for s in deduped).most_common():
        logger.info(f"   {src:25s}: {cnt:6,} ({cnt/total*100:.1f}%)")

    n_eval = min(TARGET_EVAL, int(total * 0.1))
    save_jsonl(deduped[n_eval:], TRAIN_FILE)
    save_jsonl(deduped[:n_eval], EVAL_FILE)
    logger.info(f"\n✅ Stage 4 데이터 완료: Train {total-n_eval:,} / Eval {n_eval:,}")

if __name__ == "__main__":
    main()
