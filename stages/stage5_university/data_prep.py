"""
EurekaAI — Stage 5 (University) Data Preparation
대학교 단계: 학술 텍스트 + RLHF 보상 신호 필터링
  - Teacher.score_response()로 보상 신호 생성 (학술 수준 채점)
  - 점수 ≥ 0.7인 고품질 데이터만 사용
"""
import sys, json, random, logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.teacher.ollama_teacher import OllamaTeacher

Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/stage5_data_prep_{_ts}.log"
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_log_file, encoding="utf-8"),
], format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"📄 로그: {_log_file}")

STAGE_DIR  = Path("data/processed/stage5")
TRAIN_FILE = STAGE_DIR / "train_stage5.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage5.jsonl"
TARGET_EVAL = 2000
PREV_STAGES = [Path(f"data/processed/stage{i}/train_stage{i}.jsonl") for i in range(5)]

def generate_academic_qa(n_articles: int = 100, score_threshold: float = 0.7, max_workers: int = 2) -> list[dict]:
    """학술 수준 Q&A 생성 + 고품질 필터링."""
    teacher = OllamaTeacher()
    if not teacher.is_available(): return []
    try:
        from datasets import load_dataset
        logger.info(f"🤖 학술 Q&A 생성 (위키 {n_articles}개, 임계값 {score_threshold})...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split=f"train[500:{500+n_articles}]")
        texts = [(i, item.get("text","").strip()[:800]) for i, item in enumerate(ko_ds) if len(item.get("text","").strip()) >= 150]
    except Exception as e:
        logger.warning(f"⚠️  Wiki 로드 실패: {e}"); return []

    samples = []
    def process_one(args):
        idx, text = args
        results = []
        try:
            pairs = teacher.generate_qa_pairs(passage=text, stage=5, n=3)
            for pair in pairs:
                q, a = pair.get("question","").strip(), pair.get("answer","").strip()
                if not (q and a): continue
                score = teacher.score_response(question=q, answer=a, stage=5)
                if score >= score_threshold:
                    results.append({"text": f"Q: {q}\nA: {a}", "source": "academic_qa", "stage": 5, "score": score})
        except Exception as e:
            logger.debug(f"문서 {idx} 실패: {e}")
        return results

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, t): t[0] for t in texts}
        for future in as_completed(futures):
            completed += 1
            samples.extend(future.result())
            if completed % 20 == 0:
                logger.info(f"  완료: {completed}/{len(texts)} | 고품질 Q&A {len(samples)}건")

    logger.info(f"✅ 학술 Q&A: {len(samples):,}건")
    return samples

def load_academic_wiki(max_samples: int = 15000) -> list[dict]:
    """위키 학술 섹션 — 긴 단락 위주."""
    samples = []
    try:
        from datasets import load_dataset
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:800]")
        for item in ko_ds:
            if len(samples) >= max_samples: break
            for para in item.get("text","").strip().split("\n"):
                para = para.strip()
                if 150 <= len(para) <= 800:
                    samples.append({"text": para, "source": "wiki_academic", "stage": 5})
                if len(samples) >= max_samples: break
        random.shuffle(samples)
        logger.info(f"✅ 학술 Wikipedia: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️  Wikipedia 실패: {e}")
    return samples

def load_replay(max_per_stage: int = 1500) -> list[dict]:
    samples = []
    for path in PREV_STAGES:
        if not path.exists(): continue
        lines = path.read_text().strip().split("\n")
        random.shuffle(lines)
        n = min(max_per_stage, int(len(lines) * 0.08))
        for line in lines[:n]:
            d = json.loads(line); d["source"] = f"{path.parent.name}_replay"; d["stage"] = 5
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
    logger.info("   EurekaAI — Stage 5: University Data Preparation")
    logger.info("=" * 60)
    all_samples = []
    all_samples += generate_academic_qa(n_articles=100, score_threshold=0.7)
    all_samples += load_academic_wiki(max_samples=15000)
    all_samples += load_replay(max_per_stage=1500)

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
    logger.info(f"\n✅ Stage 5 데이터 완료: Train {total-n_eval:,} / Eval {n_eval:,}")

if __name__ == "__main__":
    main()
