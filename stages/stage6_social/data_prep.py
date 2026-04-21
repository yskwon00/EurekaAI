"""
EurekaAI — Stage 6 (Social) Data Preparation
사회인 단계: 자연스러운 대화, 코드, 복잡 문제 + RLHF 선호 데이터
  - Teacher.create_preference_pairs()로 선호 데이터 생성 ← Stage 6 핵심!
  - 대화형 데이터, 코드 설명, 뉴스 스타일
"""
import sys, json, random, logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.teacher.ollama_teacher import OllamaTeacher

Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/stage6_data_prep_{_ts}.log"
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_log_file, encoding="utf-8"),
], format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"📄 로그: {_log_file}")

STAGE_DIR  = Path("data/processed/stage6")
TRAIN_FILE = STAGE_DIR / "train_stage6.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage6.jsonl"
TARGET_EVAL = 2000
PREV_STAGES = [Path(f"data/processed/stage{i}/train_stage{i}.jsonl") for i in range(6)]

# 선호 데이터 생성용 질문 목록
PREFERENCE_QUESTIONS = [
    "최근 인공지능 기술의 발전이 사회에 미치는 영향은 무엇인가요?",
    "기후 변화 문제를 해결하기 위한 현실적인 방안을 제시해주세요.",
    "파이썬으로 피보나치 수열을 구현하는 방법을 설명해주세요.",
    "좋은 리더십이란 무엇인지 설명해주세요.",
    "머신러닝과 딥러닝의 차이점을 설명해주세요.",
    "How can governments balance economic growth with environmental protection?",
    "What are the ethical implications of artificial intelligence?",
    "Explain the concept of blockchain technology in simple terms.",
    "What makes a good software architecture?",
    "How do neural networks learn from data?",
]

CONVERSATION_TEMPLATES = [
    ("안녕하세요! 오늘 날씨가 어때요?", "안녕하세요! 오늘은 맑고 따뜻한 날씨네요. 산책하기 딱 좋은 날이에요."),
    ("파이썬에서 리스트와 튜플의 차이가 뭔가요?", "리스트는 변경 가능(mutable)하고 튜플은 변경 불가(immutable)입니다. 리스트는 [], 튜플은 ()를 사용해요."),
    ("머신러닝 공부를 시작하려면 어떻게 해야 하나요?", "파이썬 기초를 먼저 익히고, numpy/pandas를 배운 후 scikit-learn으로 시작하세요. 캐글 대회도 큰 도움이 됩니다."),
    ("What is the difference between supervised and unsupervised learning?", "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."),
    ("How do I optimize a Python function for performance?", "Use profiling tools like cProfile, consider NumPy for numerical operations, use list comprehensions, and cache results with functools.lru_cache."),
]

CODE_EXAMPLES = [
    "# 파이썬 데코레이터 예시\ndef timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} 실행 시간: {time.time()-start:.3f}초')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    import time\n    time.sleep(1)\n    return 'done'",
    "# 이진 탐색 구현\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "# 제너레이터로 피보나치 수열\ndef fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nfib = fibonacci()\nprint([next(fib) for _ in range(10)])",
]


def generate_preference_data(max_workers: int = 2) -> list[dict]:
    """Teacher.create_preference_pairs()로 RLHF 선호 데이터 생성 — Stage 6 핵심!"""
    teacher = OllamaTeacher()
    if not teacher.is_available(): return []

    questions = PREFERENCE_QUESTIONS * 5
    random.shuffle(questions)
    logger.info(f"🤖 선호 데이터 쌍 생성 ({len(questions)}개 질문)...")

    def process_one(question):
        try:
            # 두 가지 답변 생성 (각각 temperature 다르게, stage=6으로 캐시)
            resp_a = teacher.generate(question, temperature=0.9, use_cache=True, stage=6)
            resp_b = teacher.generate(question, temperature=0.3, use_cache=True, stage=6)
            # 빈 응답 필터링
            if not resp_a or not resp_b: return None
            if not resp_a.content or not resp_b.content: return None
            if len(resp_a.content) < 20 or len(resp_b.content) < 20: return None

            pref = teacher.create_preference_pairs(
                question=question, answer_a=resp_a.content, answer_b=resp_b.content, stage=6
            )
            if pref and pref.get("chosen") and len(pref["chosen"]) > 20:
                return {
                    "text": f"Q: {question}\nA: {pref['chosen']}",
                    "source": "preference_chosen",
                    "stage": 6,
                    "reason": pref.get("reason", ""),
                }
        except Exception as e:
            logger.debug(f"선호 데이터 실패: {e}")
        return None

    samples = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(process_one, q): q for q in questions}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                samples.append(result)
            if i % 10 == 0:
                logger.info(f"  진행: {i}/{len(questions)} | 선호 데이터: {len(samples)}건")
    logger.info(f"✅ 선호 데이터: {len(samples):,}건")
    return samples


def generate_conversation_data() -> list[dict]:
    """대화 형식 학습 데이터."""
    samples = []
    for user_msg, assistant_msg in CONVERSATION_TEMPLATES:
        text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        for _ in range(20):
            samples.append({"text": text, "source": "conversation", "stage": 6})
    logger.info(f"✅ 대화 데이터: {len(samples):,}건")
    return samples


def generate_code_data() -> list[dict]:
    samples = []
    for code in CODE_EXAMPLES:
        for _ in range(15):
            samples.append({"text": code, "source": "code_example", "stage": 6})
    logger.info(f"✅ 코드 데이터: {len(samples):,}건")
    return samples


def load_social_wiki(max_samples: int = 10000) -> list[dict]:
    """300자 이상 단락만 수집 (Stage 5 교훈 적용)."""
    samples = []
    try:
        from datasets import load_dataset
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:1000]")
        for item in ko_ds:
            if len(samples) >= max_samples: break
            for para in item.get("text","").strip().split("\n"):
                para = para.strip()
                if 300 <= len(para) <= 1000:   # 300자 이상으로 상향 (과적합 방지)
                    samples.append({"text": para, "source": "wiki_social", "stage": 6})
                if len(samples) >= max_samples: break
        random.shuffle(samples)
        logger.info(f"✅ 사회인 수준 Wikipedia: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️  Wikipedia 실패: {e}")
    return samples


def load_replay(max_per_stage: int = 1000) -> list[dict]:
    samples = []
    for path in PREV_STAGES:
        if not path.exists(): continue
        lines = [l for l in path.read_text().strip().split("\n") if l.strip()]
        random.shuffle(lines)
        n = min(max_per_stage, int(len(lines) * 0.05))
        for line in lines[:n]:
            try:
                d = json.loads(line)
                # 300자 미만 리플레이 제외
                if len(d.get("text", "")) < 300: continue
                d["source"] = f"{path.parent.name}_replay"
                d["stage"] = 6
                samples.append(d)
            except: pass
    logger.info(f"✅ 이전 단계 리플레이: {len(samples):,}건")
    return samples


def save_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 저장: {len(samples):,}건 → {path}")


def upload_wandb_artifact(train_path: Path, eval_path: Path):
    """W&B 아티팩트 업로드 및 리니지 연결 (dataset-stage5 → dataset-stage6)."""
    try:
        import wandb, time
        run = wandb.init(
            project="EurekaAI-Curriculum",
            job_type="data_prep",
            name="stage6_data_generation",
            config={"stage": 6, "source": "wiki+preference+conversation"},
            reinit=True,
        )
        # 이전 Stage 아티팩트 연결
        try:
            run.use_artifact("dataset-stage5:latest")
            logger.info("🔗 [W&B] 이전 dataset-stage5 연결 완료")
        except Exception as e:
            logger.warning(f"⚠️  dataset-stage5 연결 실패: {e}")

        art = wandb.Artifact("dataset-stage6", type="dataset",
                             description="Stage 6 Social dataset (preference+wiki+conversation)")
        art.add_dir(str(STAGE_DIR))
        # 강제 버저닝용 dummy
        dummy = STAGE_DIR / "dummy_lineage.txt"
        dummy.write_text(f"lineage:{time.time()}")
        art.add_file(str(dummy))
        run.log_artifact(art)
        dummy.unlink(missing_ok=True)
        run.finish()
        logger.info("✅ [W&B] dataset-stage6 아티팩트 업로드 완료")
    except Exception as e:
        logger.warning(f"⚠️  W&B 업로드 실패: {e}")


def main():
    logger.info("=" * 60)
    logger.info("   EurekaAI — Stage 6: Social Data Preparation")
    logger.info("=" * 60)
    all_samples = []
    all_samples += generate_preference_data(max_workers=2)   # RLHF 선호 ★
    all_samples += generate_conversation_data()
    all_samples += generate_code_data()
    all_samples += load_social_wiki(max_samples=10000)
    all_samples += load_replay(max_per_stage=1000)

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
    logger.info(f"\n✅ Stage 6 데이터 완료: Train {total-n_eval:,} / Eval {n_eval:,}")

    # W&B 리니지 업로드
    upload_wandb_artifact(TRAIN_FILE, EVAL_FILE)


if __name__ == "__main__":
    main()
