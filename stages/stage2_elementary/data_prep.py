"""
EurekaAI — Stage 2 (Elementary) Data Preparation
초등학교 단계: 기초 지식 + Q&A 학습
  - Teacher(Ollama)로 Q&A 쌍 생성 ← 핵심!
  - 쉬운 문법, 기초 지식, 질문-답변 형식
  - Stage 0+1 데이터 리플레이 (망각 방지)
"""

import sys
import json
import random
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.teacher.ollama_teacher import OllamaTeacher

# ── 로그 설정 (파일 + 콘솔 동시 출력) ──────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/stage2_data_prep_{_ts}.log"
_fmt = "%(asctime)s [%(levelname)s] %(message)s"
_handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_log_file, encoding="utf-8"),
]
for _h in _handlers:
    _h.setFormatter(logging.Formatter(_fmt))
logging.basicConfig(level=logging.INFO, handlers=_handlers)
logger = logging.getLogger(__name__)
logger.info(f"📄 데이터 준비 로그: {_log_file}")

STAGE_DIR = Path("data/processed/stage2")
TRAIN_FILE = STAGE_DIR / "train_stage2.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage2.jsonl"
TARGET_EVAL = 2000

STAGE0_TRAIN = Path("data/processed/stage0/train_stage0.jsonl")
STAGE1_TRAIN = Path("data/processed/stage1/train_stage1.jsonl")


# ── 초등학교 수준 Seed Texts ──────────────────────────────────────────────────

KO_ELEMENTARY = [
    "지구는 태양 주위를 돌고 있어요. 이것을 공전이라고 해요.",
    "물은 섭씨 0도에서 얼고, 100도에서 끓어요.",
    "식물은 햇빛과 물과 이산화탄소로 음식을 만들어요. 이 과정이 광합성이에요.",
    "한국의 수도는 서울이에요. 서울에는 약 천만 명이 살아요.",
    "숫자 1부터 10까지 더하면 55가 돼요.",
    "삼각형의 세 각도의 합은 180도예요.",
    "우리 몸에는 206개의 뼈가 있어요.",
    "빛은 소리보다 훨씬 빠르게 이동해요.",
    "지구에는 7개의 대륙과 5개의 바다가 있어요.",
    "사계절은 봄, 여름, 가을, 겨울이에요.",
    "한글은 세종대왕이 1443년에 만드셨어요.",
    "꿀벌은 꽃에서 꿀을 모아요. 이 과정에서 꽃가루도 옮겨줘요.",
    "무지개는 빨강, 주황, 노랑, 초록, 파랑, 남색, 보라 색깔이에요.",
    "사람의 심장은 1분에 약 70번 뛰어요.",
    "지구에서 달까지의 거리는 약 38만 킬로미터예요.",
]

EN_ELEMENTARY = [
    "The Earth orbits the Sun once every 365 days.",
    "Photosynthesis is the process plants use to make food from sunlight.",
    "Water boils at 100 degrees Celsius and freezes at 0 degrees.",
    "The human body has 206 bones.",
    "Sound travels through air at about 343 meters per second.",
    "A triangle has three sides and three angles that add up to 180 degrees.",
    "The capital of the United States is Washington D.C.",
    "Gravity is the force that pulls objects toward the Earth.",
    "There are eight planets in our solar system.",
    "Lightning is caused by electrical charges in clouds.",
    "Mammals are warm-blooded animals that breathe air.",
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "Multiplication is repeated addition.",
    "The speed of light is about 300,000 kilometers per second.",
    "Dinosaurs lived on Earth millions of years before humans.",
]

QA_TEMPLATES = [
    ("물은 몇 도에서 끓나요?", "물은 섭씨 100도에서 끓어요."),
    ("지구는 무엇 주위를 도나요?", "지구는 태양 주위를 돌아요."),
    ("광합성이란 무엇인가요?", "식물이 햇빛, 물, 이산화탄소로 음식을 만드는 과정이에요."),
    ("한국의 수도는 어디인가요?", "한국의 수도는 서울이에요."),
    ("삼각형의 세 각도의 합은 얼마인가요?", "삼각형의 세 각도의 합은 180도예요."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many planets are in our solar system?", "There are eight planets in our solar system."),
    ("What is photosynthesis?", "Photosynthesis is the process plants use to make food from sunlight, water, and carbon dioxide."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit."),
    ("How fast does light travel?", "Light travels at about 300,000 kilometers per second."),
]


def format_qa(question: str, answer: str) -> str:
    """Q&A를 CLM 학습 형식으로 포맷팅."""
    return f"Q: {question}\nA: {answer}"


def generate_seed_corpus(n_repeat: int = 10) -> list[dict]:
    samples = []
    for text in (KO_ELEMENTARY + EN_ELEMENTARY):
        for _ in range(n_repeat):
            samples.append({"text": text, "source": "seed_elementary", "stage": 2})
    for q, a in QA_TEMPLATES:
        for _ in range(n_repeat):
            samples.append({"text": format_qa(q, a), "source": "seed_qa", "stage": 2})
    logger.info(f"✅ Seed corpus: {len(samples):,}건")
    return samples


def generate_teacher_qa(n_articles: int = 150, max_workers: int = 2) -> list[dict]:
    """
    Teacher(Ollama)로 Q&A 쌍 생성 — Stage 2의 핵심!
    ThreadPoolExecutor로 병렬 처리.
    max_workers=2: Ollama 로컬 단일 모델 과부하 방지.
    캐시된 응답은 즉시 반환 (OllamaTeacher 자동 처리).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    teacher = OllamaTeacher()
    if not teacher.is_available():
        logger.warning("⚠️  Ollama 미사용 — Teacher Q&A 스킵")
        return []

    try:
        from datasets import load_dataset
        logger.info(f"🤖 Teacher Q&A 병렬 생성 시작 (위키 {n_articles}개 문서, {max_workers}개 동시)...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split=f"train[:{n_articles}]")
        texts = [
            (i, item.get("text", "").strip()[:600])
            for i, item in enumerate(ko_ds)
            if len(item.get("text", "").strip()) >= 100
        ]
    except Exception as e:
        logger.warning(f"⚠️  위키 데이터 로드 실패: {e}")
        return []

    samples = []
    completed = 0

    def process_one(args):
        idx, text = args
        try:
            pairs = teacher.generate_qa_pairs(passage=text, stage=2, n=3)
            results = []
            for pair in pairs:
                q = pair.get("question", "").strip()
                a = pair.get("answer", "").strip()
                if q and a and len(q) > 5:
                    results.append({"text": format_qa(q, a), "source": "teacher_qa", "stage": 2})
            return results
        except Exception as e:
            logger.debug(f"  문서 {idx} Q&A 실패: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, t): t[0] for t in texts}
        for future in as_completed(futures):
            completed += 1
            results = future.result()
            samples.extend(results)
            if completed % 10 == 0:
                cached = sum(1 for f in futures if f.done())
                logger.info(f"  완료: {completed}/{len(texts)} 문서 | Q&A {len(samples)}건 생성")

    logger.info(f"✅ Teacher Q&A: {len(samples):,}건 (병렬 {max_workers}개 처리)")
    return samples


def generate_synthetic(n: int = 500) -> list[dict]:
    """Ollama로 초등학교 수준 텍스트 생성."""
    teacher = OllamaTeacher()
    if not teacher.is_available():
        return []
    logger.info(f"🤖 Stage 2 합성 데이터 {n}건 생성...")
    raw = teacher.generate_synthetic_stage_data(stage=2, n_samples=n, language="mixed")
    samples = [{**s, "source": "ollama_synthetic", "stage": 2} for s in raw if s.get("text", "").strip()]
    logger.info(f"✅ 합성 데이터: {len(samples)}건")
    return samples


def load_tinystories(max_samples: int = 15000) -> list[dict]:
    """TinyStories — 기초 영문 (일반화 유지)."""
    try:
        from datasets import load_dataset
        logger.info(f"📥 TinyStories ({max_samples:,}건)...")
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{max_samples}]")
        samples = []
        for item in ds:
            text = item.get("text", "").strip()
            if 50 <= len(text) <= 800:
                samples.append({"text": text[:600], "source": "tinystories", "stage": 2})
        logger.info(f"✅ TinyStories: {len(samples):,}건")
        return samples
    except Exception as e:
        logger.warning(f"⚠️  TinyStories 실패: {e}")
        return []


def load_korean_wiki(max_sentences: int = 10000) -> list[dict]:
    """한국어 위키 — 초등학교 수준 문장."""
    samples = []
    try:
        from datasets import load_dataset
        logger.info(f"📥 Korean Wikipedia ({max_sentences:,}문장)...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:400]")
        for item in ko_ds:
            if len(samples) >= max_sentences:
                break
            text = item.get("text", "").strip()
            for sent in text.replace("。", ".").split("."):
                sent = sent.strip()
                if 25 <= len(sent) <= 200:
                    samples.append({"text": sent, "source": "wiki_ko", "stage": 2})
                if len(samples) >= max_sentences:
                    break
        random.shuffle(samples)
        logger.info(f"✅ Korean wiki: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️  Korean wiki 실패: {e}")
    return samples


def load_replay(ratio: float = 0.15, max_per_stage: int = 5000) -> list[dict]:
    """Stage 0+1 데이터 리플레이 — 망각 방지."""
    samples = []
    for path, stage_name in [(STAGE0_TRAIN, "stage0"), (STAGE1_TRAIN, "stage1")]:
        if not path.exists():
            continue
        lines = path.read_text().strip().split("\n")
        random.shuffle(lines)
        n = min(max_per_stage, int(len(lines) * ratio))
        for line in lines[:n]:
            d = json.loads(line)
            d["source"] = f"{stage_name}_replay"
            d["stage"] = 2
            samples.append(d)
    logger.info(f"✅ Stage 0+1 리플레이: {len(samples):,}건")
    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 저장: {len(samples):,}건 → {path}")


def main():
    logger.info("=" * 60)
    logger.info("   EurekaAI — Stage 2: Elementary Data Preparation")
    logger.info("=" * 60)
    logger.info("목표 데이터 비율:")
    logger.info("  Teacher Q&A (Ollama)     : ~25%  ← 핵심!")
    logger.info("  TinyStories (영문 기초)   : ~30%")
    logger.info("  Korean Wikipedia         : ~20%")
    logger.info("  Stage 0+1 Replay         : ~15%")
    logger.info("  Seed / Synthetic         : ~10%")

    all_samples = []

    # 1. Teacher Q&A — Stage 2 핵심! (시간이 걸림)
    all_samples += generate_teacher_qa(n_articles=150)

    # 2. TinyStories
    all_samples += load_tinystories(max_samples=12000)

    # 3. Korean Wikipedia
    all_samples += load_korean_wiki(max_sentences=8000)

    # 4. Stage 0+1 리플레이 (망각 방지)
    all_samples += load_replay(ratio=0.15, max_per_stage=4000)

    # 5. Ollama 합성 데이터
    all_samples += generate_synthetic(n=300)

    # 6. Seed 텍스트
    all_samples += generate_seed_corpus(n_repeat=8)

    # 중복 제거 + 셔플
    seen = set()
    deduped = []
    for s in all_samples:
        key = s["text"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    random.shuffle(deduped)
    all_samples = deduped

    from collections import Counter
    total = len(all_samples)
    logger.info(f"\n📊 최종 데이터 구성 ({total:,}건):")
    for src, cnt in Counter(s["source"] for s in all_samples).most_common():
        logger.info(f"   {src:25s}: {cnt:6,} ({cnt/total*100:.1f}%)")

    n_eval = min(TARGET_EVAL, int(total * 0.1))
    random.shuffle(all_samples)
    eval_samples  = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    save_jsonl(train_samples, TRAIN_FILE)
    save_jsonl(eval_samples,  EVAL_FILE)

    logger.info(f"\n✅ Stage 2 데이터 준비 완료:")
    logger.info(f"   Train: {len(train_samples):,} → {TRAIN_FILE}")
    logger.info(f"   Eval:  {len(eval_samples):,}  → {EVAL_FILE}")


if __name__ == "__main__":
    main()
