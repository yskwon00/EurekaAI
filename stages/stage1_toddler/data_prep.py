"""
EurekaAI — Stage 1 (Toddler) Data Preparation
유아기 수준 데이터: 간단한 단어, 일상 표현, 동화책 스타일
  - Stage 0보다 문장이 길고 다양한 어휘 사용
  - 단문 위주 (문장당 10단어 이하)
  - 한국어 + 영어 병행
"""

import sys
import json
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.teacher.ollama_teacher import OllamaTeacher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGE_DIR = Path("data/processed/stage1")
TRAIN_FILE = STAGE_DIR / "train_stage1.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage1.jsonl"
TARGET_EVAL = 2000

# Stage 0 데이터 경로 (리플레이용)
STAGE0_TRAIN = Path("data/processed/stage0/train_stage0.jsonl")


# ── Stage 1 Seed Texts ────────────────────────────────────────────────────────

KO_TODDLER = [
    "오늘은 날씨가 맑아요.",
    "엄마가 밥을 지어요.",
    "아빠는 회사에 가요.",
    "나는 사과를 먹어요.",
    "강아지가 공을 가지고 놀아요.",
    "비가 오면 우산을 써요.",
    "학교에서 친구를 만나요.",
    "책을 읽으면 똑똑해져요.",
    "밤에는 달이 빛나요.",
    "아침에 일어나면 세수해요.",
    "고양이는 생선을 좋아해요.",
    "나비가 꽃 위에 앉았어요.",
    "토끼는 깡충깡충 뛰어요.",
    "나는 빨간 사과를 좋아해요.",
    "하늘에 구름이 떠 있어요.",
    "우리 가족은 다 함께 밥을 먹어요.",
    "선생님이 그림을 그려주셨어요.",
    "봄이 되면 꽃이 피어요.",
    "여름에는 수박을 먹어요.",
    "가을에는 낙엽이 떨어져요.",
    "겨울에는 눈사람을 만들어요.",
    "강물이 졸졸 흘러가요.",
    "새들이 노래를 불러요.",
    "아이들이 운동장에서 놀아요.",
    "할머니가 맛있는 떡을 만드셨어요.",
]

EN_TODDLER = [
    "The dog likes to run in the park.",
    "Mom is cooking dinner in the kitchen.",
    "The bird sings a pretty song.",
    "We go to school every morning.",
    "The cat drinks milk from a bowl.",
    "Children love to play outside.",
    "The sun sets in the evening.",
    "I brush my teeth before bed.",
    "Grandma tells us bedtime stories.",
    "The rabbit hops around the garden.",
    "Fish swim in the clear water.",
    "We plant flowers in the spring.",
    "The bear sleeps in winter.",
    "My friends and I like to draw.",
    "The bus takes us to school.",
    "Apples grow on trees.",
    "The moon shines at night.",
    "We wash our hands before eating.",
    "The puppy wags its tail happily.",
    "Frogs jump into the pond.",
    "We share our toys with friends.",
    "The farmer grows vegetables.",
    "Butterflies are very colorful.",
    "We listen to music together.",
    "Rain makes the flowers grow.",
]

MIXED_TODDLER = [
    "나는 happy해요. I am happy today!",
    "사과는 apple이에요. 바나나는 banana예요.",
    "One 하나, Two 둘, Three 셋! Let's count together.",
    "강아지 dog, 고양이 cat, 새 bird. 나는 동물을 좋아해요.",
    "봄 spring, 여름 summer, 가을 autumn, 겨울 winter.",
    "Good morning! 안녕하세요! 오늘도 좋은 하루예요.",
    "I love 음식. My favorite food is 김치!",
    "Red 빨강, Blue 파랑, Yellow 노랑, Green 초록.",
    "Dad 아빠, Mom 엄마, Brother 오빠, Sister 언니.",
    "나는 school에 가요. I go to school every day.",
]


def generate_seed_corpus(n_repeat: int = 15) -> list[dict]:
    samples = []
    for text in (KO_TODDLER + EN_TODDLER + MIXED_TODDLER):
        for _ in range(n_repeat):
            samples.append({"text": text, "source": "seed_toddler", "stage": 1})
    logger.info(f"✅ Seed corpus: {len(samples):,}건")
    return samples


def load_tinystories(max_samples: int = 20000) -> list[dict]:
    """TinyStories — 동화책 스타일 영문 (Stage 1 핵심)"""
    try:
        from datasets import load_dataset
        logger.info(f"📥 TinyStories (Stage 1용, {max_samples:,}건)...")
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{max_samples}]")
        samples = []
        for item in ds:
            text = item.get("text", "").strip()
            if not text:
                continue
            # Stage 1: 단락 전체 사용 (Stage 0보다 긴 문장 허용)
            if 30 <= len(text) <= 800:
                samples.append({"text": text[:500], "source": "tinystories", "stage": 1})
            else:
                # 긴 텍스트는 단락으로 분리
                for para in text.split("\n"):
                    para = para.strip()
                    if 30 <= len(para) <= 400:
                        samples.append({"text": para, "source": "tinystories", "stage": 1})
        logger.info(f"✅ TinyStories: {len(samples):,}건")
        return samples
    except Exception as e:
        logger.warning(f"⚠️ TinyStories 실패: {e}")
        return []


def load_korean_wiki(max_sentences: int = 15000) -> list[dict]:
    """한국어 위키 — Stage 1은 중간 난이도 문장 (20~150자)"""
    samples = []
    try:
        from datasets import load_dataset
        logger.info(f"📥 Korean Wikipedia (최대 {max_sentences:,}문장)...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:500]")
        for item in ko_ds:
            if len(samples) >= max_sentences:
                break
            text = item.get("text", "").strip()
            for sent in text.replace("。", ".").split("."):
                sent = sent.strip()
                if 20 <= len(sent) <= 150:
                    samples.append({"text": sent, "source": "wiki_ko", "stage": 1})
                if len(samples) >= max_sentences:
                    break
        random.shuffle(samples)
        logger.info(f"✅ Korean wiki: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️ Korean wiki 실패: {e}")
    return samples


def load_stage0_replay(ratio: float = 0.2, max_samples: int = 10000) -> list[dict]:
    """Stage 0 데이터 리플레이 — 망각 방지 (Continual Learning)"""
    if not STAGE0_TRAIN.exists():
        logger.warning("⚠️ Stage 0 데이터 없음 — 리플레이 스킵")
        return []
    samples = []
    with open(STAGE0_TRAIN) as f:
        all_lines = f.readlines()
    random.shuffle(all_lines)
    n = min(max_samples, int(len(all_lines) * ratio))
    for line in all_lines[:n]:
        d = json.loads(line)
        d["source"] = "stage0_replay"
        d["stage"] = 1
        samples.append(d)
    logger.info(f"✅ Stage 0 리플레이: {len(samples):,}건 (망각 방지)")
    return samples


def generate_synthetic(n: int = 500) -> list[dict]:
    """Ollama Teacher로 유아기 수준 합성 데이터 생성"""
    teacher = OllamaTeacher()
    if not teacher.is_available():
        logger.warning("⚠️ Ollama 미사용 — 합성 데이터 스킵")
        return []
    logger.info(f"🤖 Ollama로 Stage 1 합성 데이터 {n}건 생성...")
    raw = teacher.generate_synthetic_stage_data(stage=1, n_samples=n, language="mixed")
    samples = [{**s, "source": "ollama_synthetic", "stage": 1} for s in raw if s.get("text", "").strip()]
    logger.info(f"✅ 합성 데이터: {len(samples)}건")
    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 저장: {len(samples):,}건 → {path}")


def main():
    logger.info("=" * 60)
    logger.info("   EurekaAI — Stage 1: Toddler Data Preparation")
    logger.info("=" * 60)
    logger.info("목표 데이터 비율:")
    logger.info("  TinyStories (동화 스타일)  : ~40%")
    logger.info("  Korean Wikipedia (중간)    : ~25%")
    logger.info("  Stage 0 Replay (망각 방지) : ~20%")
    logger.info("  Seed / Synthetic          : ~15%")

    all_samples = []

    # 1. TinyStories (핵심: 다양한 동화 스타일)
    all_samples += load_tinystories(max_samples=15000)

    # 2. Korean Wikipedia (중간 난이도)
    all_samples += load_korean_wiki(max_sentences=12000)

    # 3. Stage 0 리플레이 (망각 방지 - Continual Learning 핵심!)
    all_samples += load_stage0_replay(ratio=0.2, max_samples=8000)

    # 4. Ollama 합성 데이터
    all_samples += generate_synthetic(n=500)

    # 5. Seed 텍스트
    all_samples += generate_seed_corpus(n_repeat=10)

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

    # Train / Eval 분리
    n_eval = min(TARGET_EVAL, int(total * 0.1))
    random.shuffle(all_samples)
    eval_samples  = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    save_jsonl(train_samples, TRAIN_FILE)
    save_jsonl(eval_samples,  EVAL_FILE)

    logger.info(f"\n✅ Stage 1 데이터 준비 완료:")
    logger.info(f"   Train: {len(train_samples):,} → {TRAIN_FILE}")
    logger.info(f"   Eval:  {len(eval_samples):,}  → {EVAL_FILE}")


if __name__ == "__main__":
    main()
