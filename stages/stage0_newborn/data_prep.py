"""
EurekaAI — Stage 0 (Newborn) Data Preparation
Generates a large bilingual corpus for character-pattern learning.
Sources (priority order):
  1. Tokenizer corpus (43MB, already downloaded Ko+En sentences) ← PRIMARY
  2. TinyStories (English, HuggingFace)
  3. Korean Wikipedia (HuggingFace)
  4. Ollama synthetic generation
  5. Built-in seed patterns (always available fallback)
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

# ── Output paths ────────────────────────────────────────────────────────────────
STAGE_DIR = Path("data/processed/stage0")
TRAIN_FILE = STAGE_DIR / "train_stage0.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage0.jsonl"
TARGET_TRAIN = 20000   # <<< 십배 이상 늘림 (과적합 방지)
TARGET_EVAL  = 2000


# ── Built-in Seed Texts ─────────────────────────────────────────────────────────

KO_SEED_TEXTS = [
    "가나다라마바사아자차카타파하",
    "아이우에오",
    "나비야 나비야 이리 날아 오너라.",
    "작은 별 반짝반짝 하늘에서 빛나네.",
    "안녕하세요. 반갑습니다.",
    "엄마, 아빠, 나는 사랑해요.",
    "하늘은 파랗고 땅은 넓어요.",
    "해가 뜨면 일어나고 달이 뜨면 자요.",
    "강아지가 멍멍 짖어요.",
    "고양이가 야옹 울어요.",
    "사과는 빨갛고 바나나는 노래요.",
    "봄에는 꽃이 피고 겨울에는 눈이 와요.",
    "물을 마시고 밥을 먹어요.",
    "학교에 가서 공부해요.",
    "친구들과 함께 놀아요.",
    "나는 매일 책을 읽어요.",
    "아침에 일어나서 이를 닦아요.",
    "밤에는 별이 반짝반짝 빛나요.",
    "산에는 나무가 많아요.",
    "바다에는 물고기가 살아요.",
]

EN_SEED_TEXTS = [
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "Twinkle twinkle little star, how I wonder what you are.",
    "Mary had a little lamb whose fleece was white as snow.",
    "Jack and Jill went up the hill to fetch a pail of water.",
    "Hello! My name is Eureka.",
    "The sun rises in the morning.",
    "The cat sat on the mat.",
    "Dogs are friendly animals.",
    "I like to eat apples and bananas.",
    "The sky is blue and the grass is green.",
    "Birds can fly high in the sky.",
    "Water flows down the river to the sea.",
    "Children laugh and play every day.",
    "Books help us learn new things.",
    "The moon shines at night.",
    "Stars are very far away.",
    "Flowers bloom in spring.",
    "Snow falls in winter.",
    "I love my family very much.",
    "We go to school to learn.",
]

MIXED_PATTERNS = [
    "Hello 안녕! Good morning 좋은 아침!",
    "사과 is apple. 바나나 is banana. 고양이 is cat.",
    "I love 음악. Music is 아름다워요.",
    "One 하나, Two 둘, Three 셋, Four 넷, Five 다섯.",
    "봄 Spring, 여름 Summer, 가을 Autumn, 겨울 Winter",
    "Happy 행복해요. Sad 슬퍼요. Angry 화나요.",
    "Mom 엄마, Dad 아빠, Friend 친구, Teacher 선생님",
]


# ── Data Sources ────────────────────────────────────────────────────────────────

def generate_seed_corpus(n_repeat: int = 10) -> list[dict]:
    """Generate seed corpus from built-in texts."""
    samples = []
    for text in (KO_SEED_TEXTS + EN_SEED_TEXTS + MIXED_PATTERNS):
        for _ in range(n_repeat):
            samples.append({"text": text, "source": "seed", "stage": 0})
    logger.info(f"✅ Built-in seed corpus: {len(samples)} samples")
    return samples


def load_from_tokenizer_corpus(
    corpus_path: str = "data/tokenizer/train_corpus.txt",
    max_samples: int = 15000,
    min_len: int = 20,
    max_len: int = 300,
) -> list[dict]:
    """
    PRIMARY source: Re-use the 43MB tokenizer corpus.
    Already contains clean Ko+En sentences — perfect for Stage 0.
    """
    samples = []
    path = Path(corpus_path)
    if not path.exists():
        logger.warning(f"⚠️  Tokenizer corpus not found: {corpus_path}")
        return []

    indices = []
    logger.info(f"📥 Scanning tokenizer corpus: {corpus_path} ...")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if min_len <= len(l.strip()) <= max_len]

    random.shuffle(lines)
    for line in lines[:max_samples]:
        samples.append({"text": line, "source": "tokenizer_corpus", "stage": 0})

    logger.info(f"✅ Tokenizer corpus: {len(samples):,} samples (from {len(lines):,} total lines)")
    return samples


def download_tiny_stories(max_samples: int = 3000) -> list[dict]:
    """Download English TinyStories — all lengths accepted."""
    try:
        from datasets import load_dataset
        logger.info("📥 Downloading TinyStories (English)...")
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{max_samples}]")
        samples = []
        for item in ds:
            text = item.get("text", "").strip()
            if text:
                # Split long stories into shorter segments for Stage 0
                for seg in text.split("\n"):
                    seg = seg.strip()
                    if 20 <= len(seg) <= 500:
                        samples.append({"text": seg[:300], "source": "tinystories", "stage": 0})
        logger.info(f"✅ TinyStories: {len(samples):,} segments")
        return samples
    except Exception as e:
        logger.warning(f"⚠️  TinyStories failed: {e}")
        return []


def download_korean_texts(max_samples: int = 2000) -> list[dict]:
    """Download Korean Wikipedia texts."""
    samples = []
    try:
        from datasets import load_dataset
        logger.info("📥 Downloading Korean Wikipedia...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split=f"train[:{max_samples}]")
        for item in ko_ds:
            text = item.get("text", "").strip()
            # Split into sentences
            for sent in text.replace("。", ".").split("."):
                sent = sent.strip()
                if 15 <= len(sent) <= 300:
                    samples.append({"text": sent, "source": "wiki_ko", "stage": 0})
        logger.info(f"✅ Korean wiki sentences: {len(samples):,}")
    except Exception as e:
        logger.warning(f"⚠️  Korean corpus failed: {e}")
    return samples


def generate_synthetic_samples(n: int = 300) -> list[dict]:
    """Generate synthetic Stage 0 data using Ollama teacher."""
    teacher = OllamaTeacher()
    if not teacher.is_available():
        logger.warning("⚠️  Ollama not available — skipping synthetic generation")
        return []
    logger.info(f"🤖 Generating {n} synthetic Stage 0 samples via Ollama...")
    samples_raw = teacher.generate_synthetic_stage_data(stage=0, n_samples=n, language="mixed")
    samples = [
        {**s, "source": "ollama_synthetic", "stage": 0}
        for s in samples_raw
        if s.get("text", "").strip()
    ]
    logger.info(f"✅ Synthetic samples: {len(samples)}")
    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 Saved {len(samples):,} samples → {path}")


def main():
    logger.info("=" * 55)
    logger.info("   EurekaAI — Stage 0: Data Preparation (v2)")
    logger.info("=" * 55)

    all_samples = []

    # 1. PRIMARY: Tokenizer corpus (43MB, Ko+En sentences)
    all_samples += load_from_tokenizer_corpus(max_samples=15000)

    # 2. TinyStories (English)
    all_samples += download_tiny_stories(max_samples=3000)

    # 3. Korean Wikipedia
    all_samples += download_korean_texts(max_samples=2000)

    # 4. Ollama synthetic
    all_samples += generate_synthetic_samples(n=300)

    # 5. Built-in seed (small, always available)
    all_samples += generate_seed_corpus(n_repeat=5)

    # Deduplicate + shuffle
    seen = set()
    deduped = []
    for s in all_samples:
        key = s["text"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    random.shuffle(deduped)
    all_samples = deduped

    logger.info(f"\n📊 Total unique samples: {len(all_samples):,}")
    from collections import Counter
    for src, cnt in Counter(s["source"] for s in all_samples).most_common():
        logger.info(f"   {src:25s}: {cnt:6,}")

    # Train / eval split (10%)
    n_eval = min(TARGET_EVAL, int(len(all_samples) * 0.1))
    eval_samples  = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    save_jsonl(train_samples, TRAIN_FILE)
    save_jsonl(eval_samples,  EVAL_FILE)

    logger.info(f"\n✅ Stage 0 data ready:")
    logger.info(f"   Train: {len(train_samples):,} → {TRAIN_FILE}")
    logger.info(f"   Eval:  {len(eval_samples):,}  → {EVAL_FILE}")


if __name__ == "__main__":
    main()
