"""
EurekaAI — Tokenizer Setup Script
Builds the bilingual Ko+En BPE tokenizer from a small corpus.
Run this ONCE before any training begins.

Usage:
    python scripts/setup_tokenizer.py
    python scripts/setup_tokenizer.py --vocab-size 32000
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model.tokenizer_utils import (
    EurekaTokenizer,
    train_tokenizer,
    prepare_tokenizer_corpus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Setup EurekaAI tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-dir", type=str, default="data/tokenizer")
    parser.add_argument("--corpus", type=str, default="data/tokenizer/train_corpus.txt")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("  EurekaAI — Tokenizer Setup")
    logger.info("=" * 50)

    tok_model = Path(args.output_dir) / "eureka.model"
    if tok_model.exists():
        logger.info(f"✅ Tokenizer already exists: {tok_model}")
        tok = EurekaTokenizer(str(tok_model))
        logger.info(f"   vocab_size = {tok.vocab_size}")

        # Quick test
        test_cases = [
            "안녕하세요! 저는 EurekaAI 입니다.",
            "Hello! I am EurekaAI, a self-learning model.",
            "1 + 1 = 2, 사과 apple, 하늘 sky",
        ]
        logger.info("\n🔤 Tokenizer test:")
        for text in test_cases:
            tokens = tok.tokenize(text)
            decoded = tok.decode(tok.encode(text))
            logger.info(f"  [{text[:40]}]")
            logger.info(f"   tokens: {tokens[:10]}...")
            logger.info(f"   decoded: {decoded[:50]}")
        return

    # Prepare corpus if not already available
    if not Path(args.corpus).exists():
        logger.info(f"Corpus not found at {args.corpus}. Downloading...")
        prepare_tokenizer_corpus(args.corpus)

    # Train tokenizer
    tok = train_tokenizer(
        text_files=[args.corpus],
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_prefix="eureka",
    )

    # Verify
    logger.info("\n🔤 Tokenizer verification:")
    test_cases = [
        "안녕하세요! 저는 EurekaAI 입니다.",
        "Hello! I am EurekaAI, a self-learning model.",
        "가나다라마바사 / ABCDEFG / 1234567",
    ]
    for text in test_cases:
        tokens = tok.tokenize(text)
        decoded = tok.decode(tok.encode(text))
        logger.info(f"  [{text[:40]}] → {len(tokens)} tokens")

    logger.info("\n✅ Tokenizer setup complete!")
    logger.info(f"   Model: {Path(args.output_dir) / 'eureka.model'}")
    logger.info(f"   Vocab: {tok.vocab_size:,} tokens")


if __name__ == "__main__":
    main()
