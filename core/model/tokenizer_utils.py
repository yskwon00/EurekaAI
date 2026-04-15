"""
EurekaAI — Tokenizer Utilities
Manages training and loading of the bilingual (Ko+En) BPE tokenizer.
Uses SentencePiece for robust Korean subword tokenization.
"""

import os
import io
import json
from pathlib import Path
from typing import Optional, Union

import sentencepiece as spm


# ── Special tokens ─────────────────────────────────────────────────────────────
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]

BOS_ID = 1
EOS_ID = 2
PAD_ID = 0
UNK_ID = 3


# ── EurekaTokenizer ──────────────────────────────────────────────────────────

class EurekaTokenizer:
    """
    BPE tokenizer for EurekaAI — Korean + English bilingual.
    Wraps SentencePiece with convenience methods.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            self.sp.Load(model_path)
            self._setup_special_ids()

    def _setup_special_ids(self):
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else BOS_ID
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else EOS_ID
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else PAD_ID
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() >= 0 else UNK_ID
        self.vocab_size = self.sp.GetPieceSize()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> list[int]:
        """Encode text to token ids."""
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids to text."""
        if skip_special:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.DecodeIds(ids)

    def tokenize(self, text: str) -> list[str]:
        """Return token strings (for debugging)."""
        return self.sp.EncodeAsPieces(text)

    def batch_encode(
        self,
        texts: list[str],
        max_length: int = 512,
        padding: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> dict:
        """Encode a batch of texts with padding."""
        encoded = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            for t in texts
        ]
        if not padding:
            return {"input_ids": encoded}

        max_len = max(len(e) for e in encoded)
        padded = []
        masks = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            padded.append(ids + [self.pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": padded,
            "attention_mask": masks,
        }

    @property
    def is_loaded(self) -> bool:
        return self.sp.GetPieceSize() > 0

    def __len__(self) -> int:
        return self.vocab_size


# ── Tokenizer Training ──────────────────────────────────────────────────────────

def train_tokenizer(
    text_files: list[str],
    output_dir: str = "data/tokenizer",
    vocab_size: int = 32000,
    model_prefix: str = "eureka",
    character_coverage_ko: float = 0.9995,
) -> EurekaTokenizer:
    """
    Train a BPE tokenizer on Korean + English text files.

    Args:
        text_files:    List of .txt file paths for training data
        output_dir:    Where to save tokenizer model
        vocab_size:    BPE vocabulary size (default 32000)
        model_prefix:  Output model filename prefix

    Returns:
        Trained EurekaTokenizer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = str(output_path / model_prefix)

    print(f"🔤 Training BPE tokenizer (vocab={vocab_size}) on {len(text_files)} files...")

    # Count corpus lines to auto-adjust vocab_size if corpus is too small
    total_lines = 0
    for f in text_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                total_lines += sum(1 for line in fh if line.strip())
        except Exception:
            pass

    # SentencePiece needs ~4x lines to support a given vocab size
    max_safe_vocab = max(500, min(vocab_size, total_lines * 4))
    if max_safe_vocab < vocab_size:
        print(f"⚠️  Corpus too small ({total_lines} lines). Reducing vocab_size: {vocab_size} → {max_safe_vocab}")
        vocab_size = max_safe_vocab

    # SentencePiece training args
    train_args = {
        "input": ",".join(text_files),
        "model_prefix": model_path,
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "pad_piece": PAD_TOKEN,
        "bos_piece": BOS_TOKEN,
        "eos_piece": EOS_TOKEN,
        "unk_piece": UNK_TOKEN,
        "user_defined_symbols": MASK_TOKEN,
        "character_coverage": character_coverage_ko,
        "num_threads": os.cpu_count(),
        "shuffle_input_sentence": True,
        "input_sentence_size": 2_000_000,
        "byte_fallback": True,
    }

    spm.SentencePieceTrainer.Train(**train_args)

    tokenizer_model_path = model_path + ".model"
    print(f"✅ Tokenizer saved: {tokenizer_model_path}")

    tok = EurekaTokenizer(tokenizer_model_path)
    print(f"   vocab_size={tok.vocab_size}")
    return tok


def prepare_tokenizer_corpus(output_file: str = "data/tokenizer/train_corpus.txt"):
    """
    Download and prepare a small Ko+En corpus for tokenizer training.
    Uses freely available datasets.
    """
    import os
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # 1. English: TinyStories (HuggingFace)
    try:
        from datasets import load_dataset
        print("📥 Loading English corpus (TinyStories)...")
        en_ds = load_dataset("roneneldan/TinyStories", split="train[:8000]", trust_remote_code=True)
        for item in en_ds:
            text = item.get("text", "").strip()
            # Split long stories into sentences for better coverage
            for sent in text.split("."):
                sent = sent.strip()
                if len(sent) > 10:
                    lines.append(sent)
        print(f"   {len(lines):,} English sentences loaded")
    except Exception as e:
        print(f"   ⚠️  TinyStories load failed: {e}")

    # 2. Korean: Wikipedia (HuggingFace)
    try:
        from datasets import load_dataset
        ko_start = len(lines)
        print("📥 Loading Korean corpus (Wikipedia)...")
        ko_ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ko",
            split="train[:5000]",
            trust_remote_code=True,
        )
        for item in ko_ds:
            text = item.get("text", "").strip()
            # Split into sentences
            for sent in text.replace("。", ".").split("."):
                sent = sent.strip()
                if len(sent) > 5:
                    lines.append(sent[:300])
        print(f"   {len(lines) - ko_start:,} Korean sentences loaded")
    except Exception as e:
        print(f"   ⚠️  Korean wiki load failed: {e}")

    # 3. Built-in seed lines (always appended for basic coverage)
    seed_lines = [
        "안녕하세요. 저는 EurekaAI입니다.",
        "Hello, I am EurekaAI, a self-learning AI model.",
        "학습은 경험을 통해 이루어집니다.",
        "Learning happens through experience and curiosity.",
        "하늘은 파랗고 바다는 넓습니다.",
        "The sky is blue and the ocean is vast.",
        "가나다라마바사아자차카타파하",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
        "사과는 빨갛고 바나나는 노랗습니다.",
        "봄 여름 가을 겨울 Spring Summer Autumn Winter",
    ] * 50
    lines.extend(seed_lines)

    # 4. Fallback if download totally failed
    if len(lines) < 500:
        print("   ⚠️  Downloads failed, using extended seed corpus for small vocab")
        lines = seed_lines * 200

    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\n", " ").strip() + "\n")

    print(f"✅ Tokenizer corpus: {len(lines):,} lines → {output_file}")
    return output_file
