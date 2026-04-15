"""
EurekaAI — Curriculum Data Manager
Handles per-stage data loading, preprocessing, and tokenization.
Supports JSONL format used across all stages.
"""

import json
import random
import logging
from pathlib import Path
from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from ..model.tokenizer_utils import EurekaTokenizer

logger = logging.getLogger(__name__)


# ── Dataset Format ─────────────────────────────────────────────────────────────
# Each JSONL line = one of:
#   {"text": "..."} — raw text for CLM
#   {"instruction": "...", "response": "..."} — instruction following
#   {"question": "...", "answer": "...", "chain_of_thought": "..."} — CoT QA


# ── Base Dataset ────────────────────────────────────────────────────────────────

class EurekaDataset(Dataset):
    """
    PyTorch Dataset that reads JSONL files and tokenizes on the fly.
    Supports: CLM (text), Instruction (instruction+response), QA (question+answer+CoT)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: EurekaTokenizer,
        max_seq_len: int = 512,
        mode: str = "clm",       # "clm" | "instruction" | "qa"
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.samples = []

        self._load(data_path, max_samples)

    def _load(self, path: str, max_samples: Optional[int]):
        path = Path(path)
        if not path.exists():
            logger.warning(f"Data path not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.samples)} samples from {path}")

    def _format_sample(self, sample: dict) -> str:
        """Convert sample dict to training text."""
        if self.mode == "clm" or "text" in sample:
            return sample.get("text", "")

        if self.mode == "instruction" or "instruction" in sample:
            inst = sample.get("instruction", sample.get("question", ""))
            resp = sample.get("response", sample.get("answer", ""))
            cot = sample.get("chain_of_thought", "")
            if cot:
                return f"### 질문\n{inst}\n\n### 풀이\n{cot}\n\n### 답\n{resp}"
            return f"### 질문\n{inst}\n\n### 답\n{resp}"

        return str(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text = self._format_sample(self.samples[idx])
        ids = self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
            max_length=self.max_seq_len,
        )
        input_ids = torch.tensor(ids, dtype=torch.long)

        # Labels = input_ids shifted (standard CLM)
        labels = input_ids.clone()

        # Pad/truncate to max_seq_len
        pad_id = self.tokenizer.pad_id
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])  # ignore pad in loss

        return {
            "input_ids": input_ids[:self.max_seq_len],
            "labels": labels[:self.max_seq_len],
            "attention_mask": (input_ids[:self.max_seq_len] != pad_id).long(),
        }


# ── Multi-file Dataset ─────────────────────────────────────────────────────────

class MultiFileDataset(Dataset):
    """Merges multiple JSONL files into a single dataset."""

    def __init__(
        self,
        data_paths: list[str],
        tokenizer: EurekaTokenizer,
        max_seq_len: int = 512,
        mode: str = "clm",
        shuffle: bool = True,
    ):
        parts = [
            EurekaDataset(p, tokenizer, max_seq_len, mode)
            for p in data_paths
            if Path(p).exists()
        ]
        self.samples = []
        for ds in parts:
            self.samples.extend(ds.samples)

        if shuffle:
            random.shuffle(self.samples)

        # Store for __getitem__
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._mode = mode
        self._base = EurekaDataset.__new__(EurekaDataset)
        self._base.tokenizer = tokenizer
        self._base.max_seq_len = max_seq_len
        self._base.mode = mode
        self._base.samples = self.samples
        logger.info(f"MultiFileDataset: {len(self.samples)} total samples from {len(parts)} files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._base[idx]


# ── Collate Function ────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """Stack batch items with dynamic padding."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    pad_id = 0

    input_ids_list, labels_list, masks_list = [], [], []
    for b in batch:
        L = b["input_ids"].size(0)
        pad = max_len - L
        input_ids_list.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
        labels_list.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        masks_list.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(masks_list),
    }


# ── Data Manager ────────────────────────────────────────────────────────────────

class CurriculumDataManager:
    """
    Central manager for curriculum data across all stages.
    Handles:
      - Loading stage-specific data files
      - Building train/eval DataLoaders
      - Splitting train/eval
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        tokenizer: Optional[EurekaTokenizer] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer

    def get_stage_files(self, stage: int) -> dict[str, list[str]]:
        """Return train/eval data file paths for a given stage."""
        stage_dir = self.data_dir / f"stage{stage}"
        train_files = sorted(str(p) for p in stage_dir.glob("train_*.jsonl"))
        eval_files = sorted(str(p) for p in stage_dir.glob("eval_*.jsonl"))

        if not train_files:
            # Fallback: look for any .jsonl in stage dir
            all_files = sorted(str(p) for p in stage_dir.glob("*.jsonl"))
            split = max(1, int(len(all_files) * 0.9))
            train_files = all_files[:split]
            eval_files = all_files[split:]

        return {"train": train_files, "eval": eval_files}

    def build_dataloaders(
        self,
        stage: int,
        config,           # EurekaConfig
        mode: str = "clm",
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Build train and eval DataLoaders for a stage."""
        files = self.get_stage_files(stage)

        if not files["train"]:
            raise FileNotFoundError(
                f"No training data for stage {stage}. "
                f"Please run: python stages/stage{stage}_*/data_prep.py"
            )

        train_ds = MultiFileDataset(
            files["train"], self.tokenizer,
            max_seq_len=config.max_seq_len,
            mode=mode,
            shuffle=True,
        )

        eval_ds = None
        if files["eval"]:
            eval_ds = MultiFileDataset(
                files["eval"], self.tokenizer,
                max_seq_len=config.max_seq_len,
                mode=mode,
                shuffle=False,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,          # 0 for Mac MPS compatibility
            collate_fn=collate_fn,
            pin_memory=False,       # pin_memory=False for MPS
        )

        eval_loader = None
        if eval_ds:
            eval_loader = DataLoader(
                eval_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

        return train_loader, eval_loader

    def save_jsonl(self, samples: list[dict], path: str):
        """Save samples to JSONL file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(samples)} samples → {output}")

    def train_eval_split(
        self,
        samples: list[dict],
        eval_ratio: float = 0.1,
        shuffle: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        """Split samples into train/eval."""
        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)
        n_eval = max(1, int(len(samples) * eval_ratio))
        return samples[n_eval:], samples[:n_eval]
