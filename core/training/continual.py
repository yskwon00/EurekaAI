"""
EurekaAI — Continual Learning
Implements two complementary strategies to prevent catastrophic forgetting:

1. EWC (Elastic Weight Consolidation)
   - Identifies important parameters after each stage
   - Adds regularization penalty to protect them in subsequent stages

2. ReplayBuffer
   - Stores a small subset of previous stage data
   - Mixes replayed samples into current stage training
"""

import random
import logging
from pathlib import Path
from typing import Optional
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

logger = logging.getLogger(__name__)


# ── EWC (Elastic Weight Consolidation) ─────────────────────────────────────────

class EWC:
    """
    EWC regularization: after completing a stage, computes Fisher Information
    Matrix for important parameters and penalizes large changes in future stages.

    Usage:
        # After stage N training:
        ewc = EWC(model, dataloader, device, lambda_=5000.0)
        ewc.compute_fisher()
        ewc.save("checkpoints/stage0/ewc.pt")

        # During stage N+1 training:
        ewc = EWC.load("checkpoints/stage0/ewc.pt", model, device)
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        lambda_: float = 5000.0,
        n_samples: int = 200,
    ):
        self.model = model
        self.device = device
        self.lambda_ = lambda_
        self.n_samples = n_samples

        # Will be populated by compute_fisher()
        self.fisher: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}
        self.dataloader = dataloader

    def compute_fisher(self):
        """
        Compute diagonal Fisher Information Matrix (approximation).
        Run after completing each stage before moving to the next.
        """
        logger.info("⚙️  Computing Fisher Information Matrix for EWC...")
        self.model.eval()

        # Save current optimal parameters (θ*)
        self.optimal_params = {
            n: p.data.clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Initialize Fisher accumulators
        fisher_accum = {
            n: torch.zeros_like(p.data)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        n_processed = 0
        for batch in self.dataloader:
            if n_processed >= self.n_samples:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            self.model.zero_grad()
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_accum[n] += p.grad.data.pow(2)

            n_processed += input_ids.size(0)

        # Average Fisher over samples
        n_batches = max(1, n_processed // self.dataloader.batch_size)
        self.fisher = {n: f / n_batches for n, f in fisher_accum.items()}

        logger.info(f"✅ Fisher computed on {n_processed} samples")

    def penalty(self, current_model: nn.Module) -> torch.Tensor:
        """
        EWC penalty term: λ/2 * Σ F_i * (θ_i - θ*_i)²
        Add to task loss during training.
        """
        if not self.fisher:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for n, p in current_model.named_parameters():
            if n in self.fisher and n in self.optimal_params:
                fisher = self.fisher[n].to(self.device)
                opt = self.optimal_params[n].to(self.device)
                loss += (fisher * (p - opt).pow(2)).sum()

        return (self.lambda_ / 2.0) * loss

    def save(self, path: str):
        """Save EWC state to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "fisher": {n: f.cpu() for n, f in self.fisher.items()},
            "optimal_params": {n: p.cpu() for n, p in self.optimal_params.items()},
            "lambda_": self.lambda_,
        }, path)
        logger.info(f"💾 EWC state saved: {path}")

    @classmethod
    def load(
        cls,
        path: str,
        model: nn.Module,
        device: torch.device,
    ) -> "EWC":
        """Load EWC state from disk."""
        state = torch.load(path, map_location="cpu")
        ewc = cls(model=model, device=device, lambda_=state["lambda_"])
        ewc.fisher = {n: f.to(device) for n, f in state["fisher"].items()}
        ewc.optimal_params = {n: p.to(device) for n, p in state["optimal_params"].items()}
        logger.info(f"📂 EWC state loaded: {path}")
        return ewc


# ── Replay Buffer ───────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay: stores samples from completed stages
    and mixes them into future stage training data.

    Usage:
        buffer = ReplayBuffer(max_size=5000)
        buffer.add_dataset(stage0_dataset, stage=0)
        buffer.add_dataset(stage1_dataset, stage=1)

        # During stage 2 training:
        replay_dataset = buffer.sample_dataset(n=1000)
        combined = ConcatDataset([stage2_dataset, replay_dataset])
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add_samples(self, samples: list[dict], stage: int):
        """Add raw samples to the buffer."""
        for s in samples:
            self.buffer.append({**s, "_replay_stage": stage})
        logger.info(
            f"📦 Replay buffer: added {len(samples)} samples from stage {stage} "
            f"(total={len(self.buffer)})"
        )

    def add_dataset(self, dataset: Dataset, stage: int, max_per_stage: int = 2000):
        """Sample from a dataset and add to buffer."""
        n = min(len(dataset), max_per_stage)
        indices = random.sample(range(len(dataset)), n)
        for i in indices:
            item = dataset[i]
            item["_replay_stage"] = stage
            self.buffer.append(item)
        logger.info(f"📦 Added {n} samples from stage-{stage} dataset to replay buffer")

    def sample(self, n: int) -> list[dict]:
        """Sample n items from the buffer."""
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def sample_dataset(self, n: int) -> "ReplayDataset":
        """Return a Dataset of n replayed samples."""
        return ReplayDataset(self.sample(n))

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self.buffer), f, ensure_ascii=False, default=str)
        logger.info(f"💾 Replay buffer saved: {path} ({len(self.buffer)} items)")

    @classmethod
    def load(cls, path: str, max_size: int = 10000) -> "ReplayBuffer":
        import json
        buf = cls(max_size=max_size)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        buf.buffer = deque(items, maxlen=max_size)
        logger.info(f"📂 Replay buffer loaded: {path} ({len(buf.buffer)} items)")
        return buf


class ReplayDataset(Dataset):
    """Wraps replayed samples as a PyTorch Dataset."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Combined Dataloader Builder ─────────────────────────────────────────────────

def build_continual_dataloader(
    current_dataset: Dataset,
    replay_buffer: Optional[ReplayBuffer] = None,
    replay_ratio: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
) -> DataLoader:
    """
    Build a DataLoader that mixes current stage data with replayed past data.

    Args:
        current_dataset:  Current stage Dataset
        replay_buffer:    ReplayBuffer from past stages (optional)
        replay_ratio:     Fraction of batch to fill with replayed data
        batch_size:       Total batch size
        ...

    Returns:
        DataLoader with mixed data
    """
    if replay_buffer is None or len(replay_buffer) == 0 or replay_ratio == 0.0:
        return DataLoader(
            current_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    n_replay = int(len(current_dataset) * replay_ratio)
    n_replay = min(n_replay, len(replay_buffer))

    if n_replay > 0:
        replay_ds = replay_buffer.sample_dataset(n_replay)
        combined = ConcatDataset([current_dataset, replay_ds])
        logger.info(
            f"🔀 Mixed dataset: {len(current_dataset)} current + {n_replay} replayed "
            f"= {len(combined)} total"
        )
    else:
        combined = current_dataset

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
