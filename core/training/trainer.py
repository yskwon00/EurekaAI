"""
EurekaAI — Universal Trainer
Handles training across all stages on Mac MPS or Cloud CUDA.
Features:
  - Device auto-detection (mps/cuda/cpu)
  - Gradient accumulation
  - LR warmup + cosine decay
  - Checkpoint save/load
  - W&B logging (optional)
  - EWC continual learning integration
"""

import os
import time
import math
import logging
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from ..model.architecture import EurekaModel
from ..model.config import EurekaConfig

logger = logging.getLogger(__name__)


# ── LR Scheduler ───────────────────────────────────────────────────────────────

def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Warmup + cosine decay schedule. Returns LR multiplier (0~1)."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ── Trainer ─────────────────────────────────────────────────────────────────────

class EurekaTrainer:
    """
    Stage-agnostic trainer for EurekaAI.
    Supports:
      - Mac MPS / CUDA / CPU
      - EWC penalty (continual learning)
      - Replay buffer mixing
      - Checkpoint resume
    """

    def __init__(
        self,
        model: EurekaModel,
        config: EurekaConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        ewc=None,                    # EWC instance (optional)
        eval_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.device = torch.device(config.resolve_device())
        logger.info(f"🖥️  Training device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_dataloader
        self.eval_loader = eval_dataloader
        self.ewc = ewc
        self.eval_fn = eval_fn

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Step counter
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # Paths
        self.ckpt_dir = Path(config.checkpoint_dir) / config.stage_names[config.current_stage]
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # W&B (optional)
        self.wandb = None
        if config.use_wandb:
            self._init_wandb()

    def _build_optimizer(self) -> AdamW:
        """AdamW with weight decay applied only to non-bias, non-LN params."""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "ln_", "ln1", "ln2", "ln_final"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=1e-8,
        )

    def _init_wandb(self):
        try:
            import wandb
            stage_name = self.config.stage_names[self.config.current_stage]
            stage_idx = self.config.current_stage
            short_stage = f"stage{stage_idx}"

            self.wandb = wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name or stage_name,
                config=vars(self.config),
                resume="allow",
            )
            
            # ── [Lineage] 자동으로 Input 연결 ──
            try:
                self.wandb.use_artifact(f"dataset-{short_stage}:latest")
                logger.info(f"🔗 [W&B Lineage] connected dataset: dataset-{short_stage}")
            except Exception as e:
                logger.warning(f"Failed to connect dataset artifact: {e}")
                
            if stage_idx > 0:
                prev_stage = f"stage{stage_idx - 1}"
                try:
                    self.wandb.use_artifact(f"model-{prev_stage}:latest")
                    logger.info(f"🔗 [W&B Lineage] connected previous model: model-{prev_stage}")
                except Exception as e:
                    logger.warning(f"Failed to connect previous model artifact: {e}")
                    
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")

    def _get_lr(self) -> float:
        lr_mul = cosine_lr_schedule(
            self.global_step,
            self.config.warmup_steps,
            self.config.max_steps,
        )
        return self.config.learning_rate * lr_mul

    def _update_lr(self):
        lr = self._get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ── Training ────────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """Run training loop. Returns final metrics."""
        self.model.train()
        total_loss = 0.0
        t0 = time.time()
        grad_accum = self.config.gradient_accumulation_steps
        data_iter = iter(self.train_loader)

        logger.info(
            f"🚀 Starting training | stage={self.config.current_stage} "
            f"| steps={self.config.max_steps} | device={self.device}"
        )

        self.optimizer.zero_grad(set_to_none=True)

        start_step = int(self.global_step * grad_accum) + 1
        for step in range(start_step, self.config.max_steps + 1):
            # ── Get batch ──────────────────────────────────────────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", None)
            if labels is None:
                # Auto-regressive: shift input_ids by 1
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100
            labels = labels.to(self.device)

            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # ── Forward ────────────────────────────────────────────────────────
            outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"] / grad_accum

            # Add EWC penalty if enabled
            if self.ewc is not None:
                ewc_loss = self.ewc.penalty(self.model)
                loss = loss + ewc_loss / grad_accum

            # ── Backward ───────────────────────────────────────────────────────
            loss.backward()
            total_loss += loss.item() * grad_accum

            if step % grad_accum == 0:
                # Gradient clipping
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                lr = self._update_lr()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # ── Logging ───────────────────────────────────────────────────
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = total_loss / self.config.log_interval
                    elapsed = time.time() - t0
                    ppl = math.exp(min(avg_loss, 20))
                    logger.info(
                        f"Step {self.global_step:5d} | loss={avg_loss:.4f} | "
                        f"ppl={ppl:.1f} | lr={lr:.2e} | {elapsed:.1f}s"
                    )
                    if self.wandb:
                        self.wandb.log({
                            "train/loss": avg_loss,
                            "train/ppl": ppl,
                            "train/lr": lr,
                            "step": self.global_step,
                        })
                    total_loss = 0.0
                    t0 = time.time()

                # ── Evaluation ────────────────────────────────────────────────
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self._evaluate()
                    self.model.train()
                    eval_loss = eval_metrics.get("eval_loss", float("inf"))

                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best")

                # ── Checkpoint ────────────────────────────────────────────────
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        self.save_checkpoint("final")
        logger.info(f"✅ Training complete | best_eval_loss={self.best_eval_loss:.4f}")
        return {"best_eval_loss": self.best_eval_loss}

    def _evaluate(self) -> dict:
        """Run evaluation loop."""
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)

                # ── labels: 배치에 이미 있으면 사용, 없으면 CLM 방식으로 생성
                labels = batch.get("labels", None)
                if labels is None:
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = -100
                labels = labels.to(self.device)

                # ── attention_mask 반영
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                total_loss += outputs["loss"].item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        logger.info(f"📊 Eval | loss={avg_loss:.4f} | ppl={ppl:.1f}")

        # Custom eval fn (e.g., benchmark)
        extra_metrics = {}
        if self.eval_fn:
            extra_metrics = self.eval_fn(self.model, self.device)

        metrics = {"eval_loss": avg_loss, "eval_ppl": ppl, **extra_metrics}
        if self.wandb:
            self.wandb.log({"step": self.global_step, **metrics})
        return metrics

    # ── Checkpointing ────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str = "latest"):
        path = self.ckpt_dir / tag
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "model.pt")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        torch.save({"global_step": self.global_step}, path / "trainer_state.pt")
        self.config.save_json(str(path / "config.json"))

        logger.info(f"💾 Checkpoint saved: {path}")

        if self.wandb and tag in ["best", "final"]:
            import wandb
            stage_idx = self.config.current_stage
            art_name = f"model-stage{stage_idx}"
            
            # 더미 파일을 통해 해시 강제 변형 및 버전 안전 생성 보장
            import time
            dummy_file = path / "dummy_lineage.txt"
            with open(dummy_file, "w") as f:
                f.write(f"Lineage Hash Force: {time.time()}")
            # Only log artifact during these specific milestones
            artifact = wandb.Artifact(
                name=art_name, 
                type="model",
                description=f"EurekaAI {stage_name} {tag} checkpoint"
            )
            artifact.add_dir(str(path))
            self.wandb.log_artifact(artifact, aliases=[tag, f"step_{self.global_step}"])
            logger.info(f"🌐 W&B Artifact uploaded: {art_name} ({tag})")

    def load_checkpoint(self, tag: str = "latest"):
        path = self.ckpt_dir / tag
        if not path.exists():
            logger.warning(f"No checkpoint found at {path}")
            return

        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location=self.device)
        )
        self.optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location=self.device)
        )
        state = torch.load(path / "trainer_state.pt")
        self.global_step = state["global_step"]
        logger.info(f"📂 Checkpoint loaded: {path} (step={self.global_step})")
