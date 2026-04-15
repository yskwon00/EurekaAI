"""
EurekaAI — Stage Progression Manager
Tracks per-stage benchmarks and decides when to graduate to the next stage.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ── Graduation Thresholds ───────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    0: {"ppl": 30.0,      "metric": "ppl",      "direction": "lower"},   # PPL < 30
    1: {"accuracy": 0.50, "metric": "accuracy",  "direction": "higher"},  # word analogy > 50%
    2: {"f1": 0.60,       "metric": "f1",        "direction": "higher"},  # reading QA F1 > 60%
    3: {"f1": 0.65,       "metric": "f1",        "direction": "higher"},  # KorQuAD > 65%
    4: {"accuracy": 0.45, "metric": "accuracy",  "direction": "higher"},  # CSAT-style > 45%
    5: {"accuracy": 0.40, "metric": "accuracy",  "direction": "higher"},  # MMLU-Ko > 40%
    6: {"score": 4.0,     "metric": "score",     "direction": "higher"},  # MT-Bench > 4.0
}

STAGE_NAMES = {
    0: "🍼 신생아 (Newborn)",
    1: "🧸 유아기 (Toddler)",
    2: "📚 초등학교 (Elementary)",
    3: "🔢 중학교 (Middle School)",
    4: "📐 고등학교 (High School)",
    5: "🎓 대학교 (University)",
    6: "🌐 사회인 (Social)",
}


# ── Stage Record ────────────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    stage: int
    name: str
    status: str = "pending"          # pending | training | completed | graduated
    best_metric: float = 0.0
    eval_history: list = field(default_factory=list)
    train_steps: int = 0
    checkpoint_path: str = ""


# ── ProgressionManager ──────────────────────────────────────────────────────────

class ProgressionManager:
    """
    Tracks training progress across all stages.
    Determines graduation and manages state persistence.
    """

    def __init__(
        self,
        save_path: str = "checkpoints/progression.json",
        thresholds: Optional[dict] = None,
    ):
        self.save_path = Path(save_path)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.records: dict[int, StageRecord] = {}
        self.current_stage = 0

        self._load_or_init()

    def _load_or_init(self):
        if self.save_path.exists():
            self._load()
        else:
            for stage in range(7):
                self.records[stage] = StageRecord(
                    stage=stage,
                    name=STAGE_NAMES[stage],
                )
            self._save()

    def _load(self):
        with open(self.save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.current_stage = data.get("current_stage", 0)
        for s, rec in data.get("records", {}).items():
            self.records[int(s)] = StageRecord(**rec)
        logger.info(f"📂 Progression loaded: current_stage={self.current_stage}")

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump({
                "current_stage": self.current_stage,
                "records": {
                    s: asdict(r) for s, r in self.records.items()
                },
            }, f, indent=2, ensure_ascii=False)

    def start_stage(self, stage: int):
        """Mark a stage as actively training."""
        self.records[stage].status = "training"
        self.current_stage = stage
        self._save()
        logger.info(f"🚀 Stage {stage} started: {STAGE_NAMES[stage]}")

    def record_eval(
        self,
        stage: int,
        metrics: dict,
        step: int,
    ) -> bool:
        """
        Record evaluation metrics for a stage.

        Returns:
            True if graduation threshold is met.
        """
        record = self.records[stage]
        record.eval_history.append({"step": step, **metrics})
        record.train_steps = step

        # Check graduation
        threshold = self.thresholds.get(stage, {})
        metric_key = threshold.get("metric", "ppl")
        direction = threshold.get("direction", "lower")
        threshold_val = threshold.get(metric_key, float("inf"))

        current_val = metrics.get(metric_key, None)
        if current_val is None:
            self._save()
            return False

        # Track best
        if direction == "lower":
            if current_val < record.best_metric or record.best_metric == 0.0:
                record.best_metric = current_val
            graduated = current_val <= threshold_val
        else:
            if current_val > record.best_metric:
                record.best_metric = current_val
            graduated = current_val >= threshold_val

        if graduated:
            record.status = "graduated"
            logger.info(
                f"🎓 Stage {stage} GRADUATED! "
                f"{metric_key}={current_val:.4f} (threshold={threshold_val})"
            )
        else:
            record.status = "training"
            logger.info(
                f"📊 Stage {stage} eval: {metric_key}={current_val:.4f} "
                f"(need {'<=' if direction == 'lower' else '>='} {threshold_val})"
            )

        self._save()
        return graduated

    def complete_stage(self, stage: int, checkpoint_path: str = ""):
        """Mark a stage as completed (even if not graduated)."""
        self.records[stage].status = "completed"
        self.records[stage].checkpoint_path = checkpoint_path
        self._save()
        logger.info(f"✅ Stage {stage} completed")

    def can_advance(self, stage: int) -> bool:
        """Check if all prerequisites are met to advance past a stage."""
        return self.records[stage].status in ("graduated", "completed")

    def next_stage(self) -> Optional[int]:
        """Return next stage to train, or None if all complete."""
        for stage in range(7):
            if self.records[stage].status in ("pending", "training"):
                return stage
        return None  # All stages complete

    def print_summary(self):
        """Print a pretty summary of all stage progress."""
        print("\n" + "=" * 60)
        print("   EurekaAI — Curriculum Progress")
        print("=" * 60)
        for stage, rec in self.records.items():
            icon = {
                "pending": "⏳",
                "training": "🔄",
                "completed": "✅",
                "graduated": "🎓",
            }.get(rec.status, "❓")
            print(f"  {icon} Stage {stage}: {rec.name:<30} [{rec.status}]")
            if rec.best_metric:
                print(f"      best_metric={rec.best_metric:.4f}, steps={rec.train_steps}")
        print("=" * 60 + "\n")
