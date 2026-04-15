"""
EurekaAI — Model Configuration
Defines the hyperparameters for TinyLearnAI-30M architecture.
Designed to be cloud-scalable: same config class, just change numbers.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import yaml


@dataclass
class EurekaConfig:
    # ── Model Architecture ──────────────────────────────────────────────────
    vocab_size: int = 32000         # BPE vocab (Ko + En)
    hidden_size: int = 512          # Embedding & hidden dimension
    num_layers: int = 6             # Transformer decoder layers
    num_heads: int = 8              # Attention heads
    head_dim: int = 64              # hidden_size // num_heads
    intermediate_size: int = 1024  # FFN inner dimension
    max_seq_len: int = 512          # Max context length (expandable per stage)
    dropout: float = 0.1
    tie_weights: bool = True        # Share embedding ↔ lm_head weights (~saves 16M params)

    # ── RoPE Settings ───────────────────────────────────────────────────────
    rope_base: float = 10000.0      # RoPE theta base

    # ── Training ────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 5000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 50

    # ── Device ──────────────────────────────────────────────────────────────
    device: str = "auto"            # auto → mps > cuda > cpu
    fp16: bool = False              # Use fp16 mixed precision (cuda)
    bf16: bool = False              # Use bf16 (A100/H100)

    # ── Paths ────────────────────────────────────────────────────────────────
    tokenizer_path: str = "data/tokenizer/eureka.model"
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data/processed"

    # ── Continual Learning ──────────────────────────────────────────────────
    ewc_lambda: float = 5000.0      # EWC regularization strength
    replay_ratio: float = 0.2       # 20% previous stage data mixed in

    # ── Teacher (Ollama) ────────────────────────────────────────────────────
    teacher_enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    teacher_model: str = "llama3.2:3b"
    teacher_synthetic_ratio: float = 0.3   # 30% synthetic data from teacher

    # ── Curriculum ──────────────────────────────────────────────────────────
    current_stage: int = 0
    stage_names: list = field(default_factory=lambda: [
        "stage0_newborn",
        "stage1_toddler",
        "stage2_elementary",
        "stage3_middle",
        "stage4_high",
        "stage5_university",
        "stage6_social",
    ])

    # ── Logging ─────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "EurekaAI"
    run_name: Optional[str] = None

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
        self.head_dim = self.hidden_size // self.num_heads

    @classmethod
    def from_yaml(cls, path: str) -> "EurekaConfig":
        """Load config from a YAML file (stage-specific overrides)."""
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "EurekaConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def save_yaml(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, allow_unicode=True)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def num_parameters_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.hidden_size
        attn = 4 * self.hidden_size * self.hidden_size  # Q K V O
        ffn = 2 * self.hidden_size * self.intermediate_size
        per_layer = attn + ffn
        total = embed + self.num_layers * per_layer
        if not self.tie_weights:
            total += self.vocab_size * self.hidden_size  # lm_head
        return total

    def resolve_device(self) -> str:
        """Auto-detect best available device: mps > cuda > cpu."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def __repr__(self):
        params = self.num_parameters_estimate() / 1e6
        return (
            f"EurekaConfig(layers={self.num_layers}, hidden={self.hidden_size}, "
            f"heads={self.num_heads}, ffn={self.intermediate_size}, "
            f"vocab={self.vocab_size}, ~{params:.1f}M params)"
        )


# ── Preset Configs ──────────────────────────────────────────────────────────────

def tiny_config() -> EurekaConfig:
    """~30M params — Mac experimental (default)"""
    return EurekaConfig()


def small_config() -> EurekaConfig:
    """~120M params — Cloud small"""
    return EurekaConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=2048,
        vocab_size=32000,
        max_seq_len=1024,
    )


def medium_config() -> EurekaConfig:
    """~350M params — Cloud medium"""
    return EurekaConfig(
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        max_seq_len=2048,
    )
