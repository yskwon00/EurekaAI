"""
EurekaAI — Stage 0 (Newborn) Training Script
The very first training stage: like a newborn learning sounds and patterns.

Run:
    cd EurekaAI
    python stages/stage0_newborn/train.py [--config stages/stage0_newborn/config.yaml]
"""

import sys
import math
import logging
import argparse
from pathlib import Path

import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model.config import EurekaConfig, tiny_config
from core.model.architecture import EurekaModel
from core.model.tokenizer_utils import EurekaTokenizer, train_tokenizer, prepare_tokenizer_corpus
from core.curriculum.data_manager import CurriculumDataManager, EurekaDataset, collate_fn
from core.training.trainer import EurekaTrainer
from core.curriculum.progression import ProgressionManager
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_tokenizer(config: EurekaConfig) -> EurekaTokenizer:
    """Load or train the tokenizer."""
    tok_path = Path(config.tokenizer_path)

    if tok_path.exists():
        logger.info(f"✅ Loading tokenizer: {tok_path}")
        return EurekaTokenizer(str(tok_path))

    logger.info("🔤 Tokenizer not found — training from scratch...")

    # Prepare corpus
    corpus_file = "data/tokenizer/train_corpus.txt"
    if not Path(corpus_file).exists():
        prepare_tokenizer_corpus(corpus_file)

    # Train tokenizer
    tok = train_tokenizer(
        text_files=[corpus_file],
        output_dir=str(tok_path.parent),
        vocab_size=config.vocab_size,
        model_prefix=tok_path.stem,
    )
    return tok


def build_dataloaders(config: EurekaConfig, tokenizer: EurekaTokenizer):
    """Load Stage 0 train/eval data."""
    import yaml
    stage_config_path = Path(__file__).parent / "config.yaml"
    with open(stage_config_path) as f:
        sc = yaml.safe_load(f)

    data_cfg = sc.get("data", {})
    train_path = data_cfg.get("train_path", "data/processed/stage0/train_stage0.jsonl")
    eval_path = data_cfg.get("eval_path", "data/processed/stage0/eval_stage0.jsonl")

    if not Path(train_path).exists():
        logger.warning(f"No training data at {train_path}")
        logger.info("Running data_prep.py first...")
        from stages.stage0_newborn.data_prep import main as prep_main
        prep_main()

    train_ds = EurekaDataset(
        train_path, tokenizer,
        max_seq_len=config.max_seq_len,
        mode="clm",
    )
    eval_ds = EurekaDataset(
        eval_path, tokenizer,
        max_seq_len=config.max_seq_len,
        mode="clm",
    ) if Path(eval_path).exists() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    ) if eval_ds else None

    return train_loader, eval_loader


def perplexity_eval_fn(model: EurekaModel, device: torch.device) -> dict:
    """Calculate perplexity on eval data."""
    # This is called by EurekaTrainer during evaluation
    # Trainer already computes eval_loss; just return empty (PPL derived from it)
    return {}


def check_graduation(eval_loss: float, threshold_ppl: float = 30.0) -> bool:
    """Check if Stage 0 graduation threshold is met."""
    ppl = math.exp(min(eval_loss, 20))
    graduated = ppl <= threshold_ppl
    logger.info(
        f"📊 Stage 0 eval PPL: {ppl:.2f} "
        f"({'✅ GRADUATED!' if graduated else f'(need <= {threshold_ppl})'})"
    )
    return graduated


def sample_generation(model: EurekaModel, tokenizer: EurekaTokenizer, device: torch.device):
    """Show some sample generations to verify the model is learning."""
    model.eval()
    prompts = [
        "안녕",
        "Hello",
        "사과",
    ]

    print("\n" + "─" * 50)
    print("📝 Sample Generations:")
    print("─" * 50)
    for p in prompts:
        ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=20,
            )
        generated = tokenizer.decode(out[0].tolist())
        print(f"  [{p}] → {generated}")
    print("─" * 50 + "\n")


def main(config_path: str = None):
    logger.info("=" * 55)
    logger.info("   🍼 EurekaAI — Stage 0: Newborn Training")
    logger.info("=" * 55)

    # ── Load Config ────────────────────────────────────────────────────────────
    if config_path:
        config = EurekaConfig.from_yaml(config_path)
    else:
        config = tiny_config()
        config.max_seq_len = 128
        config.batch_size = 64
        config.max_steps = 2000
        config.warmup_steps = 50
        config.eval_interval = 200
        config.log_interval = 20
        config.current_stage = 0

    logger.info(f"Config: {config}")
    device = config.resolve_device()
    logger.info(f"Device: {device}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = ensure_tokenizer(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EurekaModel(config)
    logger.info(
        f"Model: {model.num_parameters()/1e6:.2f}M parameters "
        f"(trainable: {model.num_parameters(trainable_only=True)/1e6:.2f}M)"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    logger.info(f"Train batches: {len(train_loader)}, Eval: {len(eval_loader) if eval_loader else 0}")

    # ── Progression Manager ───────────────────────────────────────────────────
    progression = ProgressionManager()
    progression.start_stage(0)
    progression.print_summary()

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = EurekaTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    metrics = trainer.train()

    # ── Post-training Evaluation ───────────────────────────────────────────────
    best_eval_loss = metrics.get("best_eval_loss", float("inf"))
    graduated = check_graduation(best_eval_loss, threshold_ppl=30.0)

    ppl = math.exp(min(best_eval_loss, 20))
    progression.record_eval(0, {"ppl": ppl}, step=config.max_steps)

    if graduated:
        progression.complete_stage(0, checkpoint_path="checkpoints/stage0/best")

    # ── Show Generations ──────────────────────────────────────────────────────
    sample_generation(model, tokenizer, torch.device(device))

    # ── Save Final Model ──────────────────────────────────────────────────────
    model.save_pretrained("checkpoints/stage0/final_model")

    progression.print_summary()
    logger.info("✅ Stage 0 training complete!")

    if not graduated:
        logger.info(
            f"ℹ️  Not graduated yet (PPL={ppl:.1f}). "
            "Consider running more steps or checking data quality."
        )

    return graduated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EurekaAI Stage 0 Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
