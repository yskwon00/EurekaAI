"""
EurekaAI — Main Entry Point
Orchestrates the full curriculum learning pipeline from Stage 0 to Stage 6.

Usage:
    python run.py --stage 0              # Run only Stage 0
    python run.py --stage all            # Run all stages sequentially
    python run.py --stage 0 --prep-only  # Only prepare data, skip training
    python run.py --eval --stage 0       # Evaluate a specific stage
    python run.py --status               # Show curriculum progress
"""

import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BANNER = r"""
  ______                _         _    ___ 
 |  ____|              | |       / \  |_ _|
 | |__   _   _ _ __ ___| | __  / _ \  | | 
 |  __| | | | | '__/ _ \ |/ / / ___ \ | | 
 | |____| |_| | | |  __/   < / /   \ \| | 
 |______|\__,_|_|  \___|_|\_/_/     \_\___|

  🧠 EurekaAI — Self-Learning Curriculum Model
  신생아에서 사회인까지, 스스로 성장하는 AI
  ─────────────────────────────────────────────
"""


def check_ollama():
    """Verify Ollama is running with a compatible model."""
    try:
        from core.teacher.ollama_teacher import OllamaTeacher
        teacher = OllamaTeacher()
        if teacher.is_available():
            models = teacher.list_models()
            logger.info(f"✅ Ollama available | Models: {models}")
            return True
        else:
            logger.warning(
                "⚠️  Ollama not detected at http://localhost:11434\n"
                "   Teacher-assisted data generation will be skipped.\n"
                "   To enable: ollama serve && ollama pull llama3.2:3b"
            )
            return False
    except Exception as e:
        logger.warning(f"⚠️  Ollama check failed: {e}")
        return False


def run_stage(stage: int, prep_only: bool = False, config_path: str = None):
    """Run a single curriculum stage."""
    stage_map = {
        0: ("stages.stage0_newborn.data_prep", "stages.stage0_newborn.train"),
        # Stages 1-6 will be added as they are implemented
    }

    if stage not in stage_map:
        logger.warning(f"Stage {stage} training not yet implemented. Skipping.")
        return False

    prep_module, train_module = stage_map[stage]

    # Data preparation
    logger.info(f"\n{'='*50}")
    logger.info(f"  📦 Stage {stage} Data Preparation")
    logger.info(f"{'='*50}")
    import importlib
    prep = importlib.import_module(prep_module)
    prep.main()

    if prep_only:
        logger.info("--prep-only flag set: skipping training")
        return True

    # Training
    logger.info(f"\n{'='*50}")
    logger.info(f"  🚀 Stage {stage} Training")
    logger.info(f"{'='*50}")
    train = importlib.import_module(train_module)
    graduated = train.main(config_path=config_path)
    return graduated


def show_status():
    """Display current curriculum progress."""
    from core.curriculum.progression import ProgressionManager
    progression = ProgressionManager()
    progression.print_summary()


def run_eval(stage: int):
    """Run evaluation for a trained stage."""
    logger.info(f"📊 Evaluating Stage {stage}...")
    ckpt_path = Path(f"checkpoints/stage{stage}/final_model")

    if not ckpt_path.exists():
        logger.error(f"No checkpoint found at {ckpt_path}")
        logger.info(f"Run training first: python run.py --stage {stage}")
        return

    from core.model.architecture import EurekaModel
    from core.model.tokenizer_utils import EurekaTokenizer
    from core.evaluation.benchmarks import quick_stage_eval
    import torch

    model = EurekaModel.from_pretrained(str(ckpt_path))
    config = model.config
    device = torch.device(config.resolve_device())
    model = model.to(device)

    tok_path = config.tokenizer_path
    tokenizer = EurekaTokenizer(tok_path)

    print(f"\n─── Stage {stage} Quick Evaluation ───")
    quick_stage_eval(model, tokenizer, device, stage)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="EurekaAI — Self-Learning Curriculum Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage to run: 0-6 or 'all'",
    )
    parser.add_argument(
        "--prep-only",
        action="store_true",
        help="Only prepare data, skip training",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation on trained stage",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show curriculum progress",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config override",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip Ollama availability check",
    )
    args = parser.parse_args()

    # Status only
    if args.status:
        show_status()
        return

    # Eval only
    if args.eval and args.stage:
        stage = int(args.stage)
        run_eval(stage)
        return

    if args.stage is None:
        parser.print_help()
        return

    # Check Ollama
    if not args.skip_ollama_check:
        check_ollama()

    # Run stages
    if args.stage == "all":
        logger.info("🌍 Running full curriculum (Stage 0 → 6)")
        for stage in range(7):
            success = run_stage(stage, prep_only=args.prep_only, config_path=args.config)
            if not success:
                logger.warning(f"Stage {stage} did not graduate — continuing anyway...")
    else:
        stage = int(args.stage)
        run_stage(stage, prep_only=args.prep_only, config_path=args.config)


if __name__ == "__main__":
    main()
