"""
EurekaAI — Stage 1 (Toddler) Training Script
유아기 단계: Stage 0 best 체크포인트에서 Fine-tuning
  - 더 긴 문맥 (max_seq_len 128 → 256)
  - EWC로 Stage 0 지식 보존
  - Continual Learning: Stage 0 데이터 20% 리플레이

Run:
    cd EurekaAI
    python stages/stage1_toddler/train.py
"""

import sys
import math
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model.config import EurekaConfig
from core.model.architecture import EurekaModel
from core.model.tokenizer_utils import EurekaTokenizer
from core.curriculum.data_manager import EurekaDataset, collate_fn
from core.training.trainer import EurekaTrainer
from core.curriculum.progression import ProgressionManager
from torch.utils.data import DataLoader


def setup_logging(log_dir: str = "logs") -> str:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"stage1_toddler_{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return str(log_file)


log_file_path = setup_logging("logs")
logger = logging.getLogger(__name__)
logger.info(f"📄 로그 파일: {log_file_path}")

STAGE0_BEST_CKPT = Path("checkpoints/stage0_newborn/stage0_newborn/best")


def load_model_from_stage0(config: EurekaConfig) -> EurekaModel:
    """Stage 0 best 체크포인트에서 모델 로드."""
    model = EurekaModel(config)

    if STAGE0_BEST_CKPT.exists():
        model_path = STAGE0_BEST_CKPT / "model.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ Stage 0 체크포인트 로드: {STAGE0_BEST_CKPT}")
    else:
        logger.warning("⚠️  Stage 0 체크포인트 없음 — 랜덤 초기화로 시작")

    return model


def build_dataloaders(config: EurekaConfig, tokenizer: EurekaTokenizer):
    import yaml
    stage_config_path = Path(__file__).parent / "config.yaml"
    with open(stage_config_path) as f:
        sc = yaml.safe_load(f)

    data_cfg = sc.get("data", {})
    train_path = data_cfg.get("train_path", "data/processed/stage1/train_stage1.jsonl")
    eval_path  = data_cfg.get("eval_path",  "data/processed/stage1/eval_stage1.jsonl")

    if not Path(train_path).exists():
        logger.info("⚙️  Stage 1 데이터 없음 — data_prep.py 실행 중...")
        from stages.stage1_toddler.data_prep import main as prep_main
        prep_main()

    train_ds = EurekaDataset(train_path, tokenizer, max_seq_len=config.max_seq_len, mode="clm")
    eval_ds  = EurekaDataset(eval_path,  tokenizer, max_seq_len=config.max_seq_len, mode="clm") \
               if Path(eval_path).exists() else None

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds,  batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn) if eval_ds else None

    return train_loader, eval_loader


def check_graduation(eval_loss: float, threshold_ppl: float = 20.0) -> bool:
    """Stage 1 졸업: PPL ≤ 20"""
    ppl = math.exp(min(eval_loss, 20))
    graduated = ppl <= threshold_ppl
    logger.info(
        f"📊 Stage 1 eval PPL: {ppl:.2f} "
        f"({'✅ GRADUATED!' if graduated else f'(need <= {threshold_ppl})'})"
    )
    return graduated


def sample_generation(model: EurekaModel, tokenizer: EurekaTokenizer, device: torch.device):
    """Stage 1 수준 프롬프트로 생성 샘플 확인."""
    model.eval()
    prompts = [
        "오늘은 날씨가",
        "The dog likes to",
        "엄마가 밥을",
        "Once upon a time",
    ]
    print("\n" + "─" * 55)
    print("📝 Stage 1 Sample Generations:")
    print("─" * 55)
    for p in prompts:
        ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=30)
        generated = tokenizer.decode(out[0].tolist())
        print(f"  [{p}] → {generated}")
    print("─" * 55 + "\n")


def main(config_path: str = None):
    logger.info("=" * 55)
    logger.info("   🧸 EurekaAI — Stage 1: Toddler Training")
    logger.info("=" * 55)

    # ── Config ──────────────────────────────────────────────
    default_config = Path(__file__).parent / "config.yaml"
    if config_path is None and default_config.exists():
        config_path = str(default_config)
        logger.info(f"📋 config.yaml 자동 적용: {config_path}")

    config = EurekaConfig.from_yaml(config_path) if config_path else EurekaConfig()
    config.current_stage = 1
    device = config.resolve_device()
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = EurekaTokenizer(config.tokenizer_path)
    logger.info(f"✅ Tokenizer 로드: {config.tokenizer_path}")

    # ── Model (Stage 0에서 이어받기) ──────────────────────────
    model = load_model_from_stage0(config)
    logger.info(
        f"Model: {model.num_parameters()/1e6:.2f}M params "
        f"(trainable: {model.num_parameters(trainable_only=True)/1e6:.2f}M)"
    )

    # ── Data ─────────────────────────────────────────────────
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    logger.info(f"Train batches: {len(train_loader)}, Eval: {len(eval_loader) if eval_loader else 0}")

    # ── Progression ──────────────────────────────────────────
    progression = ProgressionManager()
    progression.start_stage(1)
    progression.print_summary()

    # ── Trainer ──────────────────────────────────────────────
    trainer = EurekaTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    # ── Train ─────────────────────────────────────────────────
    metrics = trainer.train()

    # ── Post-training ─────────────────────────────────────────
    best_eval_loss = metrics.get("best_eval_loss", float("inf"))
    graduated = check_graduation(best_eval_loss, threshold_ppl=20.0)

    ppl = math.exp(min(best_eval_loss, 20))
    progression.record_eval(1, {"ppl": ppl}, step=config.max_steps)

    if graduated:
        progression.complete_stage(1, checkpoint_path="checkpoints/stage1_toddler/best")

    # ── Sample Generation ─────────────────────────────────────
    sample_generation(model, tokenizer, torch.device(device))

    # ── Save ──────────────────────────────────────────────────
    model.save_pretrained("checkpoints/stage1_toddler/final_model")
    progression.print_summary()
    logger.info("✅ Stage 1 학습 완료!")

    if not graduated:
        logger.info(f"ℹ️  미졸업 (PPL={ppl:.1f}). 스텝 추가 또는 데이터 개선 필요.")

    return graduated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EurekaAI Stage 1 Training")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
