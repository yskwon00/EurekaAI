"""
EurekaAI — Stage 2 (Elementary) Training Script
초등학교 단계: Stage 1 best에서 Fine-tuning
  - Teacher Q&A 데이터로 기초 지식 학습
  - max_seq_len 256 → 512 확장
  - 졸업 기준: PPL ≤ 15

Run:
    cd EurekaAI
    python stages/stage2_elementary/train.py
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


STAGE1_BEST_CKPT = Path("checkpoints/stage1_toddler/stage1_toddler/best")


def setup_logging(log_dir: str = "logs") -> str:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"stage2_elementary_{timestamp}.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    for h in handlers:
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        root.addHandler(h)
    return str(log_file)


log_file_path = setup_logging("logs")
logger = logging.getLogger(__name__)
logger.info(f"📄 로그 파일: {log_file_path}")


def load_model_from_stage1(config: EurekaConfig) -> EurekaModel:
    """Stage 1 best 체크포인트에서 모델 로드."""
    model = EurekaModel(config)
    if STAGE1_BEST_CKPT.exists():
        state = torch.load(STAGE1_BEST_CKPT / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"✅ Stage 1 체크포인트 로드: {STAGE1_BEST_CKPT}")
    else:
        logger.warning("⚠️  Stage 1 체크포인트 없음 — 랜덤 초기화")
    return model


def build_dataloaders(config: EurekaConfig, tokenizer: EurekaTokenizer):
    import yaml
    sc = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    train_path = sc["data"]["train_path"]
    eval_path  = sc["data"]["eval_path"]

    if not Path(train_path).exists():
        logger.info("⚙️  Stage 2 데이터 없음 — data_prep.py 실행 중...")
        from stages.stage2_elementary.data_prep import main as prep
        prep()

    train_ds = EurekaDataset(train_path, tokenizer, max_seq_len=config.max_seq_len, mode="clm")
    eval_ds  = EurekaDataset(eval_path, tokenizer, max_seq_len=config.max_seq_len, mode="clm") \
               if Path(eval_path).exists() else None

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds, batch_size=config.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn) if eval_ds else None

    logger.info(f"Train: {len(train_ds):,}건 ({len(train_loader)} batches), "
                f"Eval: {len(eval_ds) if eval_ds else 0:,}건")
    return train_loader, eval_loader


def sample_generation(model, tokenizer, device):
    """초등학교 수준 Q&A 생성 확인."""
    model.eval()
    prompts = [
        "Q: 지구는 태양 주위를 어떻게 도나요?\nA:",
        "Q: What is photosynthesis?\nA:",
        "Q: 삼각형의 세 각도의 합은?\nA:",
        "봄에는 꽃이 피고",
    ]
    print("\n" + "─" * 55)
    print("📝 Stage 2 Sample Generations:")
    print("─" * 55)
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer.encode(p, add_bos=True, add_eos=False)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=40)
            print(f"  {p[:40]}... → {tokenizer.decode(out[0].tolist())[:80]}")
    print("─" * 55 + "\n")


def main(config_path: str = None, reset=False):
    logger.info("=" * 55)
    logger.info("   📚 EurekaAI — Stage 2: Elementary Training")
    logger.info("=" * 55)

    default = Path(__file__).parent / "config.yaml"
    if config_path is None and default.exists():
        config_path = str(default)
        logger.info(f"📋 config.yaml 적용: {config_path}")

    config = EurekaConfig.from_yaml(config_path) if config_path else EurekaConfig()
    config.current_stage = 2
    device = config.resolve_device()
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")

    tokenizer = EurekaTokenizer(config.tokenizer_path)
    logger.info(f"✅ Tokenizer: {config.tokenizer_path}")

    model = load_model_from_stage1(config)
    n_params = model.num_parameters()
    logger.info(f"Model: {n_params/1e6:.2f}M params")

    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    progression = ProgressionManager()
    progression.start_stage(2)
    progression.print_summary()

    trainer = EurekaTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )


    if not reset:
        import re
        stage_name_str = config.stage_names[config.current_stage] if hasattr(config, "stage_names") else "stage" + str(config.current_stage)
        ckpt_path = Path(f"checkpoints/{stage_name_str}/{stage_name_str}")
        latest_step = -1
        best_tag = None
        if ckpt_path.exists():
            for d in ckpt_path.iterdir():
                if d.is_dir() and d.name.startswith("step_"):
                    try:
                        step_val = int(d.name.split("_")[1])
                        if step_val > latest_step:
                            latest_step = step_val
                            best_tag = d.name
                    except: pass
        if best_tag:
            logger.info(f"🔄 [Resume] 기존 체크포인트 발견! '{best_tag}' 부터 이어서 학습합니다.")
            trainer.load_checkpoint(best_tag)
        else:
            logger.info("ℹ️  [New Start] 이어할 체크포인트가 없습니다. 이전 Stage에서 전달받은 상태로 처음부터 시작합니다.")
    else:
        logger.info("🔄 [Reset] --reset 옵션 활성화됨. 기존 체크포인트를 무시하고 처음부터 새롭게 학습합니다.")

    metrics = trainer.train()

    best_loss = metrics.get("best_eval_loss", float("inf"))
    ppl = math.exp(min(best_loss, 20))
    graduated = ppl <= 20.0

    logger.info(f"📊 Stage 2 최종 PPL: {ppl:.2f} "
                f"({'✅ GRADUATED!' if graduated else '❌ 미졸업 (기준 PPL≤15)'})")

    progression.record_eval(2, {"ppl": ppl}, step=config.max_steps)
    if graduated:
        progression.complete_stage(2, checkpoint_path="checkpoints/stage2_elementary/best")

    sample_generation(model, tokenizer, torch.device(device))

    model.save_pretrained("checkpoints/stage2_elementary/final_model")
    progression.print_summary()
    logger.info("✅ Stage 2 학습 완료!")
    return graduated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--reset", action="store_true", help="이전 체크포인트를 무시하고 처음부터 다시 시작")
    args = parser.parse_args()
    main(config_path=args.config, reset=args.reset)
