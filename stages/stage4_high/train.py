"""EurekaAI — Stage 4 (High School) Training. 졸업 기준: PPL ≤ 8"""
import sys, math, logging, argparse
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

PREV_CKPT = Path("checkpoints/stage3_middle/stage3_middle/best")

def setup_logging(prefix="stage4_high"):
    Path("logs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{prefix}_{ts}.log"
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ], format="%(asctime)s [%(levelname)s] %(message)s")
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"📄 로그: {log_file}")

def load_model(config):
    model = EurekaModel(config)
    if PREV_CKPT.exists():
        state = torch.load(PREV_CKPT / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"✅ Stage 3 체크포인트 로드: {PREV_CKPT}")
    else:
        logger.warning("⚠️  이전 체크포인트 없음 — 랜덤 초기화")
    return model

def build_dataloaders(config, tokenizer):
    import yaml
    sc = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    train_path = sc["data"]["train_path"]
    eval_path  = sc["data"]["eval_path"]
    if not Path(train_path).exists():
        from stages.stage4_high.data_prep import main as prep; prep()
    train_ds = EurekaDataset(train_path, tokenizer, max_seq_len=config.max_seq_len, mode="clm")
    eval_ds  = EurekaDataset(eval_path,  tokenizer, max_seq_len=config.max_seq_len, mode="clm") if Path(eval_path).exists() else None
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds,  batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn) if eval_ds else None
    logger.info(f"Train: {len(train_ds):,}건, Eval: {len(eval_ds) if eval_ds else 0:,}건")
    return train_loader, eval_loader

def main(config_path=None):
    logger.info("=" * 55)
    logger.info("   📐 EurekaAI — Stage 4: High School Training")
    logger.info("=" * 55)
    default = Path(__file__).parent / "config.yaml"
    if config_path is None and default.exists(): config_path = str(default)
    config = EurekaConfig.from_yaml(config_path) if config_path else EurekaConfig()
    config.current_stage = 4
    device = config.resolve_device()
    logger.info(f"Device: {device}")
    tokenizer = EurekaTokenizer(config.tokenizer_path)
    model = load_model(config)
    logger.info(f"Model: {model.num_parameters()/1e6:.2f}M params")
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    progression = ProgressionManager()
    progression.start_stage(4)
    trainer = EurekaTrainer(model=model, config=config, train_dataloader=train_loader, eval_dataloader=eval_loader)
    metrics = trainer.train()
    best_loss = metrics.get("best_eval_loss", float("inf"))
    ppl = math.exp(min(best_loss, 20))
    graduated = ppl <= 8.0
    logger.info(f"📊 Stage 4 최종 PPL: {ppl:.2f} ({'✅ GRADUATED!' if graduated else '❌ 미졸업 (기준 PPL≤8)'})")
    progression.record_eval(4, {"ppl": ppl}, step=config.max_steps)
    if graduated:
        progression.complete_stage(4, checkpoint_path="checkpoints/stage4_high/best")
    model.save_pretrained("checkpoints/stage4_high/final_model")
    progression.print_summary()
    logger.info("✅ Stage 4 학습 완료!")
    return graduated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    main(config_path=parser.parse_args().config)
