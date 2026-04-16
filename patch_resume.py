import os
import re
from pathlib import Path

def patch_file(p):
    content = p.read_text(encoding="utf-8")
    
    # Check if already patched
    if "reset=False" in content and "def main" in content:
        # We might need to replace --resume logic in stage 3
        if "--resume" in content:
            content = re.sub(r'def main\(config_path=None, resume=[^)]+\):', 'def main(config_path=None, reset=False):', content)
            content = content.replace('parser.add_argument("--resume", action="store_true", help="최신 체크포인트에서 학습 재개")', 'parser.add_argument("--reset", action="store_true", help="이전 체크포인트를 무시하고 처음부터 다시 시작")')
            content = re.sub(r'main\(config_path=args\.config, resume=args\.resume\)', 'main(config_path=args.config, reset=args.reset)', content)
            
            # Remove old resume block
            old_resume_pattern = r'    if resume:.*?    metrics = trainer\.train\(\)'
            content = re.sub(old_resume_pattern, '    metrics = trainer.train()', content, flags=re.DOTALL)
            
    # Apply new logic
    # 1. Update def main
    content = re.sub(r'def main\((config_path[:a-zA-Z\s=]*|config_path=None)\):', r'def main(\1, reset=False):', content)
    
    # 2. Update argparse
    if 'parser.add_argument("--config"' in content and 'parser.add_argument("--reset"' not in content:
        content = content.replace('parser.add_argument("--config", type=str, default=None)',
                                  'parser.add_argument("--config", type=str, default=None)\n    parser.add_argument("--reset", action="store_true", help="이전 체크포인트를 무시하고 처음부터 다시 시작")')
        content = re.sub(r'main\(config_path=args\.config\)', 'main(config_path=args.config, reset=args.reset)', content)

    # 3. Insert resume logic before trainer.train()
    resume_logic = """
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

    metrics = trainer.train()"""
    
    if "logger.info(\"🔄 [Resume]" not in content:
        content = content.replace("    metrics = trainer.train()", resume_logic)
        
    p.write_text(content, encoding="utf-8")
    print(f"Patched: {p}")

for stage_dir in Path("stages").iterdir():
    if stage_dir.is_dir():
        train_py = stage_dir / "train.py"
        if train_py.exists():
            patch_file(train_py)
