import wandb
import time
from pathlib import Path

def link_stage3():
    # 1. Dataset Node
    print("🚀 Stage 3 데이터셋 아티팩트 노드 생성 중...")
    run_data = wandb.init(project="EurekaAI-Curriculum", job_type="data_prep", name="stage3_data_sync_manual")
    ds_art = wandb.Artifact(name="dataset-stage3", type="dataset")
    if Path("data/processed/stage3_middle/train_stage3_middle.jsonl").exists():
        ds_art.add_file("data/processed/stage3_middle/train_stage3_middle.jsonl")
    
    # Force new version hash
    with open("dummy.txt", "w") as f: f.write(str(time.time()))
    ds_art.add_file("dummy.txt")
    
    run_data.log_artifact(ds_art)
    run_data.finish()
    print("✅ Stage 3 데이터셋 등록 완료.")

    # 2. Training Node
    print("🚀 Stage 3 훈련 아티팩트(리니지) 연결 중...")
    run = wandb.init(project="EurekaAI-Curriculum", job_type="train", name="stage3_lineage_stitch")
    
    # [입력 1] 이전 스테이지 모델 (Stage 2)
    try:
        run.use_artifact("model-stage2:latest")
        print("🔗 Input 연결: model-stage2")
    except Exception as e:
        print(f"⚠️ model-stage2 연결 실패: {e}")
        
    # [입력 2] 지금 올린 데이터셋
    run.use_artifact("dataset-stage3:latest")
    print("🔗 Input 연결: dataset-stage3")

    # [출력] Stage 3 모델
    model_art = wandb.Artifact(name="model-stage3", type="model")
    model_art.add_dir("checkpoints/stage3_middle/stage3_middle/best")
    model_art.add_file("dummy.txt")
    run.log_artifact(model_art)
    run.finish()
    print("✅ Stage 3 모델 등록 및 리니지 연결 완벽 달성!")

    Path("dummy.txt").unlink(missing_ok=True)

if __name__ == "__main__":
    link_stage3()
