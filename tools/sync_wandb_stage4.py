import os
import sys
import time
import wandb
from pathlib import Path

PROJECT_NAME = "EurekaAI-Curriculum"

def wandb_lineage_sync_stage4():
    """
    Stage 0~3과 완벽하게 동일한 시각적/체계적 리니지 구조(v1/v2 동기화)를
    Stage 4에도 똑같이 부여하기 위해 강제 버저닝 동기화를 수행합니다.
    """
    print(f"🚀 [Lineage Sync] Stage 4 리니지 체계화 동기화 시작")
    
    stage_idx = 4
    stage_name = "stage4"
    
    dummy_file = Path("dummy_lineage_stage4.txt")
    with open(dummy_file, "w") as f:
        f.write(f"Lineage Version Force Stage 4: {time.time()}")
    
    # ── 1. Data Preparation Run (데이터 파이프라인 노드) ──
    run_data = wandb.init(
        project=PROJECT_NAME, 
        job_type="data_prep", 
        name=f"{stage_name}_data_generation_v2",
        config={"method": "Ollama Teacher Essay & Advanced Instruction Tuning", "samples": 13623},
        reinit=True
    )
    
    data_dir = Path("data/processed/stage4")
    dataset_artifact = None
    if data_dir.exists():
        dataset_artifact = wandb.Artifact(
            name=f"dataset-{stage_name}", 
            type="dataset", 
            description=f"Generated dataset for {stage_name} (Synced)"
        )
        for file_path in data_dir.glob("*.jsonl"):
            dataset_artifact.add_file(str(file_path))
        
        dataset_artifact.add_file(str(dummy_file))
        run_data.log_artifact(dataset_artifact)
        print(f"  ✅ [Data Node] dataset-{stage_name} 로깅됨 (완벽한 구조화 v1/v2)")
    run_data.finish()
    
    # ── 2. Training Run (훈련 파이프라인 노드) ──
    run_train = wandb.init(
        project=PROJECT_NAME, 
        job_type="train", 
        name=f"{stage_name}_training_v2",
        reinit=True
    )
    
    if dataset_artifact:
        run_train.use_artifact(f"dataset-{stage_name}:latest")
        print(f"  🔗 [Lineage] Input: dataset-{stage_name} 연결 완료")
        
    try:
        run_train.use_artifact(f"model-stage3:latest")
        print(f"  🔗 [Lineage] Input: model-stage3 연결 완료")
    except Exception as e:
        print(f"  ⚠️ 이전 모델 아티팩트 연결 실패: {e}")

    # [OUTPUT] 훈련 결과 모델을 아티팩트로 로깅
    best_dir = Path("checkpoints/stage4_high/stage4_high/best")
    if best_dir.exists():
        model_artifact = wandb.Artifact(
            name=f"model-{stage_name}", 
            type="model",
            description=f"{stage_name} Best Checkpoint (Synced)"
        )
        model_artifact.add_dir(str(best_dir))
        model_artifact.add_file(str(dummy_file))
        
        run_train.log_artifact(model_artifact)
        print(f"  ✅ [Model Node] model-{stage_name} 로깅됨 (Output)")
    run_train.finish()

    if dummy_file.exists():
        dummy_file.unlink()

    print("\n🎉 Stage 4 리니지 체계화 완벽 동기화 완료!")

if __name__ == "__main__":
    wandb_lineage_sync_stage4()
