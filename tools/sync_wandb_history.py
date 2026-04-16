import os
import sys
import wandb
from pathlib import Path

# Project settings
PROJECT_NAME = "EurekaAI-Curriculum"

def wandb_sync_history():
    print(f"🚀 소급 업로드 스크립트 시작: {PROJECT_NAME}")
    
    # Check if logged in (will prompt if not)
    if wandb.api.api_key is None:
        print("⚠️ W&B API Key가 없습니다. 터미널에서 'wandb login' 을 먼저 실행해주세요!")
        sys.exit(1)
        
    run = wandb.init(project=PROJECT_NAME, job_type="retroactive_upload", name="Upload_History_0_to_3")
    
    stages = range(4) # Stage 0 to 3
    
    # 1. Upload Datasets
    print("\n[1] Datasets 업로드 중...")
    for stage_idx in stages:
        data_dir = Path(f"data/processed/stage{stage_idx}")
        if data_dir.exists():
            artifact = wandb.Artifact(name=f"dataset-stage{stage_idx}", type="dataset", 
                                      description=f"EurekaAI Stage {stage_idx} training dataset (Retroactive)")
            # Add files
            for file_path in data_dir.glob("*.jsonl"):
                artifact.add_file(str(file_path))
            
            run.log_artifact(artifact)
            print(f"  ✅ 업로드 완료: dataset-stage{stage_idx}")
        else:
            print(f"  ⚠️ 데이터 폴더 없음: {data_dir}")

    # 2. Upload Models
    print("\n[2] Models 업로드 중...")
    for stage_idx in stages:
        # Find stage name
        checkpoints_dir = Path("checkpoints")
        stage_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith(f"stage{stage_idx}_")]
        
        if stage_dirs:
            stage_dir = stage_dirs[0]
            best_dir = stage_dir / stage_dir.name / "best"
            
            if best_dir.exists():
                artifact = wandb.Artifact(name=f"model-stage{stage_idx}", type="model",
                                          description=f"EurekaAI Stage {stage_idx} Best Model (Retroactive)")
                artifact.add_dir(str(best_dir))
                
                run.log_artifact(artifact)
                print(f"  ✅ 업로드 완료: model-stage{stage_idx}")
            else:
                print(f"  ⚠️ Best 모델 없음: {best_dir}")
        else:
            print(f"  ⚠️ 체크포인트 폴더 없음: checkpoints/stage{stage_idx}_*")

    run.finish()
    print("\n🎉 모든 히스토리 (Stage 0~3) 업로드 완료!")
    print(f"👉 대시보드 바로가기: https://wandb.ai/{wandb.api.viewer().get('entity')}/{PROJECT_NAME}")

if __name__ == "__main__":
    wandb_sync_history()
