import os
import sys
import time
import wandb
from pathlib import Path

PROJECT_NAME = "EurekaAI-Curriculum"

def get_data_prep_config(stage_idx):
    if stage_idx == 0:
        return {"method": "HuggingFace wiki_ko + wiki_en simple load", "samples": 70000}
    elif stage_idx == 1:
        return {"method": "Ollama Teacher Synthetic Q&A (toddler style)", "samples": 55000}
    elif stage_idx == 2:
        return {"method": "Ollama Teacher Q&A + basic CoT + Continual Replay", "samples": 25000}
    elif stage_idx == 3:
        return {"method": "Ollama Teacher Advanced CoT + Score Filtering", "samples": 15000}
    return {}

def wandb_lineage_sync():
    """
    기존 아티팩트의 해시 중복으로 인해 W&B가 같은 노드(v0)만 재사용하여
    그래프가 끊기는 문제를 해결하기 위해 고유 버전(v2)을 강제 생성하여
    DAG 트리를 완벽하게 새로 잇습니다.
    """
    print(f"🚀 [Lineage Sync V2] 강제 버저닝 리니지 구축 시작: {PROJECT_NAME}")
    
    if wandb.api.api_key is None:
        print("⚠️ W&B API Key 누락. 'wandb login' 필수.")
        sys.exit(1)
        
    stages = range(4) # 0, 1, 2, 3
    
    # 더미 해시 파일을 만들어 버전을 무조건 새로 따게 만듭니다.
    dummy_file = Path("dummy_lineage.txt")
    with open(dummy_file, "w") as f:
        f.write(f"Lineage Version Force: {time.time()}")
    
    for stage in stages:
        stage_name = f"stage{stage}"
        print(f"\n=========================================")
        print(f" ➡️  Stage {stage} Lineage 처리 중")
        print(f"=========================================")
        
        # ── 1. Data Preparation Run (데이터 파이프라인 노드) ──
        run_data = wandb.init(
            project=PROJECT_NAME, 
            job_type="data_prep", 
            name=f"{stage_name}_data_generation_v2",
            config=get_data_prep_config(stage),
            reinit=True
        )
        
        data_dir = Path(f"data/processed/{stage_name}")
        dataset_artifact = None
        if data_dir.exists():
            dataset_artifact = wandb.Artifact(
                name=f"dataset-{stage_name}", 
                type="dataset", 
                description=f"Generated dataset for {stage_name}"
            )
            for file_path in data_dir.glob("*.jsonl"):
                dataset_artifact.add_file(str(file_path))
            
            # 버전을 갱신하기 위한 더미 파일 추가
            dataset_artifact.add_file(str(dummy_file))
            
            run_data.log_artifact(dataset_artifact)
            print(f"  ✅ [Data Node] dataset-{stage_name} (New Version) 로깅됨")
        run_data.finish()
        
        # ── 2. Training Run (훈련 파이프라인 노드) ──
        run_train = wandb.init(
            project=PROJECT_NAME, 
            job_type="train", 
            name=f"{stage_name}_training_v2",
            reinit=True
        )
        
        # [INPUT 1] 방금 생성한(혹은 과거에 생성했던) 데이터셋 사용 명시
        if dataset_artifact:
            run_train.use_artifact(f"dataset-{stage_name}:latest")
            print(f"  🔗 [Lineage] Input: dataset-{stage_name} 연결 완료")
            
        # [INPUT 2] 이전 스테이지의 졸업 모델 사용 명시 (Stage 0 제외)
        if stage > 0:
            prev_stage = f"stage{stage-1}"
            try:
                run_train.use_artifact(f"model-{prev_stage}:latest")
                print(f"  🔗 [Lineage] Input: model-{prev_stage} 연결 완료")
            except Exception as e:
                print(f"  ⚠️ 이전 모델 아티팩트 연결 실패: {e}")

        # [OUTPUT] 훈련 결과 모델을 새로운 아티팩트로 로깅
        checkpoints_dir = Path("checkpoints")
        stage_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith(f"{stage_name}_")]
        
        if stage_dirs:
            best_dir = stage_dirs[0] / stage_dirs[0].name / "best"
            if best_dir.exists():
                model_artifact = wandb.Artifact(
                    name=f"model-{stage_name}", 
                    type="model",
                    description=f"{stage_name} Best Checkpoint"
                )
                model_artifact.add_dir(str(best_dir))
                
                # 버전을 갱신하기 위한 더미 파일 추가
                model_artifact.add_file(str(dummy_file))
                
                run_train.log_artifact(model_artifact)
                print(f"  ✅ [Model Node] model-{stage_name} (New Version) 로깅됨 (Output)")
        run_train.finish()

    # 정리
    if dummy_file.exists():
        dummy_file.unlink()

    print("\n🎉 완벽한 커리큘럼 리니지 그래프(Lineage Graph V2) 구축 완료!")
    print(f"👉 대시보드 바로가기: https://wandb.ai/{wandb.api.viewer().get('entity')}/{PROJECT_NAME}")

if __name__ == "__main__":
    wandb_lineage_sync()
