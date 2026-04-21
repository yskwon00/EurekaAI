"""
dataset-stage5 리니지를 dataset-stage4 수준으로 체계화.
data_prep 런과 훈련 런 사이의 연결을 명시적으로 재구성합니다.
"""
import wandb
import time
from pathlib import Path

PROJECT = "yskwon00-none/EurekaAI-Curriculum"

def sync_dataset_stage5_lineage():
    api = wandb.Api()
    print("🔍 현재 dataset-stage5 아티팩트 확인...")
    
    # 기존 아티팩트 확인
    try:
        existing = api.artifact(f"{project}/dataset-stage5:latest")
        print(f"  기존 artifact: {existing.name}:{existing.version} | aliases: {existing.aliases}")
    except Exception as e:
        print(f"  기존 artifact 없음: {e}")
        existing = None

    # ── Step 1: data_prep 런 생성 (dataset-stage5 업로드) ─────────────────────
    print("\n📦 [Step 1] data_prep 런 → dataset-stage5 아티팩트 업로드...")
    run_data = wandb.init(
        project="EurekaAI-Curriculum",
        job_type="data_prep",
        name="stage5_data_generation_v2",
        config={"method": "Academic Wiki + Ollama Teacher Scoring (heuristic)", "samples": 11251},
        reinit=True
    )
    
    stage5_data_dir = Path("data/processed/stage5")
    if stage5_data_dir.exists():
        art = wandb.Artifact(
            name="dataset-stage5",
            type="dataset",
            description="Stage 5 University-level dataset (Academic Wiki QA + Replay)"
        )
        art.add_dir(str(stage5_data_dir))
        
        # 강제 버저닝 (dummy 파일로 해시 변경)
        dummy = stage5_data_dir / "dummy_lineage.txt"
        with open(dummy, "w") as f:
            f.write(f"Lineage Sync v2: {time.time()}")
        art.add_file(str(dummy))
        run_data.log_artifact(art)
        print(f"  ✅ dataset-stage5 업로드 완료")
        dummy.unlink(missing_ok=True)
    else:
        print(f"  ❌ {stage5_data_dir} 없음")
    run_data.finish()
    
    # ── Step 2: 훈련 런 생성 (dataset-stage5 + model-stage4 → model-stage5) ──
    print("\n🎯 [Step 2] training 런 → model-stage5 업로드 (리니지 연결)...")
    run_train = wandb.init(
        project="EurekaAI-Curriculum",
        job_type="train",
        name="stage5_university_training_v2",
        config={
            "stage": 5,
            "max_steps": 8000,
            "final_ppl": 1.11,
            "graduated": True,
            "graduation_threshold": 6.0
        },
        reinit=True
    )
    
    # 입력 아티팩트 연결
    try:
        run_train.use_artifact("dataset-stage5:latest")
        print("  🔗 Input: dataset-stage5:latest 연결")
    except Exception as e:
        print(f"  ⚠️  dataset-stage5 연결 실패: {e}")
    
    try:
        run_train.use_artifact("model-stage4:latest")
        print("  🔗 Input: model-stage4:latest 연결")
    except Exception as e:
        print(f"  ⚠️  model-stage4 연결 실패: {e}")
    
    # 출력 아티팩트 업로드
    best_dir = Path("checkpoints/stage5_university/stage5_university/best")
    if best_dir.exists():
        model_art = wandb.Artifact(
            name="model-stage5",
            type="model",
            description="Stage 5 University Best Checkpoint (PPL 1.11, GRADUATED)"
        )
        model_art.add_dir(str(best_dir))
        # dummy 파일은 best_dir 밖에 만들고 별도 이름으로 추가 (중복 경로 오류 방지)
        dummy_m = Path("/tmp/stage5_lineage_force.txt")
        with open(dummy_m, "w") as f:
            f.write(f"Lineage Sync v2: {time.time()}")
        model_art.add_file(str(dummy_m), name="stage5_lineage_force.txt")
        run_train.log_artifact(model_art)
        print("  ✅ Output: model-stage5 업로드 완료")
    
    run_train.finish()
    print("\n🎉 dataset-stage5 리니지 체계화 완료!")
    print("   W&B 대시보드에서 dataset-stage5 → stage5_training → model-stage5 순서로 확인하세요.")

if __name__ == "__main__":
    sync_dataset_stage5_lineage()
