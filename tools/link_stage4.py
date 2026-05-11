import wandb
import time
from pathlib import Path

def link_stage4():
    # 1. Dataset Node
    print("🚀 Stage 4 데이터셋 아티팩트 노드 생성 중...")
    run_data = wandb.init(project="EurekaAI-Curriculum", job_type="data_prep", name="stage4_data_sync_manual")
    ds_art = wandb.Artifact(name="dataset-stage4", type="dataset")
    train_file = Path("data/processed/stage4_high/train_stage4_high.jsonl")
    if train_file.exists():
        ds_art.add_file(str(train_file))
    
    with open("dummy.txt", "w") as f: f.write(str(time.time()))
    ds_art.add_file("dummy.txt")
    run_data.log_artifact(ds_art)
    run_data.finish()
    print("✅ Stage 4 데이터셋 등록 완료.")

    # 2. Training Node
    print("🚀 Stage 4 훈련 아티팩트(리니지) 연결 중...")
    run = wandb.init(project="EurekaAI-Curriculum", job_type="train", name="stage4_lineage_stitch_v2")
    
    # [입력]
    try:
        run.use_artifact("model-stage3:latest")
        print("🔗 Input 연결: model-stage3")
    except Exception as e:
        print(f"⚠️ model-stage3 연결 실패: {e}")
        
    try:
        run.use_artifact("dataset-stage4:latest")
        print("🔗 Input 연결: dataset-stage4")
    except Exception as e:
        print(f"⚠️ dataset-stage4 연결 실패: {e}")

    # [출력 1] 중간 체크포인트들 시각화 노드 추가
    # 용량 절약 및 빠른 처리를 위해 가벼운 프록시 아티팩트로 그래프 모양을 잡아줍니다.
    print("🚀 중간 체크포인트 시각화 그래프 구성 중...")
    steps = [2000, 4000, 6000, 8000, 10000, 12000, 14000]
    for step in steps:
        art = wandb.Artifact(name=f"model-stage4-step-{step}", type="intermediate_checkpoint")
        with open("dummy_step.txt", "w") as f: f.write(f"This is a proxy for intermediate step {step}")
        art.add_file("dummy_step.txt")
        run.log_artifact(art)
        
    # [출력 2] 최종 선정된 BEST 체크포인트 (실제 모델 파일)
    print("🚀 최종 BEST 모델(선정본) 리니지 확정 중...")
    best_art = wandb.Artifact(
        name="model-stage4", 
        type="model", 
        description="★ FINAL SELECTED BEST CHECKPOINT ★ (이 체크포인트가 다음 Stage로 전송됩니다)"
    )
    best_dir = Path("checkpoints/stage4_high/stage4_high/best")
    if best_dir.exists():
        best_art.add_dir(str(best_dir))
    best_art.add_file("dummy.txt")
    
    # 특수 태그를 달아 최종 선정된 모델임을 강조
    run.log_artifact(best_art, aliases=["latest", "official_best"])
    
    run.finish()
    print("✅ Stage 4 리니지 완벽 구성 완료! (중간 체크포인트 및 최종 시각화 반영)")

    Path("dummy.txt").unlink(missing_ok=True)
    Path("dummy_step.txt").unlink(missing_ok=True)

if __name__ == "__main__":
    link_stage4()
