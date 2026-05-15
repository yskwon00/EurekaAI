"""
EurekaAI — Master Lineage Builder (Stage 0 ~ 5)
==================================================
과거 Stage 0부터 Stage 5까지의 모든 데이터 조합 과정과 학습 과정을
프록시(Proxy) 아티팩트를 사용하여 하나의 완벽한 거대 DAG로 재구성합니다.
"""
import time
import wandb
from pathlib import Path

PROJECT = "EurekaAI-Curriculum"
ENTITY = "yskwon00-none"
DUMMY_DIR = Path("dummy_lineage")
DUMMY_DIR.mkdir(exist_ok=True)

def get_dummy(name):
    path = DUMMY_DIR / f"{name}.txt"
    path.write_text(f"Master Lineage Node: {name} | ts: {time.time()}")
    return str(path)

def main():
    print("🚀 EurekaAI Master Lineage 재구성 시작...")

    # ─────────────────────────────────────────────────────────────
    # 1. 원시 데이터(Raw Data) 노드 생성
    # ─────────────────────────────────────────────────────────────
    print("\n[1] 원시 데이터(Raw Data) 등록 중...")
    raw_arts = {}
    for raw in ["raw-tinystories", "raw-wikipedia", "raw-sharegpt"]:
        run = wandb.init(project=PROJECT, job_type="raw_ingest", name=f"ingest_{raw}")
        art = wandb.Artifact(raw, type="raw_data", description=f"Original source for {raw}")
        art.add_file(get_dummy(raw))
        run.log_artifact(art)
        run.finish()
        raw_arts[raw] = f"{raw}:latest"
        print(f"  ✅ {raw}")

    # ─────────────────────────────────────────────────────────────
    # 2. Stage 0 ~ 5 순차적 재구성 (Data Mixing -> Training)
    # ─────────────────────────────────────────────────────────────
    prev_dataset = None
    prev_model = None

    stage_configs = {
        0: {"raw": ["raw-tinystories"], "syn": "synthetic-basic"},
        1: {"raw": ["raw-tinystories"], "syn": "synthetic-toddler"},
        2: {"raw": ["raw-tinystories", "raw-wikipedia"], "syn": "synthetic-qa"},
        3: {"raw": ["raw-wikipedia"], "syn": "synthetic-cot"},
        4: {"raw": ["raw-wikipedia", "raw-sharegpt"], "syn": "synthetic-essay"},
        5: {"raw": ["raw-wikipedia", "raw-sharegpt"], "syn": "synthetic-academic"},
        6: {"raw": ["raw-wikipedia", "raw-sharegpt"], "syn": "synthetic-rlhf"},
    }

    for stage in range(7):
        print(f"\n[Stage {stage}] 리니지 구성 중...")
        conf = stage_configs[stage]

        # ── A. 합성 데이터(Synthetic) 생성 노드 ──
        syn_name = conf["syn"]
        run_syn = wandb.init(project=PROJECT, job_type="synthetic_gen", name=f"stage{stage}_syn_gen")
        art_syn = wandb.Artifact(syn_name, type="synthetic_data", description=f"Stage {stage} synthetic by Teacher")
        art_syn.add_file(get_dummy(syn_name))
        run_syn.log_artifact(art_syn)
        run_syn.finish()
        syn_art_id = f"{syn_name}:latest"
        print(f"  ✅ Synthetic Data : {syn_name}")

        # ── B. Data Mixing 런 ──
        run_mix = wandb.init(project=PROJECT, job_type="data_mixing", name=f"stage{stage}_data_mix")
        # Input 연결
        run_mix.use_artifact(syn_art_id)
        for r in conf["raw"]:
            run_mix.use_artifact(raw_arts[r])
        if prev_dataset:
            run_mix.use_artifact(prev_dataset) # Replay
        
        # Output 생성
        ds_name = f"dataset-stage{stage}-master"
        art_ds = wandb.Artifact(ds_name, type="dataset", description=f"Stage {stage} Master Dataset")
        art_ds.add_file(get_dummy(ds_name))
        run_mix.log_artifact(art_ds)
        run_mix.finish()
        prev_dataset = f"{ds_name}:latest"
        print(f"  ✅ Data Mixing    : {ds_name} (Inputs: {conf['raw']} + {syn_name} + Replay)")

        # ── C. Training 런 ──
        run_train = wandb.init(project=PROJECT, job_type="train_master", name=f"stage{stage}_train_master")
        run_train.use_artifact(prev_dataset)
        if prev_model:
            run_train.use_artifact(prev_model)
            
        mod_name = f"model-stage{stage}-master"
        art_mod = wandb.Artifact(mod_name, type="model", description=f"Stage {stage} Master Model")
        art_mod.add_file(get_dummy(mod_name))
        aliases = ["master_lineage", f"stage{stage}"]
        if stage == 6:
            aliases.append("official_best")
        run_train.log_artifact(art_mod, aliases=aliases)
        run_train.finish()
        prev_model = f"{mod_name}:latest"
        print(f"  ✅ Training       : {mod_name}")

    print(f"\n🎉 전체 Master Lineage 구성 완료!")
    print(f"👉 확인 링크: https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/model/model-stage6-master/latest/lineage")

if __name__ == "__main__":
    main()
