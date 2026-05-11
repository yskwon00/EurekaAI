import wandb

def test_stitch():
    project = "yskwon00-none/EurekaAI-Curriculum"
    stage = 0
    print(f"Stitching Stage {stage}...")
    run = wandb.init(project="EurekaAI-Curriculum", job_type="train", name=f"stage{stage}_lineage_stitch_v3")
    
    # 1. Inputs
    try:
        run.use_artifact(f"dataset-stage{stage}:latest")
    except Exception as e:
        print(f"Dataset error: {e}")
        
    # 2. Output
    out_art = wandb.Artifact(name=f"model-stage{stage}", type="model")
    try:
        out_art.add_reference(f"wandb-artifact://{project}/model-stage{stage}:latest")
        run.log_artifact(out_art)
        print("Success adding reference!")
    except Exception as e:
        print(f"Reference error: {e}")
    
    run.finish()

if __name__ == "__main__":
    test_stitch()
