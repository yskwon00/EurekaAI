import wandb
from pathlib import Path
import time

PROJECT_NAME = "EurekaAI-Curriculum"

def upload_stage4_dataset():
    run = wandb.init(
        project=PROJECT_NAME, 
        job_type="data_prep", 
        name="stage4_data_generation",
        config={"method": "Ollama Teacher Essay & Advanced Instruction Tuning", "samples": 13623}
    )
    
    data_dir = Path("data/processed/stage4")
    if data_dir.exists():
        dataset_artifact = wandb.Artifact(
            name="dataset-stage4", 
            type="dataset", 
            description="Generated dataset for stage4"
        )
        for file_path in data_dir.glob("*.jsonl"):
            dataset_artifact.add_file(str(file_path))
            
        dummy_file = Path("dummy_lineage.txt")
        with open(dummy_file, "w") as f:
            f.write(f"Lineage Hash Force: {time.time()}")
        dataset_artifact.add_file(str(dummy_file))
        
        run.log_artifact(dataset_artifact)
        print("✅ dataset-stage4 업로드 완료!")
        
        dummy_file.unlink()
    
    run.finish()

if __name__ == "__main__":
    upload_stage4_dataset()
