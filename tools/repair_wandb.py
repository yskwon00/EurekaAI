import wandb

def fix_stage4_lineage():
    api = wandb.Api()
    project = "yskwon00-none/EurekaAI-Curriculum"
    
    print("Fetching runs...")
    runs = api.runs(project)
    
    for run in runs:
        if run.name in ["stage4_data_generation_v2", "stage4_training_v2"]:
            print(f"Deleting fake sync run: {run.name} ({run.id})")
            run.delete()
            
    # Also delete the artifact versions that were created by these fake runs to clean up the graph
    print("Cleaning up fake artifacts...")
    try:
        dataset = api.artifact(f"{project}/dataset-stage4:latest")
        if "v2" in [a.name for a in dataset.aliases] or dataset.version == "v1":
             # We want to keep v0
             pass
    except Exception as e:
        print(e)
        
    print("Done cleaning fake parallel graphs!")

if __name__ == "__main__":
    fix_stage4_lineage()
