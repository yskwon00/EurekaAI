import wandb

def delete_v1_artifact():
    api = wandb.Api()
    project = "yskwon00-none/EurekaAI-Curriculum"
    
    try:
        # v0를 가져와서 latest 태그를 강제로 v0에게 부여함
        print("Restoring latest alias to v0...")
        art_v0 = api.artifact(f"{project}/dataset-stage4:v0")
        art_v0.aliases.append("latest")
        art_v0.save()
        print("v0 is now latest.")
    except Exception as e:
        print(f"Error shifting tag: {e}")
        
    try:
        print("Fetching dataset-stage4:v1 ...")
        # 이제 v1은 latest가 아니므로 편하게 지울 수 있음
        artifact = api.artifact(f"{project}/dataset-stage4:v1")
        if "latest" in artifact.aliases:
            artifact.aliases.remove("latest")
            artifact.save()
            print("Removed latest from v1 manually.")
            
        print("Deleting dataset-stage4:v1 ...")
        artifact.delete()
        print("✅ 삭제 성공!")
    except Exception as e:
        print(f"Error deleting v1: {e}")

if __name__ == "__main__":
    delete_v1_artifact()
