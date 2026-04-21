import wandb

def fix_model_stage4_garbage():
    api = wandb.Api()
    project = "yskwon00-none/EurekaAI-Curriculum"
    
    print("Restoring latest alias to v1 (True Final) ...")
    try:
        art_v1 = api.artifact(f"{project}/model-stage4:v1")
        art_v1.aliases.append("latest")
        art_v1.save()
        print("v1 is now latest.")
    except Exception as e:
        print(f"Error shifting tag to v1: {e}")

    try:
        print("Fetching model-stage4:v2 ...")
        # 이제 v2는 latest가 아니므로 편하게 지울 수 있음
        artifact = api.artifact(f"{project}/model-stage4:v2")
        if "latest" in artifact.aliases:
            artifact.aliases.remove("latest")
            artifact.save()
            print("Removed latest from v2 manually.")
            
        print("Deleting model-stage4:v2 ...")
        artifact.delete()
        print("✅ 삭제 성공!")
    except Exception as e:
        print(f"Error deleting v2: {e}")

if __name__ == "__main__":
    fix_model_stage4_garbage()
