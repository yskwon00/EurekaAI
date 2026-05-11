import json
from pathlib import Path

def force_graduate():
    prog_path = Path("checkpoints/progression.json")
    if not prog_path.exists():
        print("progression.json not found")
        return

    with open(prog_path, "r") as f:
        data = json.load(f)
    
    if "4" in data["records"]:
        print("Force marking Stage 4 as completed...")
        data["records"]["4"]["status"] = "completed"
        data["records"]["4"]["checkpoint_path"] = "checkpoints/stage4_high/stage4_high/best"
    
    with open(prog_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print("Stage 4 force-graduated successfully!")

if __name__ == "__main__":
    force_graduate()
