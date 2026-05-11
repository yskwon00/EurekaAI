import json
from pathlib import Path

def force_graduate():
    prog_path = Path("checkpoints/progression.json")
    if not prog_path.exists():
        print("progression.json not found")
        return

    with open(prog_path, "r") as f:
        data = json.load(f)
    
    if "3" in data["records"]:
        print("Force marking Stage 3 as completed...")
        data["records"]["3"]["status"] = "completed"
        data["records"]["3"]["checkpoint_path"] = "checkpoints/stage3_middle/stage3_middle/best"
    
    with open(prog_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print("Stage 3 force-graduated successfully!")

if __name__ == "__main__":
    force_graduate()
