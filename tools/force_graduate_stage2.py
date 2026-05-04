import json
from pathlib import Path

def force_graduate():
    prog_path = Path("checkpoints/progression.json")
    if not prog_path.exists():
        print("progression.json not found")
        return

    with open(prog_path, "r") as f:
        data = json.load(f)
    
    if "2" in data["records"]:
        print("Force marking Stage 2 as completed...")
        data["records"]["2"]["status"] = "completed"
        data["records"]["2"]["checkpoint_path"] = "checkpoints/stage2_elementary/stage2_elementary/best"
    
    with open(prog_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print("Stage 2 force-graduated successfully!")

if __name__ == "__main__":
    force_graduate()
