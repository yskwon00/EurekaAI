"""
EurekaAI — Model Generation Tester
Tests the model's generation quality at any stage.
"""

import sys
import torch
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).parent))

from core.model.architecture import EurekaModel
from core.model.config import EurekaConfig
from core.model.tokenizer_utils import EurekaTokenizer

def generate_sample(checkpoint_path: str, prompt: str, max_new_tokens: int = 50):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = EurekaTokenizer("data/tokenizer/eureka.model")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = EurekaConfig(**ckpt["config"])
    model = EurekaModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Simple greedy search
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.sp.eos_id():
                break
                
    decoded = tokenizer.decode(input_ids[0].tolist())
    print(f"Generated: {decoded}")
    return decoded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/stage0_newborn/best/model.pt")
    parser.add_argument("--prompt", type=str, default="A B C")
    args = parser.parse_args()
    
    if Path(args.ckpt).exists():
        generate_sample(args.ckpt, args.prompt)
    else:
        print(f"Checkpoint not found: {args.ckpt}")
