"""
EurekaAI — Evaluation Benchmarks
Per-stage automated evaluation metrics.
"""

import math
import json
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_perplexity(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return ppl


def compute_qa_accuracy(
    model,
    tokenizer,
    qa_samples: list[dict],
    device: torch.device,
    max_new_tokens: int = 64,
) -> float:
    """
    Approximate QA accuracy: check if correct answer appears in generation.
    Used for Stage 2+ evaluation.
    """
    model.eval()
    correct = 0

    for sample in qa_samples[:100]:  # Cap at 100 for speed
        question = sample.get("question", "")
        answer = sample.get("answer", "").strip().lower()

        prompt = f"### 질문\n{question}\n\n### 답\n"
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.1, top_k=5)

        generated = tokenizer.decode(out[0][len(ids):].tolist()).lower()

        # Simple substring match (EM-lite)
        if answer in generated or generated in answer:
            correct += 1

    return correct / max(len(qa_samples[:100]), 1)


def run_stage_benchmark(
    stage: int,
    model,
    tokenizer,
    device: torch.device,
    eval_dataloader: Optional[DataLoader] = None,
    qa_samples: Optional[list[dict]] = None,
) -> dict:
    """
    Run the appropriate benchmark for a given stage.

    Returns:
        dict with metric values (e.g., {"ppl": 25.0} or {"f1": 0.65})
    """
    metrics = {}

    # PPL — all stages
    if eval_dataloader:
        ppl = compute_perplexity(model, eval_dataloader, device)
        metrics["ppl"] = ppl
        metrics["eval_loss"] = math.log(ppl)
        logger.info(f"  Stage {stage} | PPL: {ppl:.2f}")

    # Stage-specific
    if stage == 0:
        # Stage 0: PPL < 30
        pass

    elif stage == 1:
        # Stage 1: Basic vocabulary coverage (proxy via PPL + generation check)
        pass

    elif stage >= 2 and qa_samples:
        # Stage 2+: QA accuracy
        acc = compute_qa_accuracy(model, tokenizer, qa_samples, device)
        metrics["accuracy"] = acc
        # Approximate F1 (same as accuracy for now — full F1 needs token-level scoring)
        metrics["f1"] = acc
        logger.info(f"  Stage {stage} | QA Accuracy: {acc:.3f}")

    return metrics


# ── Quick benchmark runner ───────────────────────────────────────────────────────

def quick_stage_eval(model, tokenizer, device, stage: int) -> dict:
    """
    Run a super-fast sanity check across all stages.
    Tests basic generation quality without a full dataset.
    """
    test_prompts = {
        0: ["안", "He", "나"],
        1: ["안녕하세요", "Hello", "사과는"],
        2: ["1 더하기 1은", "What is 2+2?", "한국의 수도는"],
        3: ["피타고라스 정리를 설명해줘", "Explain photosynthesis", "이차방정식 풀이"],
        4: ["기후변화의 원인과 해결책", "Analyze the causes of the French Revolution"],
        5: ["양자역학의 불확정성 원리", "Explain transformer attention mechanism"],
        6: ["오늘 기분이 어때요?", "What are the pros and cons of remote work?"],
    }

    prompts = test_prompts.get(stage, test_prompts[0])
    results = []

    model.eval()
    for p in prompts:
        ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=40, temperature=0.8, top_k=30)
        generated = tokenizer.decode(out[0].tolist())
        results.append({"prompt": p, "generated": generated})
        print(f"  [{p[:20]}] → {generated[:80]}")

    return {"samples": results}
