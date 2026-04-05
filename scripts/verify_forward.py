#!/usr/bin/env python3
"""Verify APR forward pass produces reasonable loss on known input.

PMAT-497: APR loss starts at 18.9 (worse than random 11.93 for vocab=151936).
This script computes a reference loss using HuggingFace transformers and
compares to the loss reported by APR training on the same sample.

Usage:
    uv run --extra cuda python scripts/verify_forward.py
    uv run --extra cuda python scripts/verify_forward.py --apr-result results/canary-apr-gx10-profile-20260405.json
"""

import argparse
import json
import math
import sys


def compute_reference_loss():
    """Compute cross-entropy loss on first canary sample using HF model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("error: torch/transformers not available. Run with: uv run --extra cuda python scripts/verify_forward.py")
        sys.exit(1)

    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()

    # Use first canary sample
    import yaml
    with open("prompts/canary-dataset.yaml") as f:
        data = yaml.safe_load(f)
    sample = data["samples"][0]
    text = f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n{sample['response']}<|im_end|>"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Standard next-token prediction: logits[:-1] predicts labels[1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_val = loss.item()

    # Also compute loss for random logits (expected = ln(vocab_size))
    random_loss = math.log(logits.shape[-1])

    print(f"\nReference forward pass:")
    print(f"  model: {model_id}")
    print(f"  seq_len: {seq_len}")
    print(f"  vocab_size: {logits.shape[-1]}")
    print(f"  reference loss: {loss_val:.4f}")
    print(f"  random baseline: {random_loss:.4f}")
    print(f"  ratio (ref/random): {loss_val/random_loss:.2f}")

    return loss_val, random_loss


def check_apr_result(apr_path, ref_loss, random_loss):
    """Compare APR result loss to reference."""
    with open(apr_path) as f:
        result = json.load(f)

    apr_loss = result["metrics"].get("final_loss", 0)
    if apr_loss == 0:
        print(f"\nAPR result has no loss data ({apr_path})")
        return

    print(f"\nAPR vs Reference (F-CONV-01):")
    print(f"  APR final loss:     {apr_loss:.4f}")
    print(f"  Reference loss:     {ref_loss:.4f}")
    print(f"  Random baseline:    {random_loss:.4f}")
    print(f"  APR/reference:      {apr_loss/ref_loss:.1f}x")
    print(f"  APR/random:         {apr_loss/random_loss:.1f}x")

    if apr_loss > random_loss:
        print(f"\n  FALSIFIED: APR loss ({apr_loss:.2f}) > random ({random_loss:.2f})")
        print(f"  The model is actively predicting WRONG tokens with high confidence.")
        print(f"  Root cause: Q4K→F32 dequantization likely corrupts weight values.")
        return False
    elif apr_loss > ref_loss * 3:
        print(f"\n  WARNING: APR loss ({apr_loss:.2f}) is {apr_loss/ref_loss:.0f}x higher than reference")
        print(f"  Training may converge but is significantly suboptimal.")
        return False
    else:
        print(f"\n  PASS: APR loss ({apr_loss:.2f}) is within expected range")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apr-result", default=None,
                        help="Path to APR canary result JSON")
    args = parser.parse_args()

    ref_loss, random_loss = compute_reference_loss()

    if args.apr_result:
        check_apr_result(args.apr_result, ref_loss, random_loss)
    else:
        # Check latest APR result
        import glob
        import os
        apr_results = sorted(
            glob.glob("results/canary-apr-*.json"),
            key=os.path.getmtime, reverse=True
        )
        if apr_results:
            check_apr_result(apr_results[0], ref_loss, random_loss)
        else:
            print("\nNo APR results found. Run: make canary-apr-gx10")


if __name__ == "__main__":
    main()
