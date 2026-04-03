#!/usr/bin/env python3
"""PyTorch baseline canary — full fine-tune benchmark for Qwen2.5-Coder-1.5B."""

import argparse
import json
import time
import os
import socket
import statistics
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


class CanaryDataset(TorchDataset):
    """Simple text dataset from canary YAML."""

    def __init__(self, path: str, tokenizer, seq_len: int):
        with open(path) as f:
            data = yaml.safe_load(f)

        self.examples = []
        for sample in data["samples"]:
            text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
    }


def main():
    parser = argparse.ArgumentParser(description="PyTorch baseline canary")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="prompts/canary-dataset.yaml")
    parser.add_argument("--output", default="/tmp/canary-pytorch.json")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (PMAT-426)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps (PMAT-428/459)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (full precision for baseline)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    ).to(device)

    # Gradient checkpointing required on <=8GB VRAM (yoga: full FT exceeds 8GB without it)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb <= 16:
            model.gradient_checkpointing_enable()

    # torch.compile for graph-mode optimization (PMAT-426)
    if args.compile:
        model = torch.compile(model)

    model.train()

    # Dataset & dataloader
    dataset = CanaryDataset(args.dataset, tokenizer, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer — use 8-bit on constrained VRAM (F-EXEC-02: full FT OOMs with fp32 optimizer states)
    use_8bit = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / (1024**3) <= 16
    if use_8bit and HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        optim_name = "adamw_8bit"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        optim_name = "adamw"

    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Reset peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Training loop
    step = 0
    step_times = []
    losses = []
    data_iter = iter(dataloader)

    t0 = time.perf_counter()
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        step_start = time.perf_counter()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss = loss * args.gradient_accumulation_steps  # un-scale for logging

        step_time = (time.perf_counter() - step_start) * 1000  # ms
        step_times.append(step_time)
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"step {step}/{args.steps}  loss={loss.item():.4f}  step_time={step_time:.1f}ms")

        step += 1

    wall_time = time.perf_counter() - t0

    # Metrics
    peak_vram = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
    final_loss = statistics.mean(losses[-10:])  # average of last 10 steps
    tokens_per_sec = (args.batch_size * args.seq_len * args.steps) / wall_time
    samples_per_sec = (args.batch_size * args.steps) / wall_time

    output = {
        "canary": "pytorch-compile" if args.compile else "pytorch",
        "backend": "cuda" if torch.cuda.is_available() else "cpu",
        "host": socket.gethostname(),
        "gpu": get_gpu_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "dtype": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
            "optimizer": optim_name,
            "quantization": "none",
            "compiled": args.compile,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "metrics": {
            "throughput_samples_sec": round(samples_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mb": peak_vram,
            "final_loss": round(final_loss, 4),
            "step_time_ms": {
                "mean": round(statistics.mean(step_times), 1),
                "p50": round(sorted(step_times)[len(step_times) // 2], 1),
                "p95": round(sorted(step_times)[int(len(step_times) * 0.95)], 1),
                "p99": round(sorted(step_times)[int(len(step_times) * 0.99)], 1),
            },
            "wall_time_sec": round(wall_time, 2),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nCanary complete: {args.steps} steps in {wall_time:.1f}s")
    print(f"  throughput: {tokens_per_sec:.0f} tok/s ({samples_per_sec:.1f} samples/s)")
    print(f"  peak VRAM: {peak_vram} MB")
    print(f"  final loss: {final_loss:.4f}")
    print(f"  step time: {statistics.mean(step_times):.1f}ms mean, {sorted(step_times)[int(len(step_times)*0.95)]:.1f}ms p95")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
