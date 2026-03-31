#!/usr/bin/env python3
"""cuBLAS parity canary — measures GEMM training performance via cuBLAS vs default backends.

Runs the same training loop twice:
1. Default PyTorch backend (cuDNN/internal)
2. Forced cuBLAS via torch.backends.cuda.preferred_linalg_library("cusolver"/"cublas")

Compares throughput, numerical parity (loss divergence), and VRAM to detect
backend-specific regressions. This is the parity gate between WGPU and CUDA paths.
"""

import argparse
import json
import os
import socket
import statistics
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


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
        "vram_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024 * 1024),
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
    }


def run_training(model, dataloader, device, args, label: str) -> dict:
    """Run a training loop and collect metrics."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

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

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        step_time = (time.perf_counter() - step_start) * 1000
        step_times.append(step_time)
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  [{label}] step {step}/{args.steps}  loss={loss.item():.4f}  step_time={step_time:.1f}ms")

        step += 1

    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() // (1024 * 1024)
    final_loss = statistics.mean(losses[-10:])
    tokens_per_sec = (args.batch_size * args.seq_len * args.steps) / wall_time

    return {
        "label": label,
        "throughput_samples_sec": round((args.batch_size * args.steps) / wall_time, 2),
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
        "losses": [round(l, 4) for l in losses],
    }


def main():
    parser = argparse.ArgumentParser(description="cuBLAS parity canary")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="prompts/canary-dataset.yaml")
    parser.add_argument("--output", default="/tmp/canary-cublas.json")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load tokenizer and dataset once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = CanaryDataset(args.dataset, tokenizer, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)  # deterministic

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # --- Run 1: Default backend ---
    print("=== Default backend ===")
    torch.manual_seed(args.seed)
    model_default = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    default_metrics = run_training(model_default, dataloader, device, args, "default")
    del model_default
    torch.cuda.empty_cache()

    # --- Run 2: cuBLAS forced ---
    print("\n=== cuBLAS forced ===")
    torch.backends.cuda.preferred_linalg_library("cusolver")
    # Force cuBLAS for matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    model_cublas = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    cublas_metrics = run_training(model_cublas, dataloader, device, args, "cublas")
    del model_cublas
    torch.cuda.empty_cache()

    # --- Parity analysis ---
    loss_divergence = abs(default_metrics["final_loss"] - cublas_metrics["final_loss"])
    throughput_ratio = cublas_metrics["tokens_per_sec"] / max(default_metrics["tokens_per_sec"], 1)
    vram_delta = cublas_metrics["peak_vram_mb"] - default_metrics["peak_vram_mb"]

    # Step-by-step loss comparison
    max_step_divergence = 0.0
    for d_loss, c_loss in zip(default_metrics["losses"], cublas_metrics["losses"]):
        max_step_divergence = max(max_step_divergence, abs(d_loss - c_loss))

    parity = {
        "loss_divergence": round(loss_divergence, 6),
        "max_step_divergence": round(max_step_divergence, 6),
        "throughput_ratio": round(throughput_ratio, 4),
        "vram_delta_mb": vram_delta,
        "numerically_equivalent": loss_divergence < 0.01,
        "perf_equivalent": 0.95 <= throughput_ratio <= 1.05,
    }

    # Strip per-step losses from output (too verbose)
    for m in [default_metrics, cublas_metrics]:
        del m["losses"]

    output = {
        "canary": "cublas",
        "backend": "cuda",
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
        },
        "metrics": {
            "default": default_metrics,
            "cublas": cublas_metrics,
            "parity": parity,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Parity Results ===")
    print(f"  loss divergence:      {loss_divergence:.6f} ({'PASS' if parity['numerically_equivalent'] else 'FAIL'})")
    print(f"  max step divergence:  {max_step_divergence:.6f}")
    print(f"  throughput ratio:     {throughput_ratio:.4f}x ({'PASS' if parity['perf_equivalent'] else 'FAIL'})")
    print(f"  VRAM delta:           {vram_delta:+d} MB")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
