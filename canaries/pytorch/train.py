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
import torch.profiler
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
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler for parity profiling (PMAT-487)")
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

    # PMAT-487: Parity profiling with torch.profiler
    profile_data = {}
    if args.profile and torch.cuda.is_available():
        print("\nRunning profiled steps for parity analysis...")
        warmup_steps = 2
        active_steps = min(10, max(args.steps - warmup_steps, 5))
        data_iter2 = iter(dataloader)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=warmup_steps, active=active_steps
            ),
            with_flops=True,
        ) as prof:
            for ps in range(warmup_steps + active_steps):
                try:
                    batch = next(data_iter2)
                except StopIteration:
                    data_iter2 = iter(dataloader)
                    batch = next(data_iter2)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                prof.step()

        # Parse profiler output into parity-profile-v1 schema
        ka = prof.key_averages()

        def cuda_time(e):
            """Get CUDA time from profiler event (API varies by PyTorch version)."""
            for attr in ["self_cuda_time_total", "cuda_time_total", "cuda_time", "self_device_time_total"]:
                val = getattr(e, attr, None)
                if val is not None and val > 0:
                    return val
            return 0

        total_cuda_us = sum(cuda_time(e) for e in ka)
        kernel_count = sum(e.count for e in ka if cuda_time(e) > 0)

        # Map kernels to operation categories
        op_times = {"attention_ms": 0, "ffn_ms": 0, "norm_ms": 0, "embed_ms": 0, "projection_ms": 0, "other_ms": 0}
        for e in ka:
            t_ms = cuda_time(e) / 1000.0 / active_steps
            name = e.key.lower()
            if any(k in name for k in ["attention", "softmax", "flash", "sdpa"]):
                op_times["attention_ms"] += t_ms
            elif any(k in name for k in ["layer_norm", "rms_norm", "norm"]):
                op_times["norm_ms"] += t_ms
            elif any(k in name for k in ["embedding", "embed"]):
                op_times["embed_ms"] += t_ms
            elif any(k in name for k in ["mm", "gemm", "linear", "addmm", "bmm"]):
                # Distinguish projection vs FFN by shape if available
                op_times["ffn_ms"] += t_ms
            else:
                op_times["other_ms"] += t_ms

        step_cuda_ms = total_cuda_us / 1000.0 / active_steps
        total_ops = sum(op_times.values())
        profile_data = {
            "_schema": "parity-profile-v1",
            "runtime": "pytorch",
            "steps_profiled": active_steps,
            "step_time_ms": {"mean": round(statistics.mean(step_times), 1)},
            "ops": {k: {"mean": round(v, 2), "pct": round(v / total_ops * 100, 1) if total_ops > 0 else 0}
                    for k, v in op_times.items() if v > 0},
            "hardware": {
                "kernel_launches_per_step": kernel_count // active_steps,
                "total_cuda_time_ms": round(step_cuda_ms, 1),
            },
        }
        print(f"  profiled {active_steps} steps: {kernel_count // active_steps} kernel launches/step, {step_cuda_ms:.1f}ms CUDA/step")

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
        **({"profile": profile_data} if profile_data else {}),
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
