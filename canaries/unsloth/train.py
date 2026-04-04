#!/usr/bin/env python3
"""Unsloth QLoRA canary — fast fine-tuning benchmark for Qwen2.5-Coder-1.5B."""

import argparse
import json
import statistics
import time
import os
import socket
from datetime import datetime, timezone

import torch
import torch.profiler
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from transformers import TrainerCallback
import yaml


class StepTimerCallback(TrainerCallback):
    """Capture per-step wall time for latency percentiles."""

    def __init__(self):
        self.step_times = []
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self.step_times.append((time.perf_counter() - self._step_start) * 1000)


def load_canary_dataset(path: str, seq_len: int) -> Dataset:
    """Load canary dataset from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    records = []
    for sample in data["samples"]:
        text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"
        records.append({"text": text})

    return Dataset.from_list(records)


def get_gpu_info() -> dict:
    """Collect GPU metadata."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
    }


def main():
    parser = argparse.ArgumentParser(description="Unsloth QLoRA canary")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="prompts/canary-dataset.yaml")
    parser.add_argument("--output", default="/tmp/canary-unsloth.json")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler for parity profiling (PMAT-487)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model with unsloth 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    dataset = load_canary_dataset(args.dataset, args.seq_len)

    # Training args
    training_args = TrainingArguments(
        output_dir="/tmp/canary-unsloth-ckpt",
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        seed=args.seed,
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
    )

    timer = StepTimerCallback()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        callbacks=[timer],
    )

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Run training
    t0 = time.perf_counter()
    result = trainer.train()
    wall_time = time.perf_counter() - t0

    # PMAT-487: Parity profiling with torch.profiler
    profile_data = {}
    if args.profile and torch.cuda.is_available():
        print("\nRunning profiled steps for parity analysis...")
        warmup_steps = 2
        active_steps = min(10, max(args.steps - warmup_steps, 5))

        # Re-run a few steps under profiler (SFTTrainer doesn't support mid-train profiling)
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
            # Manual profiled steps using the same model
            train_ds_iter = iter(trainer.get_train_dataloader())
            for ps in range(warmup_steps + active_steps):
                try:
                    batch = next(train_ds_iter)
                except StopIteration:
                    train_ds_iter = iter(trainer.get_train_dataloader())
                    batch = next(train_ds_iter)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                model.zero_grad()
                prof.step()

        # Parse into parity-profile-v1 schema
        ka = prof.key_averages()
        total_cuda_us = sum(e.self_cuda_time_total for e in ka if e.self_cuda_time_total > 0)
        kernel_count = sum(e.count for e in ka if e.self_cuda_time_total > 0)

        op_times = {"attention_ms": 0, "ffn_ms": 0, "norm_ms": 0, "embed_ms": 0, "other_ms": 0}
        for e in ka:
            t_ms = e.self_cuda_time_total / 1000.0 / active_steps
            name = e.key.lower()
            if any(k in name for k in ["attention", "softmax", "flash", "sdpa", "triton_flash"]):
                op_times["attention_ms"] += t_ms
            elif any(k in name for k in ["layer_norm", "rms_norm", "norm"]):
                op_times["norm_ms"] += t_ms
            elif any(k in name for k in ["embedding", "embed"]):
                op_times["embed_ms"] += t_ms
            elif any(k in name for k in ["mm", "gemm", "linear", "addmm", "bmm", "triton_"]):
                op_times["ffn_ms"] += t_ms
            else:
                op_times["other_ms"] += t_ms

        step_cuda_ms = total_cuda_us / 1000.0 / active_steps
        total_ops = sum(op_times.values())
        profile_data = {
            "_schema": "parity-profile-v1",
            "runtime": "unsloth",
            "steps_profiled": active_steps,
            "ops": {k: {"mean": round(v, 2), "pct": round(v / total_ops * 100, 1) if total_ops > 0 else 0}
                    for k, v in op_times.items() if v > 0},
            "hardware": {
                "kernel_launches_per_step": kernel_count // active_steps,
                "total_cuda_time_ms": round(step_cuda_ms, 1),
            },
        }
        print(f"  profiled {active_steps} steps: {kernel_count // active_steps} launches/step, {step_cuda_ms:.1f}ms CUDA/step")

    # Collect metrics
    peak_vram = torch.cuda.max_memory_allocated() // (1024 * 1024)
    train_loss = result.training_loss
    samples_per_sec = (args.batch_size * args.steps) / wall_time
    tokens_per_sec = (args.batch_size * args.seq_len * args.steps) / wall_time

    # Step time percentiles
    st = timer.step_times if timer.step_times else [0.0]
    sorted_st = sorted(st)
    step_time_ms = {
        "mean": round(statistics.mean(st), 1),
        "p50": round(sorted_st[len(sorted_st) // 2], 1),
        "p95": round(sorted_st[int(len(sorted_st) * 0.95)], 1),
        "p99": round(sorted_st[int(len(sorted_st) * 0.99)], 1),
    }

    output = {
        "canary": "unsloth",
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
            "optimizer": "adamw_8bit",
            "quantization": "nf4",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        },
        "metrics": {
            "throughput_samples_sec": round(samples_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mb": peak_vram,
            "final_loss": round(train_loss, 4),
            "step_time_ms": step_time_ms,
            "wall_time_sec": round(wall_time, 2),
        },
        **({"profile": profile_data} if profile_data else {}),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Canary complete: {args.steps} steps in {wall_time:.1f}s")
    print(f"  throughput: {tokens_per_sec:.0f} tok/s ({samples_per_sec:.1f} samples/s)")
    print(f"  peak VRAM: {peak_vram} MB")
    print(f"  final loss: {train_loss:.4f}")
    print(f"  step time: {step_time_ms['mean']:.1f}ms mean, {step_time_ms['p95']:.1f}ms p95")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
