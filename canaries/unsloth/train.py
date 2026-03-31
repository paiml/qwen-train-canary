#!/usr/bin/env python3
"""Unsloth QLoRA canary — fast fine-tuning benchmark for Qwen2.5-Coder-1.5B."""

import argparse
import json
import time
import os
import socket
from datetime import datetime, timezone

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import yaml


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
        "vram_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024 * 1024),
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
    )

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Run training
    t0 = time.perf_counter()
    result = trainer.train()
    wall_time = time.perf_counter() - t0

    # Collect metrics
    peak_vram = torch.cuda.max_memory_allocated() // (1024 * 1024)
    train_loss = result.training_loss
    samples_per_sec = (args.batch_size * args.steps) / wall_time
    tokens_per_sec = (args.batch_size * args.seq_len * args.steps) / wall_time

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
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "quantization": "nf4",
        },
        "metrics": {
            "throughput_samples_sec": round(samples_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mb": peak_vram,
            "final_loss": round(train_loss, 4),
            "wall_time_sec": round(wall_time, 2),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Canary complete: {args.steps} steps in {wall_time:.1f}s")
    print(f"  throughput: {tokens_per_sec:.0f} tok/s")
    print(f"  peak VRAM: {peak_vram} MB")
    print(f"  final loss: {train_loss:.4f}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
