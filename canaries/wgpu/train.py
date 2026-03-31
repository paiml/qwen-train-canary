#!/usr/bin/env python3
"""WGPU/Burn canary — training benchmark via burn-rs subprocess.

This canary measures training viability on non-NVIDIA hardware (AMD Radeon, Intel Arc)
using WGPU/Vulkan backends via the burn framework.

The burn-canary Rust binary performs the actual training loop and emits JSON metrics.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


def run_burn_canary(args) -> dict:
    """Run burn-based training canary via subprocess."""
    cmd = [
        "burn-canary",
        "--model-dir", args.model,
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--backend", args.backend,
        "--dataset", args.dataset,
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)

    wall_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"burn-canary failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return {
            "error": f"burn-canary exited {result.returncode}",
            "stderr": result.stderr[-500:],
            "wall_time_sec": round(wall_time, 2),
        }

    # Parse JSON output from burn-canary
    try:
        metrics = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Try to find JSON in output
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                metrics = json.loads(line)
                break
        else:
            metrics = {"raw_output": result.stdout[-500:]}

    return metrics


def main():
    parser = argparse.ArgumentParser(description="WGPU/Burn training canary")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="prompts/canary-dataset.yaml")
    parser.add_argument("--output", default="/tmp/canary-wgpu.json")
    parser.add_argument("--backend", default="wgpu", choices=["wgpu", "vulkan"])
    args = parser.parse_args()

    metrics = run_burn_canary(args)

    output = {
        "canary": "wgpu",
        "backend": args.backend,
        "host": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
        },
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"WGPU canary complete. Output: {args.output}")


if __name__ == "__main__":
    main()
