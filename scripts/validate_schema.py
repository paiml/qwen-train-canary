#!/usr/bin/env python3
"""Validate canary result JSON files against metrics schema contract.

Implements canary-metrics-schema-v1.yaml from provable-contracts.
Falsification condition: F-MET-01.
"""

import glob
import json
import os
import sys

REQUIRED_TOP_LEVEL = ["canary", "backend", "host", "timestamp", "config", "metrics"]
VALID_CANARIES = {"unsloth", "pytorch", "pytorch-compile", "cublas", "wgpu", "apr"}
VALID_BACKENDS = {"cuda", "wgpu", "vulkan", "cpu", "metal", "auto"}
REQUIRED_CONFIG = ["batch_size", "seq_len", "steps", "lr", "seed"]
REQUIRED_GPU = ["device", "vram_total_mb", "cuda_version", "compute_capability"]
REQUIRED_STANDARD_METRICS = ["throughput_samples_sec", "tokens_per_sec", "final_loss"]
REQUIRED_PARITY = ["loss_divergence", "throughput_ratio", "numerically_equivalent"]


def validate_result(path: str) -> list[str]:
    """Validate a single result JSON. Returns list of error strings."""
    errors = []
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return [f"Invalid JSON: {e}"]

    fname = os.path.basename(path)

    # F-SCHEMA-001: Required top-level fields
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(f"[{fname}] Missing required field: {field}")

    canary = data.get("canary", "")
    if canary and canary not in VALID_CANARIES:
        errors.append(f"[{fname}] Unknown canary: {canary!r} (expected {VALID_CANARIES})")

    backend = data.get("backend", "")
    if backend and backend not in VALID_BACKENDS:
        errors.append(f"[{fname}] Unknown backend: {backend!r} (expected {VALID_BACKENDS})")

    # F-SCHEMA-005: Config required fields
    config = data.get("config", {})
    for field in REQUIRED_CONFIG:
        if field not in config:
            errors.append(f"[{fname}] Missing config.{field}")

    # F-SCHEMA-010: GPU info required for CUDA
    if backend == "cuda" and "gpu" not in data:
        errors.append(f"[{fname}] CUDA backend missing gpu block")
    if "gpu" in data:
        for field in REQUIRED_GPU:
            if field not in data["gpu"]:
                errors.append(f"[{fname}] Missing gpu.{field}")

    metrics = data.get("metrics", {})

    # F-SCHEMA-020: Standard metrics for non-cublas
    if canary != "cublas":
        for field in REQUIRED_STANDARD_METRICS:
            if field not in metrics:
                errors.append(f"[{fname}] Missing metrics.{field}")

    # F-SCHEMA-030: Parity metrics for cublas
    if canary == "cublas":
        for block in ["default", "cublas", "parity"]:
            if block not in metrics:
                errors.append(f"[{fname}] cublas canary missing metrics.{block}")
        parity = metrics.get("parity", {})
        for field in REQUIRED_PARITY:
            if field not in parity:
                errors.append(f"[{fname}] Missing metrics.parity.{field}")

    # F-DOMAIN-001: Throughput positive
    tok_s = metrics.get("tokens_per_sec", None)
    if tok_s is not None and tok_s <= 0:
        errors.append(f"[{fname}] tokens_per_sec must be > 0 (got {tok_s})")

    # F-DOMAIN-002: Loss finite and non-negative
    loss = metrics.get("final_loss", None)
    if loss is not None and (loss < 0 or loss >= 500):
        errors.append(f"[{fname}] final_loss must be in [0, 100) (got {loss})")

    # F-DOMAIN-003: VRAM within hardware limits
    vram = metrics.get("peak_vram_mb", None)
    gpu_total = data.get("gpu", {}).get("vram_total_mb", None)
    if vram is not None and gpu_total is not None and vram > gpu_total:
        errors.append(
            f"[{fname}] peak_vram_mb ({vram}) > gpu.vram_total_mb ({gpu_total})"
        )

    return errors


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    files = sorted(glob.glob(os.path.join(results_dir, "canary-*.json")))

    if not files:
        print(f"No canary-*.json files in {results_dir}/")
        sys.exit(1)

    all_errors = []
    for path in files:
        errors = validate_result(path)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"  PASS  {os.path.basename(path)}")

    if all_errors:
        print(f"\n{len(all_errors)} schema violation(s):")
        for e in all_errors:
            print(f"  FAIL  {e}")
        sys.exit(1)
    else:
        print(f"\nAll {len(files)} result files pass schema validation (F-MET-01)")


if __name__ == "__main__":
    main()
