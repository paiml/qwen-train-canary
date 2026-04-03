#!/usr/bin/env python3
"""Score canary results against baselines — pass/fail regression detection."""

import argparse
import glob
import json
import os
import sys


# Default baselines (overridden by baselines.json if present)
# Keep in sync with baselines.json — see canary-score-gate-v1.yaml F-BASE-002
DEFAULT_BASELINES = {
    "apr": {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 20.0},
    "unsloth": {"tokens_per_sec": 6600, "peak_vram_mb": 3600, "final_loss": 2.0},
    "pytorch": {"tokens_per_sec": 4000, "peak_vram_mb": 51000, "final_loss": 2.0},
    "pytorch-compile": {"tokens_per_sec": 3500, "peak_vram_mb": 35000, "final_loss": 2.0},
    "cublas": {"tokens_per_sec": 4000, "peak_vram_mb": 51000, "final_loss": 2.0,
               "max_loss_divergence": 0.01, "min_throughput_ratio": 0.95},
    "wgpu": {"tokens_per_sec": 6600, "final_loss": 2.5},
}

THROUGHPUT_TOLERANCE = 0.10  # 10% regression threshold
VRAM_TOLERANCE = 0.05  # 5% memory regression threshold


def load_baselines(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_BASELINES


def score_result(result: dict, baseline: dict) -> dict:
    """Score a single result against its baseline."""
    m = result.get("metrics", {})
    canary = result.get("canary", "unknown")
    checks = {}

    # cuBLAS parity canary has nested metrics
    if canary == "cublas":
        return score_cublas_result(result, baseline)

    # Throughput regression check
    tok_s = m.get("tokens_per_sec", 0)
    base_tok = baseline.get("tokens_per_sec", 0)
    checks["throughput"] = {
        "value": tok_s,
        "baseline": base_tok,
        "pass": tok_s >= base_tok * (1 - THROUGHPUT_TOLERANCE),
    }

    # VRAM regression check — skip when baseline omits peak_vram_mb (apr, wgpu)
    if "peak_vram_mb" in baseline:
        vram = m.get("peak_vram_mb", 0)
        base_vram = baseline["peak_vram_mb"]
        checks["vram"] = {
            "value": vram,
            "baseline": base_vram,
            "pass": vram <= base_vram * (1 + VRAM_TOLERANCE),
        }

    # Loss convergence check
    loss = m.get("final_loss", float("inf"))
    base_loss = baseline.get("final_loss", 2.0)
    checks["loss"] = {
        "value": loss,
        "baseline": base_loss,
        "pass": loss <= base_loss,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {"canary": canary, "pass": all_pass, "checks": checks}


def score_cublas_result(result: dict, baseline: dict) -> dict:
    """Score cuBLAS parity canary — checks numerical equivalence and perf parity."""
    m = result.get("metrics", {})
    parity = m.get("parity", {})
    default_m = m.get("default", {})
    checks = {}

    # Numerical parity
    loss_div = parity.get("loss_divergence", float("inf"))
    max_div = baseline.get("max_loss_divergence", 0.01)
    checks["numerical_parity"] = {
        "value": loss_div,
        "baseline": max_div,
        "pass": loss_div <= max_div,
    }

    # Throughput parity (cuBLAS should be within 5% of default: 0.95-1.05)
    ratio = parity.get("throughput_ratio", 0)
    min_ratio = baseline.get("min_throughput_ratio", 0.95)
    max_ratio = baseline.get("max_throughput_ratio", 1.05)
    checks["perf_parity"] = {
        "value": ratio,
        "baseline": f"{min_ratio}-{max_ratio}",
        "pass": min_ratio <= ratio <= max_ratio,
    }

    # Default backend throughput regression
    tok_s = default_m.get("tokens_per_sec", 0)
    base_tok = baseline.get("tokens_per_sec", 0)
    checks["throughput"] = {
        "value": tok_s,
        "baseline": base_tok,
        "pass": tok_s >= base_tok * (1 - THROUGHPUT_TOLERANCE),
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {"canary": "cublas", "pass": all_pass, "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baselines", default="baselines.json")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baselines = load_baselines(args.baselines)
    scores = []

    for path in sorted(glob.glob(os.path.join(args.results_dir, "canary-*.json"))):
        with open(path) as f:
            result = json.load(f)
        canary = result.get("canary", "unknown")
        host = result.get("host", "unknown")
        # Host-specific baseline lookup: try "canary@host" first, fall back to "canary"
        baseline = baselines.get(f"{canary}@{host}", baselines.get(canary, {}))
        score = score_result(result, baseline)
        score["file"] = os.path.basename(path)
        scores.append(score)

    if args.format == "json":
        output = json.dumps(scores, indent=2)
    else:
        lines = ["Canary       | Pass | Checks"]
        lines.append("-------------|------|-------")
        for s in scores:
            c = s["checks"]
            check_strs = [f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in c.items()]
            lines.append(
                f"{s['canary']:12} | {'PASS' if s['pass'] else 'FAIL':4} | {', '.join(check_strs)}"
            )
        output = "\n".join(lines)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)

    # Exit non-zero if any canary failed
    if any(not s["pass"] for s in scores):
        sys.exit(1)


if __name__ == "__main__":
    main()
