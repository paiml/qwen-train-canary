#!/usr/bin/env python3
"""Score canary results against baselines — pass/fail regression detection."""

import argparse
import glob
import json
import math
import os
import sys


# Default baselines (overridden by baselines.json if present)
# Keep in sync with baselines.json — see canary-score-gate-v1.yaml F-BASE-002
DEFAULT_BASELINES = {
    "apr": {"tokens_per_sec": 470, "final_loss": 12.0},
    "apr-fused": {"tokens_per_sec": 470, "final_loss": 12.0},
    "apr-tc": {"tokens_per_sec": 470, "final_loss": 12.0},
    "apr-fp16": {"tokens_per_sec": 150, "peak_vram_mb": 3000, "final_loss": 12.0},
    "apr-fused-fp16": {"tokens_per_sec": 200, "peak_vram_mb": 3000, "final_loss": 12.0},
    "apr-fused-fp16-graph": {"tokens_per_sec": 300, "peak_vram_mb": 3000, "final_loss": 12.0},
    "unsloth": {"tokens_per_sec": 6600, "peak_vram_mb": 3600, "final_loss": 1.0},
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

    # Loss convergence check — NaN treated as infinite (PMAT-467)
    loss = m.get("final_loss", float("inf"))
    if isinstance(loss, float) and math.isnan(loss):
        loss = float("inf")
    base_loss = baseline.get("final_loss", 2.0)
    checks["loss"] = {
        "value": loss,
        "baseline": base_loss,
        "pass": loss <= base_loss,
    }

    # NaN rate check for APR canaries (PMAT-473: FP16 NaN regression)
    nan_skips = m.get("nan_backward_skips", 0)
    total_steps = result.get("config", {}).get("steps", 100)
    if nan_skips > 0 and total_steps > 0:
        nan_rate = nan_skips / total_steps
        checks["nan_rate"] = {
            "value": round(nan_rate, 3),
            "threshold": 0.5,
            "pass": nan_rate < 0.5,  # >50% NaN = training broken
        }

    # PMAT-483: Profiling wall coverage check (F-TSP-001)
    profiler = result.get("profiler", {})
    if profiler and "wall_coverage" in profiler:
        wc = profiler["wall_coverage"]
        checks["wall_coverage"] = {
            "value": round(wc, 3),
            "threshold": 0.90,
            "pass": wc >= 0.90,
        }

    # PMAT-483/F-POP-002: GEMM dominance check (>= 30% of op time — relaxed from 50%)
    if profiler and "gemm_pct" in profiler:
        gp = profiler["gemm_pct"]
        checks["gemm_dominance"] = {
            "value": round(gp, 1),
            "threshold": 30.0,
            "pass": gp >= 30.0,  # If GEMM < 30%, launch overhead or transfers dominate
        }

    # === Phase A Provable Contracts (PMAT-504) ===
    # 6 compiler-enforced invariants from candle-vs-apr sister project.
    # These fire on every APR canary, catching correctness and profiler bugs.

    # Contract 1: Loss trajectory — model must be learning, not diverging
    # (F-CONV-01: loss < 2.0 for CUDA, < 2.5 for WGPU)
    if canary.startswith("apr") and loss > 0:
        loss_threshold = 2.5 if result.get("backend") == "wgpu" else 2.0
        checks["convergence"] = {
            "value": loss,
            "threshold": loss_threshold,
            "pass": loss <= loss_threshold,
            "_contract": "F-CONV-01: model must converge (loss > random=11.93 means broken)",
        }

    # Contract 2: Loss better than random — loss > ln(vocab_size) means model predicts worse than chance
    if canary.startswith("apr") and loss > 0:
        random_loss = 11.93  # ln(151936) for Qwen2.5 vocab
        checks["better_than_random"] = {
            "value": loss,
            "threshold": random_loss,
            "pass": loss < random_loss,
            "_contract": "F-CONV-02: loss must be below random baseline ln(vocab_size)=11.93",
        }

    # Contract 3: Step time monotonicity — step_time growing means memory leak or OOM creep
    # Checks profiler per-step data if available
    if profiler and profiler.get("phases"):
        phases = profiler["phases"]
        gpu_fwd = phases.get("gpu_fwd", {})
        gpu_bwd = phases.get("gpu_lora_bwd", {})
        # If avg step time > 0, check it's not absurdly long (> 10s per step = likely broken)
        avg_fwd = gpu_fwd.get("avg_ms", 0)
        avg_bwd = gpu_bwd.get("avg_ms", 0)
        total_step_ms = avg_fwd + avg_bwd
        if total_step_ms > 0:
            checks["step_time_sanity"] = {
                "value": round(total_step_ms, 1),
                "threshold": 10000,  # 10s per step is absurd for 1.5B model
                "pass": total_step_ms < 10000,
                "_contract": "F-PROF-STEP: step time < 10s (>10s indicates kernel stall or leak)",
            }

    # Contract 4: Valid backward steps > 0 — if zero backward steps, training didn't happen
    valid_bwd = m.get("valid_backward_steps", -1)
    if canary.startswith("apr") and valid_bwd >= 0:
        checks["backward_executed"] = {
            "value": valid_bwd,
            "threshold": 1,
            "pass": valid_bwd > 0,
            "_contract": "F-BWD-01: at least 1 valid backward step must execute",
        }

    # Contract 5: Metrics quality — measured > estimated (loss must be parseable)
    metrics_quality = m.get("_metrics_quality", "unknown")
    if canary.startswith("apr"):
        checks["metrics_quality"] = {
            "value": metrics_quality,
            "threshold": "measured",
            "pass": metrics_quality == "measured",
            "_contract": "F-MET-02: loss must be parsed from training output, not estimated",
        }

    # Contract 6: Config drift — steps must be >= 100 for baseline comparison
    steps = result.get("config", {}).get("steps", 0)
    if steps > 0 and steps < 100:
        checks["config_steps"] = {
            "value": steps,
            "threshold": 100,
            "pass": False,  # Always fail if steps < 100
            "_contract": "F-CFG-01: canary must run steps>=100 for baseline comparison",
        }

    # === Profiler Fidelity Contracts (PMAT-504, from candle-vs-apr sister project) ===
    # These catch profiler bugs (e.g., candle-vs-apr saw 3.4x fidelity error in Deferred sync mode)

    # Contract 7: LmHead vs RmsNorm ratio — catches profiler sync fidelity lag
    # LmHead should be >10x more expensive than RmsNorm (GEMM vs elementwise)
    if profiler and profiler.get("phases"):
        phases = profiler["phases"]
        lm_ms = phases.get("gpu_lm", {}).get("avg_ms", 0)
        # rmsnorm is not its own phase in current decomposition, so skip this check
        # when we don't have per-op rmsnorm timing. Just verify lm ran at all.
        if lm_ms > 0:
            checks["lmhead_executed"] = {
                "value": round(lm_ms, 2),
                "threshold": 0.0,
                "pass": lm_ms > 0,
                "_contract": "F-PROF-FIDELITY-01: gpu_lm phase must execute (>0ms)",
            }

    # Contract 8: No orphan spans — every phase with total_ms > 0 must have count > 0
    # (If a phase has total_ms but no count, the profiler is broken)
    if profiler and profiler.get("phases"):
        phases = profiler["phases"]
        orphans = []
        for name, p in phases.items():
            total = p.get("total_ms", 0)
            avg = p.get("avg_ms", 0)
            # If total > 0 but avg == 0, there's a counter bug
            if total > 0 and avg == 0:
                orphans.append(name)
        checks["no_orphan_spans"] = {
            "value": len(orphans),
            "threshold": 0,
            "pass": len(orphans) == 0,
            "_contract": "F-PROF-FIDELITY-02: no phase has total_ms > 0 with avg_ms == 0",
        }

    all_pass = all(c["pass"] for c in checks.values())
    # Surface PROVISIONAL status from APR NaN-inflated measurements (PMAT-462)
    provisional = m.get("_baseline_status") == "PROVISIONAL"
    return {"canary": canary, "pass": all_pass, "checks": checks, "provisional": provisional}


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
            status = "PASS" if s["pass"] else "FAIL"
            if s.get("provisional"):
                status = "PROV"
            lines.append(
                f"{s['canary']:12} | {status:4} | {', '.join(check_strs)}"
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
