#!/usr/bin/env python3
"""Parity profiling report — cross-runtime training performance comparison.

Reads canary JSON files with parity-profile-v1 sections and produces a
side-by-side comparison showing WHERE time goes differently per runtime.

Usage:
    python scripts/parity-report.py results/canary-apr-*.json results/canary-pytorch-*.json results/canary-unsloth-*.json

Contract: parity-profiling-system-v1.yaml (PMAT-487)
"""

import argparse
import json
import sys
from pathlib import Path


def load_canary(path: str) -> dict:
    """Load a canary JSON result file."""
    with open(path) as f:
        return json.load(f)


def extract_profile(canary: dict) -> dict:
    """Extract parity-profile-v1 data from canary result."""
    profile = canary.get("profile", {})
    runtime = profile.get("runtime", canary.get("canary", "unknown"))

    # Normalize: ensure all fields exist
    return {
        "runtime": runtime,
        "tok_s": canary.get("metrics", {}).get("tokens_per_sec", 0),
        "step_ms": canary.get("metrics", {}).get("step_time_ms", {}).get("mean", 0),
        "peak_vram_mb": canary.get("metrics", {}).get("peak_vram_mb", 0),
        "final_loss": canary.get("metrics", {}).get("final_loss", 0),
        "kernel_launches": profile.get("hardware", {}).get("kernel_launches_per_step", 0),
        "cuda_ms": profile.get("hardware", {}).get("total_cuda_time_ms", 0),
        "ops": profile.get("ops", {}),
        "has_profile": bool(profile),
    }


def parity_delta(apr_val: float, baseline_val: float) -> str:
    """Compute parity delta as formatted string."""
    if baseline_val <= 0:
        return "N/A"
    delta = (apr_val - baseline_val) / baseline_val
    if abs(delta) < 0.10:
        return f"  ={delta:+.0%}"  # parity
    elif delta > 0:
        return f" +{delta:.0%} SLOWER"
    else:
        return f" {delta:.0%} faster"


def format_report(profiles: list[dict]) -> str:
    """Generate Markdown parity comparison report."""
    lines = []
    lines.append("# Training Parity Report")
    lines.append("")
    lines.append(f"Runtimes compared: {', '.join(p['runtime'] for p in profiles)}")
    lines.append("")

    # Throughput comparison table
    lines.append("## Throughput & Efficiency")
    lines.append("")
    header = "| Metric |"
    sep = "|--------|"
    for p in profiles:
        header += f" {p['runtime']} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    metrics = [
        ("Throughput (tok/s)", "tok_s", "{:.0f}"),
        ("Step time (ms)", "step_ms", "{:.1f}"),
        ("Peak VRAM (MB)", "peak_vram_mb", "{}"),
        ("Final loss", "final_loss", "{:.4f}"),
        ("Kernel launches/step", "kernel_launches", "{}"),
        ("CUDA time/step (ms)", "cuda_ms", "{:.1f}"),
    ]

    for label, key, fmt in metrics:
        row = f"| {label} |"
        for p in profiles:
            val = p.get(key, 0)
            row += f" {fmt.format(val)} |"
        lines.append(row)

    # Per-operation comparison
    all_ops = set()
    for p in profiles:
        all_ops.update(p.get("ops", {}).keys())

    if all_ops:
        lines.append("")
        lines.append("## Per-Operation Breakdown (ms/step)")
        lines.append("")
        header = "| Operation |"
        sep = "|-----------|"
        for p in profiles:
            header += f" {p['runtime']} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        for op in sorted(all_ops):
            row = f"| {op} |"
            for p in profiles:
                op_data = p.get("ops", {}).get(op, {})
                mean = op_data.get("mean", 0)
                pct = op_data.get("pct", 0)
                row += f" {mean:.1f} ({pct:.0f}%) |"
            lines.append(row)

    # Parity delta analysis (APR vs best baseline)
    apr = next((p for p in profiles if "apr" in p["runtime"]), None)
    baselines = [p for p in profiles if "apr" not in p["runtime"]]

    if apr and baselines:
        best = min(baselines, key=lambda p: p.get("step_ms", float("inf")) or float("inf"))
        lines.append("")
        lines.append(f"## Parity Delta (APR vs {best['runtime']})")
        lines.append("")

        if apr["tok_s"] > 0 and best["tok_s"] > 0:
            gap = best["tok_s"] / apr["tok_s"]
            lines.append(f"- **Throughput gap**: {gap:.1f}x ({best['runtime']} is {gap:.1f}x faster)")

        if apr["kernel_launches"] > 0 and best["kernel_launches"] > 0:
            launch_ratio = apr["kernel_launches"] / best["kernel_launches"]
            lines.append(f"- **Kernel launch ratio**: {launch_ratio:.1f}x (APR launches {launch_ratio:.1f}x more kernels)")

        # Per-op gaps
        gaps = []
        for op in sorted(all_ops):
            apr_ms = apr.get("ops", {}).get(op, {}).get("mean", 0)
            best_ms = best.get("ops", {}).get(op, {}).get("mean", 0)
            if apr_ms > 0 and best_ms > 0:
                ratio = apr_ms / best_ms
                gaps.append((op, ratio, apr_ms, best_ms))

        if gaps:
            gaps.sort(key=lambda x: -x[1])  # worst gap first
            lines.append("")
            lines.append("### Top Operation Gaps")
            lines.append("")
            for op, ratio, apr_ms, best_ms in gaps[:5]:
                lines.append(f"- **{op}**: APR {apr_ms:.1f}ms vs {best['runtime']} {best_ms:.1f}ms ({ratio:.1f}x slower)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Training parity profiling report")
    parser.add_argument("files", nargs="+", help="Canary JSON result files")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    args = parser.parse_args()

    profiles = []
    for path in args.files:
        try:
            canary = load_canary(path)
            profile = extract_profile(canary)
            profiles.append(profile)
            print(f"Loaded: {path} → {profile['runtime']} ({profile['tok_s']:.0f} tok/s)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}", file=sys.stderr)

    if not profiles:
        print("Error: no valid canary files loaded", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(profiles, indent=2))
    else:
        print(format_report(profiles))


if __name__ == "__main__":
    main()
