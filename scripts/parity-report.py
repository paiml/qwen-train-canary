#!/usr/bin/env python3
"""Parity profiling report — cross-runtime training performance comparison.

Reads canary JSON files and produces a side-by-side comparison showing
throughput, VRAM, loss, and per-operation gaps across runtimes.

Usage:
    python scripts/parity-report.py                          # Auto-discover latest results
    python scripts/parity-report.py results/canary-*.json    # Explicit files
    python scripts/parity-report.py --latest                 # Latest per-canary type

Contract: parity-profiling-system-v1.yaml (PMAT-487)
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def load_canary(path: str) -> dict:
    """Load a canary JSON result file."""
    with open(path) as f:
        return json.load(f)


def discover_latest_results(results_dir: str = "results") -> list[str]:
    """Auto-discover latest result per canary type (most recent file wins)."""
    files = sorted(glob.glob(os.path.join(results_dir, "canary-*.json")),
                   key=os.path.getmtime, reverse=True)
    seen = {}
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
            canary = data.get("canary", "unknown")
            host = data.get("host", "unknown")
            key = f"{canary}@{host}"
            if key not in seen:
                seen[key] = path
        except Exception:
            continue
    return list(seen.values())


def extract_profile(canary: dict) -> dict:
    """Extract parity-profile-v1 data from canary result."""
    profile = canary.get("profile", {})
    canary_name = canary.get("canary", "unknown")
    host = canary.get("host", "")
    runtime = profile.get("runtime", canary_name)

    # Add host suffix for disambiguation
    if host and host not in runtime:
        runtime = f"{runtime}@{host}"

    # Handle cublas nested metrics
    m = canary.get("metrics", {})
    tok_s = m.get("tokens_per_sec", 0)
    if canary_name == "cublas" and "default" in m:
        tok_s = m["default"].get("tokens_per_sec", tok_s)

    return {
        "runtime": runtime,
        "canary": canary_name,
        "host": host,
        "tok_s": tok_s,
        "step_ms": m.get("step_time_ms", {}).get("mean", 0),
        "peak_vram_mb": m.get("peak_vram_mb", 0),
        "final_loss": m.get("final_loss", 0),
        "kernel_launches": profile.get("hardware", {}).get("kernel_launches_per_step", 0),
        "cuda_ms": profile.get("hardware", {}).get("total_cuda_time_ms", 0),
        "ops": profile.get("ops", {}),
        "has_profile": bool(profile),
        "config_steps": canary.get("config", {}).get("steps", 0),
        "wall_time": m.get("wall_time_sec", 0),
        "nan_skips": m.get("nan_backward_skips", 0),
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


def format_regression_summary(profiles: list[dict], baselines_path: str = "baselines.json") -> str:
    """Generate regression summary comparing latest measurements to baselines."""
    lines = []
    baselines = {}
    if os.path.exists(baselines_path):
        with open(baselines_path) as f:
            baselines = json.load(f)

    lines.append("")
    lines.append("## Regression Summary")
    lines.append("")
    lines.append("| Runtime | tok/s | Baseline | Delta | Status |")
    lines.append("|---------|-------|----------|-------|--------|")

    for p in sorted(profiles, key=lambda x: -x["tok_s"]):
        rt = p["runtime"]
        canary = p.get("canary", rt)
        host = p.get("host", "")
        tok = p["tok_s"]
        # Host-specific baseline lookup: try "canary@host" first, fall back to "canary"
        bl_key = f"{canary}@{host}" if host else canary
        bl_data = baselines.get(bl_key, baselines.get(canary, {}))
        bl = bl_data.get("tokens_per_sec", 0)
        steps = p.get("config_steps", 0)
        nan = p.get("nan_skips", 0)
        notes = []
        if steps < 100:
            notes.append(f"{steps}st")
        if nan > 0:
            notes.append(f"{nan}NaN")
        note_str = f" ({', '.join(notes)})" if notes else ""
        if bl > 0:
            delta = (tok - bl) / bl
            status = "PASS" if delta >= -0.10 else "**FAIL**"
            lines.append(f"| {rt} | {tok:,.0f} | {bl:,.0f} | {delta:+.1%} | {status}{note_str} |")
        else:
            lines.append(f"| {rt} | {tok:,.0f} | — | — | NEW{note_str} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Training parity profiling report")
    parser.add_argument("files", nargs="*", help="Canary JSON result files (auto-discovers if omitted)")
    parser.add_argument("--latest", action="store_true", help="Auto-discover latest per-canary type")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--baselines", default="baselines.json", help="Baselines file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    args = parser.parse_args()

    # Auto-discover if no files provided or --latest
    if not args.files or args.latest:
        args.files = discover_latest_results(args.results_dir)
        print(f"Auto-discovered {len(args.files)} latest results", file=sys.stderr)

    profiles = []
    for path in args.files:
        try:
            canary = load_canary(path)
            profile = extract_profile(canary)
            profile["_file"] = os.path.basename(path)
            profiles.append(profile)
            print(f"  {profile['runtime']:20s} {profile['tok_s']:>8,.0f} tok/s  ({os.path.basename(path)})", file=sys.stderr)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}", file=sys.stderr)

    if not profiles:
        print("Error: no valid canary files loaded", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(profiles, indent=2))
    else:
        report = format_report(profiles)
        report += format_regression_summary(profiles, args.baselines)
        print(report)


if __name__ == "__main__":
    main()
