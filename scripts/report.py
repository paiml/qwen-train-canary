#!/usr/bin/env python3
"""Generate performance comparison report from canary results."""

import argparse
import glob
import json
import os
from datetime import datetime


def load_results(results_dir: str) -> list[dict]:
    """Load all JSON result files."""
    results = []
    for path in sorted(glob.glob(os.path.join(results_dir, "canary-*.json"))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def format_table(results: list[dict]) -> str:
    """Format results as markdown table."""
    lines = [
        "| Canary | Backend | Host | tok/s | VRAM (MB) | Loss | Wall (s) |",
        "|--------|---------|------|-------|-----------|------|----------|",
    ]
    for r in results:
        m = r.get("metrics", {})
        canary = r.get("canary", "?")

        # cuBLAS parity canary has nested metrics — extract default run
        if canary == "cublas" and "default" in m:
            dm = m.get("default", {})
            parity = m.get("parity", {})
            lines.append(
                f"| cublas (default) "
                f"| {r.get('backend', '?')} "
                f"| {r.get('host', '?')} "
                f"| {dm.get('tokens_per_sec', '?')} "
                f"| {dm.get('peak_vram_mb', '?')} "
                f"| {dm.get('final_loss', '?')} "
                f"| {dm.get('wall_time_sec', '?')} |"
            )
            cm = m.get("cublas", {})
            lines.append(
                f"| cublas (forced) "
                f"| {r.get('backend', '?')} "
                f"| {r.get('host', '?')} "
                f"| {cm.get('tokens_per_sec', '?')} "
                f"| {cm.get('peak_vram_mb', '?')} "
                f"| {cm.get('final_loss', '?')} "
                f"| {cm.get('wall_time_sec', '?')} |"
            )
        else:
            lines.append(
                f"| {canary} "
                f"| {r.get('backend', '?')} "
                f"| {r.get('host', '?')} "
                f"| {m.get('tokens_per_sec', '?')} "
                f"| {m.get('peak_vram_mb', '?')} "
                f"| {m.get('final_loss', '?')} "
                f"| {m.get('wall_time_sec', '?')} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="performance.md")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print("No results found.")
        return

    report = f"# Training Canary Performance\n\nGenerated: {datetime.now().isoformat()}\n\n"
    report += format_table(results) + "\n"

    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
