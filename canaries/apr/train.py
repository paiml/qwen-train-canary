#!/usr/bin/env python3
"""APR fine-tune canary — measures aprender/entrenar training throughput.

Wraps `apr finetune` CLI to benchmark the Sovereign Stack training path
against unsloth (Python/QLoRA) and pytorch (Python/full FT).

Binary: apr (from ~/src/aprender, requires `training` feature).
NOTE: `apr finetune` is behind #[cfg(feature = "training")] — the installed
binary MUST be built with this feature (it's in default features, but
trueno-gpu compile errors can silently drop it). Verify with `apr finetune --help`.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


def get_gpu_info() -> dict:
    """Collect GPU metadata via nvidia-smi.

    Handles partial data (e.g. gx10 GB10 returns "[N/A]" for memory.total)
    by parsing each field independently so a single failure doesn't drop
    all fields (F-MET-01 schema compliance).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
    except Exception:
        return {
            "device": "unknown",
            "vram_total_mb": 0,
            "cuda_version": "unknown",
            "compute_capability": "unknown",
        }

    def _safe_int(s: str) -> int:
        try:
            return int(s)
        except (ValueError, TypeError):
            return 0  # e.g. "[N/A]" on gx10 GB10

    return {
        "device": parts[0] if len(parts) > 0 else "unknown",
        "vram_total_mb": _safe_int(parts[1]) if len(parts) > 1 else 0,
        "cuda_version": parts[2] if len(parts) > 2 and parts[2] else "unknown",
        "compute_capability": parts[3] if len(parts) > 3 and parts[3] else "unknown",
    }


def _apr_canary_name() -> str:
    """Determine canary name from environment variables."""
    parts = ["apr"]
    if os.environ.get("NF4_FUSED_GEMM") == "1":
        parts.append("fused")
    if os.environ.get("NF4_TC_GEMM") == "1":
        parts.append("tc")
    if os.environ.get("FP16_GEMM") == "1":
        parts.append("fp16")
    if os.environ.get("CUDA_GRAPH") == "1":
        parts.append("graph")
    return "-".join(parts)


def prepare_dataset(yaml_path: str, output_path: str):
    """Convert canary YAML dataset to JSONL for apr finetune."""
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    with open(output_path, "w") as f:
        for sample in data["samples"]:
            record = {
                "instruction": sample["instruction"],
                "response": sample["response"],
            }
            f.write(json.dumps(record) + "\n")

    return len(data["samples"])


def main():
    parser = argparse.ArgumentParser(description="APR fine-tune canary")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="prompts/canary-dataset.yaml")
    parser.add_argument("--output", default="/tmp/canary-apr.json")
    parser.add_argument("--method", default="qlora", choices=["qlora", "lora", "full"])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gpu-backend", default="auto", choices=["auto", "cuda", "wgpu"])
    parser.add_argument("--model-path", default=None, help="Local path to APR/GGUF model file")
    parser.add_argument("--profile", action="store_true",
                        help="Enable apr step profiler (writes JSON to /tmp/canary-apr-profile.json)")
    args = parser.parse_args()

    # Check apr is available (with training feature)
    apr_bin = shutil.which("apr")
    if apr_bin is None:
        print("error: apr binary not found in PATH", file=sys.stderr)
        sys.exit(1)

    # Verify apr finetune exists (training feature must be compiled in)
    check = subprocess.run([apr_bin, "finetune", "--help"], capture_output=True, text=True, timeout=10)
    if check.returncode != 0 and "unrecognized" in (check.stderr or ""):
        print("error: apr finetune not available — binary missing training feature", file=sys.stderr)
        print("  Fix: cd ~/src/aprender && cargo build --release -p apr-cli --features apr-cli/training", file=sys.stderr)
        print("  Then: cp /mnt/nvme-raid0/targets/aprender/release/apr ~/.cargo/bin/apr", file=sys.stderr)
        sys.exit(1)

    # Convert dataset to JSONL
    jsonl_path = "/tmp/canary-apr-dataset.jsonl"
    num_samples = prepare_dataset(args.dataset, jsonl_path)

    # Detect VRAM
    try:
        vram_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        vram_gb = int(vram_out) / 1024
    except Exception:
        vram_gb = 8.0

    # Resolve model file: explicit path, local search, or import from HuggingFace
    model_path = args.model_path or args.model
    if not os.path.exists(model_path):
        # Try common local paths
        for candidate in [
            os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4k.apr"),
            os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr"),
            os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break
        else:
            # Import from HuggingFace
            print(f"Model not found locally, importing from {args.model}...")
            import_cmd = [apr_bin, "import", f"hf://{args.model}"]
            result = subprocess.run(import_cmd, capture_output=True, text=True, check=False, timeout=600)
            if result.returncode == 0:
                # Parse imported path from output
                for line in result.stdout.strip().split("\n"):
                    if line.strip().endswith((".apr", ".gguf")):
                        model_path = line.strip()
                        break

    print(f"Using model: {model_path}")

    # Build apr finetune command
    cmd = [
        apr_bin, "finetune",
        model_path,
        "--method", args.method,
        "--rank", str(args.rank),
        "--data", jsonl_path,
        "--learning-rate", str(args.lr),
        # Match unsloth: 100 steps at 50 samples / batch_size = ~8 epochs
        "--epochs", str(max(1, (args.steps * args.batch_size) // max(num_samples, 1))),
        "--vram", f"{vram_gb:.1f}",
        "--max-seq-len", str(args.seq_len),
        "--gpu-backend", args.gpu_backend,
        "--output", "/tmp/canary-apr-adapter",
        "--json",
    ]

    if args.method == "qlora":
        cmd.append("--quantize-nf4")

    if args.profile:
        cmd.append("--profile")

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()

    # APR throughput is ~194 tok/s (GPU pipeline). Allow 7200s (2 hours) with margin.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=7200)

    wall_time = time.perf_counter() - t0

    # Parse JSON output from apr (planning phase → stdout)
    planning = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    planning = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    # Parse training metrics from stderr (apr emits trace logs there)
    stderr = result.stderr or ""
    peak_vram = 0
    final_loss = 0

    # Extract VRAM from: [GPU-SHARE] VRAM reserved: N MB
    vram_match = re.search(r'\[GPU-SHARE\] VRAM reserved: (\d+) MB', stderr)
    if vram_match:
        peak_vram = int(vram_match.group(1))

    # Extract loss from stderr — matches multiple formats:
    #   CUDA NF4: "[CUDA] loss=X.XXXX (finite, proceeding with backward)"
    #   WGPU:     "avg_loss=X.XXXX"  or  "Epoch N complete: avg_loss=X.XXXX"
    #   Generic:  "loss=X.XX" or "loss: X.XX"
    loss_matches = re.findall(r'(?:avg_)?loss[=:]\s*([\d.]+)', stderr)
    if loss_matches:
        final_loss = float(loss_matches[-1])  # last loss value

    # Extract per-epoch loss trajectory (for loss_improved contract, F-CONV-03)
    epoch_matches = re.findall(r'Epoch\s+\d+\s+complete:\s+avg_loss=([\d.]+)', stderr)
    loss_trajectory = [float(l) for l in epoch_matches]

    # Count NaN-skipped backward passes (PMAT-462: inflates throughput)
    nan_skips = len(re.findall(r'NaN/Inf loss detected.*skipping backward', stderr))
    valid_backward_count = len(re.findall(r'loss[=:]\s*[\d.]+', stderr))  # non-NaN losses

    # Extract VRAM from planning if available
    if not peak_vram and planning.get("memory_breakdown"):
        peak_vram = int(planning["memory_breakdown"].get("total_bytes", 0) / (1024 * 1024))

    # Parse step profiler output if present (step-profiler-v1 contract)
    # PMAT-483: Prefer structured JSON output (print_json_report), fall back to text table
    profiler_data = {}
    json_profiler_match = re.search(r'(\{"_profiler":"step_profiler_v[12]".*\})', stderr)
    if json_profiler_match:
        try:
            profiler_data = json.loads(json_profiler_match.group(1))
        except json.JSONDecodeError:
            pass

    if not profiler_data:
        # Legacy: parse text table output from print_report()
        profiler_match = re.search(
            r'Step Profiler.*?TOTAL\s*│\s*([\d.]+)\s*│\s*100%\s*│\s*([\d.]+)',
            stderr, re.DOTALL
        )
        if profiler_match:
            profiler_data["total_ms"] = float(profiler_match.group(1))
            profiler_data["avg_step_ms"] = float(profiler_match.group(2))
            # Parse per-phase breakdown
            for phase_match in re.finditer(
                r'│\s*(\w+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)%\s*│\s*([\d.]+)\s*│',
                stderr
            ):
                phase_name = phase_match.group(1)
                if phase_name != "TOTAL":
                    profiler_data[phase_name] = {
                        "total_ms": float(phase_match.group(2)),
                        "pct": float(phase_match.group(3)),
                        "avg_ms": float(phase_match.group(4)),
                    }

    # Compute throughput from wall_time (apr doesn't emit tok/s yet — aprender#566)
    # APR uses --epochs, so planned steps = epochs * ceil(num_samples / batch_size)
    epochs = max(1, (args.steps * args.batch_size) // max(num_samples, 1))
    planned_steps = epochs * ((num_samples + args.batch_size - 1) // args.batch_size)

    # Use actual completed steps if profiler data available, else planned
    profiler_steps = 0
    if profiler_data and profiler_data.get("steps_profiled"):
        profiler_steps = profiler_data["steps_profiled"]

    # Count [PROFILE] step lines as fallback for completed step count
    profile_lines = re.findall(r'\[PROFILE\] step:', stderr)
    completed_steps = profiler_steps or len(profile_lines) or planned_steps

    # For crashed runs (returncode != 0), use actual completed steps
    actual_steps = completed_steps if result.returncode != 0 else planned_steps
    total_tokens = actual_steps * args.batch_size * args.seq_len
    tok_s = 0
    if wall_time > 0:
        tok_s = total_tokens / wall_time

    metrics = planning if planning else {}
    if result.returncode != 0:
        metrics["error"] = f"apr finetune exited {result.returncode}"
        metrics["stderr_tail"] = stderr[-500:]

    output = {
        "canary": _apr_canary_name(),
        "backend": "cuda" if args.gpu_backend == "auto" else args.gpu_backend,
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
            "method": args.method,
            "rank": args.rank,
            "quantization": "nf4" if args.method == "qlora" else "none",
            "runtime": "apr (aprender/entrenar)",
        },
        "metrics": {
            "throughput_samples_sec": round((actual_steps * args.batch_size) / wall_time, 2) if wall_time > 0 else 0,
            "tokens_per_sec": round(tok_s, 1),
            "peak_vram_mb": peak_vram,
            "final_loss": round(final_loss, 4) if final_loss else 0,
            "wall_time_sec": round(wall_time, 2),
            "nan_backward_skips": nan_skips,
            "valid_backward_steps": valid_backward_count,
            "loss_trajectory": loss_trajectory,  # per-epoch losses (F-CONV-03 loss_improved contract)
            # PMAT-462: If nan_skips > 0, throughput is inflated (NaN steps lack backward).
            # True training throughput is unknown until NaN cascade is fixed upstream.
            "_baseline_status": "PROVISIONAL" if nan_skips > 0 else "measured",
            "_metrics_quality": "estimated" if not loss_matches else "measured",
        },
        "apr_output": {
            "raw_output": result.stdout[-1000:] if result.stdout else "",
            "stderr_summary": stderr[-4000:] if stderr else "",
        },
        **({"profiler": profiler_data} if profiler_data else {}),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAPR canary complete in {wall_time:.1f}s")
    print(f"  throughput: {tok_s:.0f} tok/s")
    print(f"  peak VRAM: {peak_vram} MB")
    print(f"  final loss: {final_loss}")
    if nan_skips > 0:
        print(f"  WARNING: {nan_skips} NaN backward skips — throughput PROVISIONAL (PMAT-462)")
        print(f"  valid backward steps: {valid_backward_count}")
    if profiler_data:
        wc = profiler_data.get("wall_coverage", 0)
        bn = profiler_data.get("bottleneck", "unknown")
        gp = profiler_data.get("gemm_pct", 0)
        print(f"  profiler: wall_coverage={wc:.1%}, bottleneck={bn}, gemm_pct={gp:.1f}%")
        if wc < 0.90:
            print(f"  WARNING: wall coverage {wc:.1%} < 90% — missing instrumentation (F-TSP-001)")
        ops = profiler_data.get("ops", {})
        if ops:
            top_ops = sorted(ops.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  top ops: {', '.join(f'{k}={v:.1f}ms' for k, v in top_ops)}")
    print(f"  output: {args.output}")

    # PMAT-506: Apply scoring contracts immediately — catch silent regressions
    # (F-REGRESS-01: gx10 binary regressed loss 11.74→100 undetected).
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        from score import score_result, load_baselines
        baselines = load_baselines(
            os.path.join(os.path.dirname(__file__), "..", "..", "baselines.json")
        )
        canary_name = output["canary"]
        baseline = baselines.get(canary_name, baselines.get("apr", {}))
        score = score_result(output, baseline)
        print(f"\n=== Scoring Contracts (PMAT-506) ===")
        for check, info in score["checks"].items():
            status = "PASS" if info["pass"] else "FAIL"
            value = info.get("value", "?")
            threshold = info.get("threshold", info.get("baseline", "?"))
            print(f"  [{status}] {check}: {value} (threshold: {threshold})")
        if not score["pass"]:
            print(f"\nFAIL: canary failed {sum(1 for c in score['checks'].values() if not c['pass'])} contract(s)")
            sys.exit(1)
        print(f"\nPASS: all contracts satisfied")
    except ImportError:
        print("  (score.py not found — contracts not applied)")


if __name__ == "__main__":
    main()
