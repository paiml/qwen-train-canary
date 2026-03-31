# qwen-train-canary

## What This Is

Canary benchmarks for **training performance** across GPU backends. We take Qwen2.5-Coder-1.5B and run small, fast fine-tuning workloads to detect performance regressions and compare backends.

Three canary workloads, two backends:

| Canary | Backend | What It Measures |
|--------|---------|-----------------|
| **unsloth** (QLoRA) | CUDA | Optimized LoRA fine-tuning throughput |
| **pytorch** (full fine-tune) | CUDA | Baseline PyTorch training loop |
| **wgpu** (burn) | WGPU/Vulkan | Non-NVIDIA training viability |

## Why Canaries?

A canary is a short, reproducible training run (~100 steps) that produces consistent metrics. Run it before and after changes to detect:

- Training throughput regressions (samples/sec, tokens/sec)
- Memory regressions (peak VRAM, OOM thresholds)
- Loss convergence changes (final loss after N steps)
- Backend-specific issues (WGPU driver updates, CUDA version changes)

## Hardware Targets

```
Yoga (RTX 4060 Laptop, 8GB, sm_89)     gx10 (Grace Blackwell GB10, 120GB, sm_121)
├── unsloth QLoRA canary                ├── unsloth QLoRA canary (large batch)
├── pytorch baseline canary             ├── pytorch baseline canary
└── Clock-locked at 1900 MHz            └── 120 GB unified memory

Intel (192.168.50.100, Radeon W5700X)
├── wgpu/burn training canary
└── Vulkan backend, 8GB VRAM
```

## Quick Start

```bash
# Run all CUDA canaries on yoga
make canary-yoga

# Run individual canaries
make canary-unsloth      # QLoRA fine-tune (yoga, ~2 min)
make canary-pytorch      # PyTorch baseline (yoga, ~3 min)
make canary-wgpu         # WGPU/burn training (intel, ~5 min)

# Compare results
make report              # Generate comparison table
make score               # Grade pass/fail against baselines
```

## Key Metrics

Each canary produces:

| Metric | Unit | What It Tells You |
|--------|------|------------------|
| `throughput` | samples/sec | Training speed |
| `tokens_per_sec` | tok/s | Effective token processing rate |
| `peak_vram_mb` | MB | Memory high-water mark |
| `final_loss` | float | Convergence sanity check |
| `step_time_ms` | ms | Per-step latency (mean, p50, p95, p99) |
| `wall_time_sec` | sec | Total canary duration |

## Model & Dataset

- **Model**: Qwen2.5-Coder-1.5B-Instruct (same model as qwen-coder-deploy)
- **Dataset**: 500-sample code instruction subset (deterministic, checked in)
- **Steps**: 100 (canary), 1000 (extended)
- **Sequence length**: 512 tokens

## Results

Results are JSON files in `results/`, tracked in git. Format:

```json
{
  "canary": "unsloth",
  "backend": "cuda",
  "host": "yoga",
  "gpu": "RTX 4060 Laptop",
  "timestamp": "2026-03-31T12:00:00Z",
  "config": { "batch_size": 4, "seq_len": 512, "steps": 100, "lr": 2e-4 },
  "metrics": {
    "throughput_samples_sec": 12.5,
    "tokens_per_sec": 6400,
    "peak_vram_mb": 5200,
    "final_loss": 1.23,
    "step_time_ms": { "mean": 320, "p50": 315, "p95": 340, "p99": 355 },
    "wall_time_sec": 34.2
  }
}
```
