# CLAUDE.md

## Project Overview

Training performance canary benchmarks for Qwen2.5-Coder-1.5B across CUDA and WGPU backends.
Four canary workloads: unsloth (QLoRA), pytorch (baseline full fine-tune), cublas (parity gate), wgpu (burn framework).

Uses **forjar** for declarative deployment and deterministic canary datasets for reproducibility.

## Architecture

```
Yoga (PRIMARY canary target)             gx10 (Grace Blackwell GB10)
├── unsloth QLoRA    (CUDA, sm_89)       ├── unsloth QLoRA    (CUDA, sm_121)
├── pytorch baseline (CUDA, sm_89)       ├── pytorch baseline (CUDA, sm_121)
├── RTX 4060 Laptop, 8GB VRAM           └── 120 GB unified memory
└── Clock-locked 1900 MHz

Intel (192.168.50.100, WGPU target)
├── wgpu/burn canary (Vulkan, W5700X)
└── Radeon Pro W5700X, 8GB VRAM
```

## Commands

```bash
# Yoga canaries (CUDA)
make canary-yoga           # All CUDA canaries (unsloth + pytorch + cublas)
make canary-unsloth        # Unsloth QLoRA only (~2 min)
make canary-pytorch        # PyTorch baseline only (~3 min)
make canary-cublas         # cuBLAS parity gate (~4 min, runs model twice)
make deploy-yoga           # Deploy canary scripts to yoga
make teardown-yoga         # Clean up

# Intel canaries (WGPU)
make canary-wgpu           # Burn/WGPU training canary (~5 min)
make deploy-wgpu           # Deploy to intel

# GB10 canaries
make deploy-gx10           # Deploy to gx10
make canary-gx10           # All canaries on GB10

# Reports & scoring
make report                # Generate comparison report
make score                 # Pass/fail against baselines
make score-json            # JSON scorecards to results/
```

## Canary Configuration

| Parameter | Canary (default) | Extended |
|-----------|-----------------|----------|
| Steps | 100 | 1000 |
| Batch size | 4 | 4 |
| Seq length | 512 | 512 |
| Learning rate | 2e-4 | 2e-4 |
| Dataset | 50 samples | 50 samples |
| Warmup | 10 steps | 50 steps |

## Key Files

- `canaries/unsloth/train.py` — Unsloth QLoRA canary script
- `canaries/pytorch/train.py` — PyTorch baseline canary script
- `canaries/cublas/train.py` — cuBLAS parity canary (default vs cuBLAS GEMM)
- `canaries/wgpu/train.py` — Burn/WGPU canary script
- `prompts/canary-dataset.yaml` — Deterministic training dataset (50 samples)
- `forjar-yoga.yaml` — Yoga deployment (CUDA canaries)
- `forjar-intel-wgpu.yaml` — Intel deployment (WGPU canary)
- `forjar-gx10.yaml` — GB10 deployment
- `results/` — JSON benchmark results (git-tracked)
- `scripts/nightly.sh` — Automated nightly canary pipeline

## Testing

Canary correctness = deterministic output given fixed seed + dataset.
A canary **passes** if:
1. Training completes without OOM or crash
2. Final loss < 2.0 (convergence sanity)
3. Throughput within 10% of baseline (no regression)
4. Peak VRAM within 5% of baseline (no memory regression)

## Model & Dataset

- **Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (HuggingFace)
- **GGUF**: Not used — training requires full-precision or LoRA-compatible weights
- **Dataset**: `prompts/canary-dataset.yaml` — 50 seed samples, deterministic
- **Unsloth**: QLoRA with 4-bit quantization (NF4), rank=16, alpha=32
- **PyTorch**: Full fine-tune, AdamW, cosine schedule
- **cuBLAS**: Same as PyTorch but runs twice (default backend, then cuBLAS-forced) to detect parity gaps
- **WGPU/Burn**: Full fine-tune via burn-canary Rust binary
