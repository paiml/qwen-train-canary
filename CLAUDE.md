# CLAUDE.md

## Project Overview

Competitive fine-tuning benchmarks for Qwen2.5-Coder-1.5B across five training runtimes.
Like qwen-coder-deploy compares inference runtimes, this project compares training paths head-to-head.
Five canary workloads: **apr** (Sovereign Stack/entrenar), unsloth (QLoRA), pytorch (full FT), cublas (parity gate), wgpu (burn).

Uses **forjar** for declarative deployment and deterministic canary datasets for reproducibility.

## Architecture

```
Yoga (PRIMARY — RTX 4060L, 8GB, sm_89)    gx10 (SECONDARY — GB10, 120GB, sm_121)
├── apr QLoRA (Sovereign Stack)            ├── apr QLoRA (batch=16)
├── unsloth QLoRA (Python baseline)        ├── unsloth QLoRA (batch=16)
├── Clock-locked 1900 MHz                  ├── pytorch full fine-tune
└── F-EXEC-02: full FT impossible on 8GB   └── cublas parity gate

Intel (SECONDARY — Radeon W5700X, 8GB)
├── wgpu/burn training canary
└── Vulkan backend
```

## Commands

```bash
# Yoga canaries (CUDA)
make canary-yoga           # Yoga canaries (apr + unsloth QLoRA)
make canary-apr            # APR fine-tune canary (Sovereign Stack, yoga)
make canary-apr-gx10       # APR fine-tune canary (gx10)
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

- `canaries/apr/train.py` — APR fine-tune canary (Sovereign Stack, wraps `apr finetune`)
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
