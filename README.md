# qwen-train-canary

<picture>
  <img src="docs/hero.svg" alt="Training throughput comparison across 5 runtimes" width="800">
</picture>

Competitive fine-tuning benchmarks for **Qwen2.5-Coder-1.5B** across five training runtimes — the training analog of [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy)'s inference runtime comparison.

## Measured Results

| Runtime | Engine | yoga (8GB) | gx10 (120GB) | intel (Vulkan) |
|---------|--------|-----------|-------------|---------------|
| **apr** (cuBLAS) | entrenar (Rust) | TBD | **2,101 tok/s** (4.5x over WGPU) | — |
| **apr** (WGPU) | entrenar (Rust) | BLOCKED (PMAT-498) | **470 tok/s** (loss 11.74) | — |
| **unsloth** | Python QLoRA | **6,697 tok/s** (loss 0.47) | **16,118 tok/s** (loss 0.14) | — |
| **pytorch** | Python full FT | OOM (F-EXEC-02) | **3,957 tok/s** (loss 0.009) | — |
| **cublas** | Python parity | OOM | **4,000 tok/s** | divergence: 0.000 |
| **wgpu** (synthetic) | burn (Rust) | — | — | **6,730 tok/s** (hidden=1536, not real model) |

All measurements: locked clocks, seed=42, deterministic dataset, steps=100. Yoga variance: 0.34% across 5 runs. **APR cuBLAS parity gap: 2.5x vs unsloth, 1.9x vs pytorch** (gx10). WGPU gap: 11.2x (was 151x on 2026-03-31).

## Why Canaries?

100-step training runs (~2 min) that produce machine-readable JSON. Run before and after changes to catch:

- Throughput regressions >10% (tok/s)
- Memory regressions >5% (peak VRAM)
- Convergence failures (loss threshold)
- GEMM backend divergence (cuBLAS parity gate)

## Hardware

```
Yoga (PRIMARY — RTX 4060L, 8GB, sm_89)    gx10 (GB10, 120GB, sm_121)
├── apr QLoRA (Sovereign Stack)            ├── apr QLoRA cuBLAS (2,101 tok/s)
├── unsloth QLoRA (6,697 tok/s)            ├── apr QLoRA WGPU (470 tok/s)
└── Clock-locked 1900 MHz                  ├── unsloth QLoRA (16,118 tok/s)
                                           ├── pytorch full FT (3,957 tok/s)
                                           └── cublas parity (0.000 divergence)
Intel (Radeon W5700X, 8GB, Vulkan)
└── wgpu/burn (6,730 tok/s @ hidden=1536, synthetic MLP)
```

## Quick Start

```bash
# Yoga (QLoRA canaries)
make canary-yoga           # apr + unsloth on yoga
make canary-apr            # APR/entrenar only
make canary-unsloth        # Unsloth QLoRA only

# gx10 (full fine-tune + parity)
make canary-gx10           # pytorch + cublas on GB10
make canary-compile-gx10   # torch.compile comparison

# Intel (WGPU/Vulkan)
make canary-wgpu           # burn/WGPU training

# Scoring, validation & testing
make test                  # 64 pytest tests (schema + scoring + 8 provable contracts)
make validate-schema       # JSON schema validation (F-MET-01)
make score                 # Pass/fail against baselines
make report                # Markdown comparison table
make profile-yoga          # apr roofline analysis
make nsys-yoga             # NVIDIA kernel timeline
```

## Key Findings

**F-EXEC-02 (falsified):** Full fine-tune of 1.5B is impossible on 8GB. Model weights (3.5GB) + gradients (3.5GB) = 7GB floor. QLoRA is the only viable path on consumer GPUs.

**F-RD-01 (falsified):** torch.compile regresses -11% at canary length. Compilation cost (~90s) dominates 200s runs. Not suitable for short benchmarks.

**F-HW-01 (confirmed):** Locked clocks give 0.34% throughput variance. VRAM and loss are perfectly deterministic.

**F-WL-03 (confirmed):** cuBLAS parity is perfect on Blackwell. Zero loss divergence, 1.004x throughput ratio.

**F-WL-06 (apr gap narrowing):** 65+ upstream fixes. **cuBLAS routing fix (PMAT-494): 2,101 tok/s** — 4.5x over WGPU 470. Gap vs unsloth: 2.5x (cuBLAS) / 11.2x (WGPU), down from 151x on 2026-03-31.

**Path B CONFIRMED (2026-04-06):** cuBLAS hybrid backend chosen. One routing fix = 4.5x throughput. Backward not yet confirmed on CUDA path (backward_steps=0, PMAT-512). Convergence is the remaining blocker.

**F-PROGRESS-01 (PARTIALLY RESOLVED):** cuBLAS routing delivered >2x improvement. 7 WGPU tiers still UNMEASURED but deprioritized. Provable-contract Grade A REQUIRED for all cuBLAS training code (PMAT-515).

**WGPU parity (synthetic only):** burn/Vulkan at 6,730 tok/s matches unsloth/CUDA at 6,697 tok/s on equivalent hidden dim — but this is a synthetic MLP, NOT real Qwen model training.

## Parity Mandate

Gaps are defects to fix, not findings to document. Every runtime must achieve throughput parity or be actively improved. See [spec](docs/specifications/training-canary-spec.md) for the full parity enforcement protocol.

## Model & Dataset

- **Model**: Qwen2.5-Coder-1.5B-Instruct (1.78B params)
- **Dataset**: 50 code instruction pairs (deterministic, `prompts/canary-dataset.yaml`)
- **Config**: 100 steps, batch=4, seq_len=512, lr=2e-4, seed=42 (unsloth@gx10 baseline uses batch=16)

## Specification

Full spec with falsification conditions: [docs/specifications/training-canary-spec.md](docs/specifications/training-canary-spec.md)
