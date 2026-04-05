# qwen-train-canary

<picture>
  <img src="docs/hero.svg" alt="Training throughput comparison across 5 runtimes" width="800">
</picture>

Competitive fine-tuning benchmarks for **Qwen2.5-Coder-1.5B** across five training runtimes — the training analog of [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy)'s inference runtime comparison.

## Measured Results

| Runtime | Engine | yoga (8GB) | gx10 (120GB) | intel (Vulkan) |
|---------|--------|-----------|-------------|---------------|
| **apr** | entrenar (Rust) | BLOCKED (WGPU crash, PMAT-498) | **470 tok/s** (WGPU async, loss 11.74 F-WL-07) | — |
| **unsloth** | Python QLoRA | **6,697 tok/s** (loss 0.47) | **16,118 tok/s** (loss 0.14) | — |
| **pytorch** | Python full FT | OOM (F-EXEC-02) | **4,017 tok/s** | — |
| **cublas** | Python parity | OOM | **4,000 tok/s** | divergence: 0.000 |
| **wgpu** (synthetic) | burn (Rust) | — | — | **6,730 tok/s** (hidden=1536, not real model) |

All measurements: locked clocks, seed=42, deterministic dataset, steps=100. Yoga variance: 0.34% across 5 runs. APR parity gap: **11.2x** vs unsloth on same gx10 hardware (was 151x on 2026-03-31).

## Why Canaries?

100-step training runs (~2 min) that produce machine-readable JSON. Run before and after changes to catch:

- Throughput regressions >10% (tok/s)
- Memory regressions >5% (peak VRAM)
- Convergence failures (loss threshold)
- GEMM backend divergence (cuBLAS parity gate)

## Hardware

```
Yoga (PRIMARY — RTX 4060L, 8GB, sm_89)    gx10 (GB10, 120GB, sm_121)
├── apr QLoRA (Sovereign Stack)            ├── apr QLoRA WGPU (470 tok/s, async)
├── unsloth QLoRA (6,697 tok/s)            ├── unsloth QLoRA (16,118 tok/s)
└── Clock-locked 1900 MHz                  ├── pytorch full FT (4,017 tok/s)
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

**F-WL-06 (apr gap narrowing):** 65+ upstream fixes across trueno/aprender/entrenar. APR WGPU async pipeline reaches **470 tok/s** on gx10 (11.2x gap vs unsloth, down from 151x on 2026-03-31). Profiler: 100% GPU compute, zero sync, `gpu_lora_bwd` 55.7% dominant.

**F-WL-07 (apr oscillation):** APR loss trajectory oscillates 18.9→9.15→12.3→16.3→15.5→12.0→10.8→11.7 across 8 epochs. Model IS learning (epoch 2 reached 9.15 < random 11.93) but oscillates. Likely cause: LR 2e-4 too high, needs cosine decay + warmup.

**F-PROGRESS-01 (TRIGGERED):** 6 optimization tiers SHIPPED, 0 MEASURED. Only measured delta in 5 days is +12% from async pipeline. Three-phase execution plan in effect (Phase A: contracts, Phase B: cuBLAS hybrid, Phase C: A/B matrix). Deadline: 2026-04-12.

**WGPU parity (synthetic only):** burn/Vulkan at 6,730 tok/s matches unsloth/CUDA at 6,697 tok/s on equivalent hidden dim — but this is a synthetic MLP, NOT real Qwen model training.

## Parity Mandate

Gaps are defects to fix, not findings to document. Every runtime must achieve throughput parity or be actively improved. See [spec](docs/specifications/training-canary-spec.md) for the full parity enforcement protocol.

## Model & Dataset

- **Model**: Qwen2.5-Coder-1.5B-Instruct (1.78B params)
- **Dataset**: 50 code instruction pairs (deterministic, `prompts/canary-dataset.yaml`)
- **Config**: 100 steps, batch=4, seq_len=512, lr=2e-4, seed=42 (unsloth@gx10 baseline uses batch=16)

## Specification

Full spec with falsification conditions: [docs/specifications/training-canary-spec.md](docs/specifications/training-canary-spec.md)
