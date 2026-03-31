# Optimization Roadmap

**Parent:** [Training Canary Spec](../training-canary-spec.md) Section 8

---

## Phase 0: Establish Baselines (Current -- Yoga Primary)

| PMAT | Item | Status |
|------|------|--------|
| PMAT-420 | Initial repo scaffold + 4 canary workloads | DONE |
| PMAT-421 | First yoga canary run (unsloth + pytorch) | Planned |
| PMAT-422 | First cuBLAS parity measurement (yoga) | Planned |
| PMAT-423 | First WGPU/burn training measurement (intel) | Planned |
| PMAT-424 | Establish yoga baselines from 5 nightly runs | Planned |
| PMAT-425 | Batuta stack registration (47 crates) | DONE |

> **Gate:** Phase 1 blocked until PMAT-424 (yoga baselines) completes successfully.

## Phase 1: Training Throughput Optimization

| PMAT | Item | Expected Impact |
|------|------|----------------|
| PMAT-426 | Torch.compile() canary (PyTorch 2.x graph mode) | +20-40% throughput |
| PMAT-427 | Flash Attention 2 for training (if not default) | +15-25% step time |
| PMAT-428 | Gradient accumulation canary (batch=1, accum=4) | Memory vs throughput tradeoff |
| PMAT-429 | DeepSpeed ZeRO Stage 2 canary | Enable full FT on 8 GB |
| PMAT-430 | FSDP canary (multi-GPU, future) | Distributed training baseline |

## Phase 2: WGPU Training Maturity

| PMAT | Item | Expected Impact |
|------|------|----------------|
| PMAT-431 | burn-canary Rust binary (MVP) | Enable WGPU canary |
| PMAT-432 | WGPU compute shader optimization | 2-5x throughput |
| PMAT-433 | WGPU vs CUDA numerical parity | Cross-backend convergence |
| PMAT-434 | Apple Metal backend canary (M-series) | macOS training feasibility |

## Phase 3: Advanced Canaries

| PMAT | Item | Expected Impact |
|------|------|----------------|
| PMAT-435 | Multi-model canary (3B, 7B) | Scale sensitivity |
| PMAT-436 | Long-context canary (seq_len=2048) | Attention scaling |
| PMAT-437 | Mixed-precision canary (fp8 training) | Next-gen precision |
| PMAT-438 | LoRA merge + inference parity gate | End-to-end validation |

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-RD-01 | If torch.compile() causes >10% throughput regression | Graph mode overhead exceeds gains on this model size |
| F-RD-02 | If DeepSpeed ZeRO-2 still OOMs on yoga at batch=4 | ZeRO not sufficient -- need offload or smaller batch |
