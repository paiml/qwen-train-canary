# Optimization Roadmap

**Parent:** [Training Canary Spec](../training-canary-spec.md) Section 8

---

## Phase 0: Establish Baselines — COMPLETE

| PMAT | Item | Status |
|------|------|--------|
| PMAT-420 | Initial repo scaffold + 4 canary workloads | DONE |
| PMAT-421 | First yoga canary run (unsloth) | DONE (6697 tok/s) |
| PMAT-422 | First cuBLAS parity measurement (gx10) | DONE (0.000 divergence) |
| PMAT-423 | First WGPU/burn training measurement (intel) | DONE (6730 tok/s @ hidden=1536) |
| PMAT-424 | Establish baselines from 5 nightly runs | DONE (0.34% variance) |
| PMAT-425 | Batuta stack registration (47 crates) | DONE |

## Phase 1: Training Throughput Optimization

| PMAT | Item | Status |
|------|------|--------|
| PMAT-426 | Torch.compile() canary | DONE: -11% at 100 steps (F-RD-01 falsified) |
| PMAT-427 | Flash Attention 2 for training | Planned |
| PMAT-428 | Gradient accumulation canary (batch=1, accum=4) | Planned |
| PMAT-429 | DeepSpeed ZeRO Stage 2 canary | Planned |
| PMAT-430 | FSDP canary (multi-GPU, future) | Planned |

## Phase 2: WGPU Training Maturity

| PMAT | Item | Status |
|------|------|--------|
| PMAT-431 | burn-canary Rust binary (MVP) | DONE (6730 tok/s on Vulkan) |
| PMAT-432 | WGPU compute shader optimization | Measured: parity at hidden=1536 |
| PMAT-433 | WGPU vs CUDA numerical parity | Planned |
| PMAT-434 | Apple Metal backend canary (M-series) | Planned |

## Phase 3: Advanced Canaries

| PMAT | Item | Status |
|------|------|--------|
| PMAT-435 | Multi-model canary (3B, 7B) | Planned |
| PMAT-436 | Long-context canary (seq_len=2048) | Planned |
| PMAT-437 | Mixed-precision canary (fp8 training) | Planned |
| PMAT-438 | LoRA merge + inference parity gate | Planned |

## APR Parity: Upstream Fix Tracker

14 fixes landed in entrenar/trueno/aprender. Pipeline verified complete.

| # | Fix | Repo | Impact |
|---|-----|------|--------|
| 1 | VRAM-aware training init | entrenar | No OOM on 8GB |
| 2 | Buffer guards for minimal embeds | entrenar | No CUDA poisoning |
| 3 | Clean VRAM routing (skip bad paths) | entrenar | No context corruption |
| 4 | Optimized CPU lm_head (matmul_nt_compute) | entrenar | 31min→3min/epoch |
| 5-7 | GpuBuffer ctx + make_current + set_context | trueno | Thread-safe transfers |
| 8 | Fresh embed upload (trainer.upload) | entrenar | Data reaches GPU |
| 9 | Inference forward for training | entrenar | Correct loss (11.93) |
| 10 | CPU lm_head backward (matmul_compute) | entrenar | Gradient computation |
| 11 | CPU autograd routing on 8GB | entrenar | Correct backward path |
| 12 | Inference forward + save layer inputs | entrenar | GPU backward enabled |
| 13 | Gradient upload via trainer.upload | entrenar | Non-zero backward gradients |
| 14 | Grad buffer re-allocation at seq_len | entrenar | Size-compatible backward |

### Tickets

| Ticket | Issue | Status |
|--------|-------|--------|
| paiml/trueno#231 | cuLinkCreate on sm_89 | Open (workaround: legacy JIT) |
| paiml/trueno#232 | cuMemcpy context | Fixed upstream |
| paiml/aprender#563 | CUDA training forward | Partially fixed (14 fixes) |
| paiml/aprender#564 | CUDA-free compilation | Open |
| paiml/aprender#565 | WgpuInstructPipeline sig | Fixed |
| paiml/entrenar#316 | NF4 forward NaN | Worked around (inference forward) |

### Contracts

| Contract | Status |
|----------|--------|
| cuda-training-forward-v1.yaml | 5/5 falsified, 3 resolved |
| chunked-lm-head-v1.yaml | Design complete, superseded by CPU lm_head |

## Recommended Next Steps

### P0: APR convergence (loss not decreasing)

The pipeline is complete but loss stays at 11.93. Root cause: Q4K dequantized
weights produce near-uniform logits. Three fix paths:

1. **Use FP16 model for training** — `apr import` the HF model in FP16 instead of Q4K.
   Q4K is designed for inference speed, not training. FP16 weights produce differentiated
   logits that LoRA can learn from. This is how unsloth works (NF4 quant via bitsandbytes,
   not Q4K GGUF).

2. **Fix NF4 training forward NaN** (entrenar#316) — the pre-allocated scratch buffer
   path produces NaN after 28 layers. Fixing this would enable the full GPU pipeline
   (2000+ tok/s) instead of the inference-forward workaround (36 tok/s).

3. **Use NF4 via bitsandbytes-style quantization** — the model weights should be loaded
   in full precision then quantized to NF4 during training (like unsloth). Current path
   loads pre-quantized Q4K weights which lose precision at initialization.

### P1: Throughput optimization

Once convergence works:

1. **Fix NF4 forward NaN** → full GPU pipeline (36→2000+ tok/s)
2. **Chunked GPU lm_head** → eliminate CPU lm_head bottleneck
3. **Fix copy_from_host_at** (trueno#232) → eliminate fresh alloc per step
4. **cuBLAS tensor cores** for training GEMMs (currently PTX naive kernels)

### P2: Cross-platform parity

1. **wgpu/burn real model loading** — burn can't load HF safetensors/APR yet.
   Current 6730 tok/s is on synthetic MLP. Real Qwen training on Vulkan needs
   model loading support in burn or APR format reader.

2. **entrenar CUDA-free compilation** (aprender#564) — enables wgpu-only builds
   for AMD/Intel GPU training without CUDA toolkit.

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-RD-01 | torch.compile >10% regression | FALSIFIED: -11% at canary length |
| F-RD-02 | DeepSpeed ZeRO-2 OOMs on yoga | Planned |
