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

**15 fixes landed** in entrenar/trueno/aprender. Pipeline verified complete and IS LEARNING (loss 4.86→3.27, 2026-04-01).

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
| 15 | NF4 forward NaN fix (entrenar#316) | entrenar | Loss now decreasing: 4.86→3.27 |

### Tickets

| Ticket | Issue | Status |
|--------|-------|--------|
| paiml/trueno#231 | cuLinkCreate on sm_89 | Open (workaround: legacy JIT) |
| paiml/trueno#232 | cuMemcpy context | Fixed upstream |
| paiml/aprender#563 | CUDA training forward | Partially fixed (15 fixes) |
| paiml/aprender#564 | CUDA-free compilation | Open |
| paiml/aprender#565 | WgpuInstructPipeline sig | Fixed |
| paiml/entrenar#316 | NF4 forward NaN | **FIXED** (2026-04-01, fix #15) |

### Contracts

| Contract | Status |
|----------|--------|
| cuda-training-forward-v1.yaml | 5/5 falsified, 3 resolved |
| chunked-lm-head-v1.yaml | Design complete, superseded by CPU lm_head |

## Findings Summary (2026-04-01)

### Measured Parity (tok/s)

| Runtime | yoga (8GB) | gx10 (120GB) | vs Unsloth |
|---------|-----------|-------------|------------|
| **unsloth** QLoRA | **6,628** | **16,118** | baseline |
| **apr** QLoRA (Q4K) | **44** | TBD | **0.7%** (151x gap) |
| **pytorch** full FT | OOM | **4,017** | 24.9% (4x, expected) |
| **cublas** parity | OOM | **4,000** | 0.000 divergence |
| **wgpu** synthetic | — | — | 6,730 (Vulkan) |

### Root Cause: APR 151x Parity Gap

Five-whys analysis:

1. **APR is 151x slower than unsloth** — 44 vs 6,628 tok/s on yoga
2. **GPU at 0% utilization** — all compute is on CPU (493% CPU, 0% GPU)
3. **CPU lm_head fallback** — `[CUDA] Skipping GPU embeddings (1780MB > 1228MB free)`
4. **NaN in GPU forward** — `[CUDA] NaN in forward output — inference-style forward failed`
5. **Q4K dequantized weights produce near-uniform logits** — Q4K is designed for inference
   speed, not training. FP16 weights produce differentiated logits that LoRA can learn from.

The model file format is the root cause. Unsloth loads HF safetensors in FP16 and
quantizes to NF4 at runtime (bitsandbytes). APR loads a pre-quantized Q4K GGUF which
has already lost the precision needed for training gradients.

### Two Paths to Parity

**Path 1: Safetensors training (Python stack)** — ALREADY AT PARITY

The Python stack (unsloth, pytorch, cublas) uses HuggingFace safetensors models.
No parity gap exists within this stack:
- unsloth QLoRA: 6,628 tok/s (yoga) — the throughput target
- pytorch full FT: 4,017 tok/s (gx10) — expected 4x slower (full precision vs QLoRA)
- cublas parity: 0.000 loss divergence — GEMM backends identical

**Path 2: APR training (Sovereign Stack)** — 151x gap, 3 fix tiers

## Recommended Next Steps

### P0: Fix NF4 dequant NaN in trueno — the GPU forward blocker

**Contract: apr-training-parity-v1.yaml — hypothesis-driven, falsify-first**

~~**H-PARITY-001 (FALSIFIED 2026-04-02):** Switch from Q4K to FP16 model.~~
Tested: FP16 model with `--quantize-nf4` shows **identical** behavior — 0% GPU,
146% CPU, `[CUDA] NaN in forward output`. The NaN is NOT caused by the Q4K input
format. Both Q4K and FP16+NF4 produce the same NaN through 28 transformer layers.
This means the bug is in **trueno's NF4 runtime quantization kernel**, not the model file.

**H-PARITY-002 (TRACED 2026-04-02):** Layer tracing found the precise root cause.

**Root cause:** trueno's NF4 dequant zeros out **V (value) projection weights**.

```
Diagnostic chain:
  apr profile --granular → inference works (151 tok/s, Q4K, grade D)
  apr finetune --quantize-nf4 → NaN in training forward, 0% GPU
  apr finetune stderr trace → 11 of 196 tensors dequant to ALL ZEROS
  All 11 are shape 256x1536 = V projection (GQA, 2 KV heads × 128 dim)
  K projection (same shape) dequantizes correctly
  Zero V weights → softmax(0/0) = NaN → propagates 28 layers
```

**Filed:** paiml/trueno#233 — NF4 dequant zeros V projection weights

**Confirmation test (next):**
```bash
# Bypass NF4 entirely — use FP16 LoRA (no quantization)
apr finetune qwen2.5-coder-1.5b-instruct-fp16.apr --method lora --rank 16
# If GPU works: NF4 V-proj dequant confirmed as sole root cause
# If NaN persists: forward pass has deeper issue (H-PARITY-003)
```

**Why this matters:** Every downstream optimization (chunked lm_head, cuBLAS tensor
cores, fused kernels) is BLOCKED until GPU forward works. Fixing 11 V-projection
tensors in trueno's NF4 dequant is the single gate that unlocks 44 → 2000+ tok/s.

### P1: APR GPU kernel optimization (after P0 unblocks GPU path)

Once FP16 model enables the GPU forward path:

1. **Chunked GPU lm_head** → eliminate CPU lm_head bottleneck entirely
   (if FP16 still falls back to CPU for embeddings)
2. **Fix copy_from_host_at** (trueno#232) → eliminate fresh alloc per step
3. **cuBLAS tensor cores** for training GEMMs (currently PTX naive kernels)
4. **Emit structured training metrics** (aprender#566) → proper loss/VRAM tracking

Expected impact: 2000 → 4000+ tok/s (parity with pytorch full FT baseline).

### P2: APR throughput parity with unsloth

To match unsloth's 6,628 tok/s, APR needs:
- Fused NF4 dequant + matmul kernels (like unsloth's custom Triton kernels)
- Gradient checkpointing integration (unsloth saves 60% memory)
- 8-bit optimizer (AdamW 8-bit via trueno, matching bitsandbytes)

This is the "beat unsloth" tier — requires trueno GEMM parity with Triton/cuBLAS.

### P3: Cross-platform parity

1. **wgpu/burn real model loading** — burn can't load HF safetensors/APR yet.
   Current 6,730 tok/s is on synthetic MLP. Real Qwen training on Vulkan needs
   model loading support in burn or APR format reader.

2. **entrenar CUDA-free compilation** (aprender#564) — enables wgpu-only builds
   for AMD/Intel GPU training without CUDA toolkit.

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-RD-01 | torch.compile >10% regression | FALSIFIED: -11% at canary length |
| F-RD-02 | DeepSpeed ZeRO-2 OOMs on yoga | Planned |
