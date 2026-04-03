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
| PMAT-428 | Gradient accumulation canary (batch=1, accum=4) | DONE (PMAT-459): --gradient-accumulation-steps N in pytorch canary |
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

**21 fixes landed** in entrenar/trueno/aprender. Pipeline verified complete and IS LEARNING (loss 16.80→converging, 43 tok/s canary confirmed 2026-04-02).

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
| 16 | Scratch zeroing per-step (cuMemsetD32Async) | entrenar | Cross-step NaN eliminated |
| 17 | gemm_forward_bt (BT GEMM for lm_head) | entrenar | GPU-resident lm_head forward |
| 18 | Dual-backend NF4 (cuBLAS/fused PTX) | entrenar | Backend comparison enabled |
| 19 | Logits trace removal (296MB D2H blocker) | entrenar | Unblocked GPU pipeline |
| 20 | TF32 tensor core cuBLAS handle | entrenar | Tensor cores tested |
| 21 | Zero training state backward buffers (PMAT-453) | entrenar | Multi-epoch NaN cascade fixed |

### Tickets

| Ticket | Issue | Status |
|--------|-------|--------|
| paiml/trueno#231 | cuLinkCreate on sm_89 | Open (workaround: legacy JIT) |
| paiml/trueno#232 | cuMemcpy context | Fixed upstream |
| paiml/aprender#563 | CUDA training forward | Partially fixed (15 fixes) |
| paiml/aprender#564 | CUDA-free compilation | Open |
| paiml/aprender#565 | WgpuInstructPipeline sig | Fixed |
| paiml/entrenar#316 | NF4 forward NaN | **FIXED** (2026-04-01, fix #15) |
| paiml/entrenar#319 | Multi-epoch NaN cascade (unzeroed training state) | **FIXED** (2026-04-02, fix #21) |

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
| **apr** QLoRA (NF4) | **194** (canary, 2026-04-02) | TBD | **2.9%** (34x gap) |
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

**H-PARITY-002 (RUNNING 2026-04-02):** FP16 LoRA (no NF4) on gx10 shows **96% GPU utilization**.
(Cannot test on yoga — FP16 LoRA needs 23.5 GB, yoga has 8 GB.)

```
Observation:  96% GPU utilization sustained for 30+ minutes
Comparison:   NF4 QLoRA = 0% GPU (CPU lm_head fallback)
Conclusion:   NF4 dequant is the SOLE root cause — GPU forward works fine without it
```

**Two bugs found, one fixed:**

1. **trueno#233 (FIXED in local HEAD):** V-projection NF4 dequant produced all zeros
   in crates.io release. Fixed in local entrenar — V-proj now dequants correctly
   (nonzero=350K+). Needs crates.io publish.

2. **entrenar#317 (NEW):** Even with correct dequant, NaN persists in the inference-style
   CPU forward path. On yoga (8GB), GPU can't fit embeddings (1780MB > 1228MB free),
   so lm_head runs on CPU via inference-style forward, which produces NaN.
   On gx10 (120GB), everything fits on GPU → 96% utilization → works.

**VRAM budget on yoga 8GB (measured 2026-04-02):**

| Component | Size | Running Total |
|-----------|------|---------------|
| Base model (NF4 28 layers) | ~4.0 GB | 4.0 GB |
| Embedding (single layout) | 0.89 GB | 4.9 GB |
| LoRA optimizer states | 0.74 GB | 5.6 GB |
| Training scratch buffers | ~1.5 GB | 7.1 GB |
| CUDA overhead | ~0.5 GB | 7.6 GB |

**Three fixes landed upstream (2026-04-02), OOM resolved:**
- entrenar f9845e07: Embedding VRAM 1780→890MB (single layout)
- aprender ba0e392f: Rank override `--rank 16` respected (optimizer 0.05 GB)
- entrenar (local): V-projection NF4 dequant fixed (nonzero=350K+)

**Training state initializes — no OOM.** But layer 0 forward produces ALL ZEROS:
```
[CUDA-FWD] layer 0: first8=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Weights are correct (dequant nonzero=350K+), input is non-zero (from embedding),
but `CudaNf4TransformerBlock::forward` outputs zeros. Filed: entrenar#318.

**DIAG-002 finding (contract apr-training-parity-v1.yaml):**
```
[DIAG-002] m=74 k=1536 n=1536 A=[0,0,0,0] B=[0,0,0,0] C=[0,0,0,0]
```
Both GEMM inputs (A=RMSNorm output, B=weight buffer) read as zeros in forward,
despite upload traces showing nonzero=350K+ at construction time. `make_current()`
added to NF4 forward (pushed to entrenar) — didn't fix it.

**CORRECTION (2026-04-02):** Previous diagnostics (DIAG-002 through 005) were WRONG —
`copy_to_host` with partial buffer silently fails, producing zero readback. Fixed in
trueno (4a7838a4: partial readback support). Weights ARE on GPU (nonzero=2.1M).

**Actual finding — activation explosion:**
```
[LAYER] L0  output=[-0.09, -0.08, 0.36, -0.12]     (normal)
[LAYER] L14 output=[332, 33, 1913, 259]             (4 orders of magnitude growth)
[LAYER] L27 output=[NaN, NaN, NaN, NaN]             (overflow)
```

Within layer 0: RMSNorm, Q-GEMM, attention, FFN all valid. Activations grow ~100x
per 14 layers across the residual chain until NaN at layer 27. RMSNorm should prevent
this. Inference path (realizr, same Q4K model, 151 tok/s) works fine — so the norm
weights are correct. The bug is in how entrenar's NF4 block applies RMSNorm or
accumulates residuals.

**Root cause (traced 2026-04-02):** Q4K dequantized weights have sufficient precision
for single-pass inference (realizr: 151 tok/s) but compound numerical errors through
28 cuBLAS GEMM residual layers in training. Error grows ~100x per 14 layers until NaN.
Direct fp32 upload (bypassing NF4 roundtrip) doesn't help — the Q4K→fp32 weights
themselves are the problem. FP16 model works on gx10 (96% GPU, no NaN) but OOMs on yoga.

**Fix path for yoga 8GB:** mixed-precision (bf16) cuBLAS GEMM or activation clamping.
**Fix path for gx10 120GB:** use FP16 model (already works, 96% GPU).

**Per-layer trace (2026-04-02):** Activation explosion across samples, not just layers.
First sample L0-L5: values grow from 0.19 to 7.11 (normal). Between samples,
values jump to 56, then 1251, then 31668, then NaN. The shared scratch or
ping-pong buffers may carry contamination between training steps.

The fp32 CudaTransformerBlock (per-block scratch) works. The NF4 block
(shared scratch C-SCRATCH-001) fails. The shared scratch is used for BOTH
forward and backward, and may not be properly reset between training steps.

**TRAINING WORKING (2026-04-02):** APR NF4 QLoRA training on yoga 8GB — 43 tok/s canary confirmed.
Full GPU pipeline: embed → 28 NF4 layers → BT GEMM lm_head → fused cross-entropy.
Root cause chain: (1) backward gradient contamination of CudaBlockScratch (C-SCRATCH-001) — 
fixed by zeroing 21 scratch buffers per forward. (2) Multi-epoch NaN cascade from
InstructGpuTrainingState backward buffers (grad_buf_a/b, grad_hidden_buf, output_scratch,
logits_buf) — fixed by zeroing 5 training state buffers per forward (PMAT-453).
(3) L5 violations: `let _ =` on copy_from_buffer silenced GPU errors — all 4 fixed.

**Filed:** entrenar#318 (10+ comments with progressive diagnosis).
**Upstream fixes pushed (21 commits, 3 repos):**
- trueno 4a7838a4: copy_to_host partial readback
- entrenar f9845e07: VRAM embedding 1780→890MB
- entrenar c605ea16: make_current in NF4 forward
- entrenar 475256c6: direct_transpose_upload (skip NF4 roundtrip)
- entrenar a515d2f9: original fp32 upload (no NF4, no transpose)
- aprender ba0e392f: respect --rank 16 flag
- entrenar 8966424b: zero training state GPU buffers + L5 violations (PMAT-453)

**Why this matters:** Every downstream optimization (chunked lm_head, cuBLAS tensor
cores, fused kernels) is BLOCKED until GPU forward works. Fixing 11 V-projection
tensors in trueno's NF4 dequant is the single gate that unlocks 44 → 2000+ tok/s.

### Parity Roadmap: 194 → 6,628 tok/s (34x gap)

Five-whys root cause: entrenar uses per-GEMM cuBLAS calls with fp32 dequantized weights
(9.4 MB/GEMM at 256 GB/s = memory-BW bound), while unsloth uses fused Triton kernels
that stay entirely on GPU with fp16 tensor core compute.

| Tier | Fix | Expected | Measured | Status |
|------|-----|----------|----------|--------|
| **1** | `cuMemsetD32Async` (GPU-side zero) | 186→300 | **194** (canary, 2026-04-02) | DONE — zeroing was NOT the bottleneck |
| **2** | FP16 weights + cuBLAS fp16 GEMM (tensor cores) | →390 | — | **SHIPPED** — Q/K/V fp16 dispatch (entrenar@82b484fa), cast kernel (trueno@05217c48). FP16_GEMM=1 to enable. |
| **3** | CUDA graphs (capture 28-layer forward, replay) | →1200 | — | **CONTRACT DESIGNED** — cuda-graph-training-v1.yaml, entrenar#322 |
| **4** | Fused NF4 dequant+GEMM kernels (like Triton) | →3000 | — | Requires trueno kernel work |
| **5** | Fused attention + FFN blocks (196→56 launches) | →5000 | — | Requires trueno kernel work |
| **6** | Flash attention + memory BW optimization | →6000+ | — | **parity** |

**Measured GPU utilization: 7%** (nvidia-smi). Root cause (revised): NOT kernel launch
overhead (588 launches × 5μs = 3ms, negligible). The 93% idle is **CPU-bound work**:
- CPU embedding lookup (151936×1536 matmul per sample)
- CPU loss gradient through lm_head
- CPU LoRA AdamW optimizer step
- Dataset tokenization/iteration

Also: cuBLAS uses SIMD (~2 TFLOPS) not tensor cores (~83 TFLOPS) due to ALB-076
(TF32 + transposed backward = NaN at gradient magnitude ~1e5). BF16 backward
would avoid this bug and use tensor cores.

**DEFINITIVE FINDING (2026-04-02):** The 197 tok/s ceiling is MEMORY BANDWIDTH bound.

TF32 tensor cores enabled (41x compute): 196→197 tok/s (0% improvement).
cuBLAS SIMD (no tensor cores): 196 tok/s. Same throughput.
Reason: weight matrix loads dominate. 1536×1536 = 9.4 MB at 256 GB/s = 37μs.
Compute: 1.9μs (TF32) or 80μs (SIMD). Both dwarfed by 37μs memory latency.

**The ONLY path to parity: reduce memory traffic per sample.**

1. **Fused QKV projection** — read weight matrix ONCE, compute Q+K+V in one kernel
   (3x reduction in weight reads for attention projections)
2. **Flash attention** — don't materialize seq×seq attention matrix (seq²×heads BW savings)
3. **FP16 weights** — halve weight memory from 9.4 MB to 4.7 MB per GEMM
4. **Fused FFN** — gate+up+SwiGLU+down in one kernel (4x reduction in FFN weight reads)

These are all kernel fusion tasks in trueno — the same optimizations that make
unsloth's Triton kernels fast. Without fusion, the GPU's 256 GB/s bandwidth
limits throughput regardless of tensor core compute power.

**Tier 2** requires FP16 model (already exists on yoga: 3.4 GB). With the VRAM
optimizations from this session (embedding halving, rank override), FP16 weights
should fit. cuBLAS fp16 GEMM uses tensor cores at 2x throughput vs fp32.

**Tiers 3-6** require deeper trueno/entrenar kernel engineering. CUDA graphs (Tier 3)
are the highest-leverage: `apr profile` showed 84.2% kernel launch overhead on inference.
Training has even more launches (forward + backward).

### Backend Parity Mandate

**ALL backends must be tested for parity.** A runtime that works on one backend
but not another is broken. The canary must validate every backend combination.

| Backend | Compute Path | Status | Canary |
|---------|-------------|--------|--------|
| **CPU** | Scalar Rust | Baseline (slow) | `--gpu-backend cpu` |
| **SIMD** | trueno AVX2/NEON | Should match CPU results | `--gpu-backend cpu` + SIMD feature |
| **cuBLAS SIMD** | cuBLAS `DEFAULT_MATH` | **197 tok/s** (yoga) | `--gpu-backend cuda` |
| **cuBLAS TF32** | cuBLAS tensor cores | 197 tok/s (memory-bound) | `--gpu-backend cuda` (current) |
| **PTX naive** | Hand-written PTX GEMM | Fallback when no cuBLAS | PTX path auto-selected |
| **NF4 fused PTX** | `gemm_nf4_forward` | **33 tok/s** (23% slower, loss=15.80) | Canary confirmed |
| **wgpu/Vulkan** | WGSL compute shaders | 6,730 tok/s synthetic, real model TBD | `--gpu-backend wgpu` |

**Parity tests required:**
1. Every backend must produce loss < 200 on the same 50-sample canary dataset
2. Throughput regression: each backend must stay within 10% of its own baseline
3. Numerical parity: cuBLAS vs PTX vs NF4 fused must produce < 0.01 loss divergence
4. New backends (wgpu training, Metal) must pass the same canary before merge

**NF4 fused kernel finding (2026-04-03 update):**
`gemm_nf4_forward` reads 8x less data (NF4 packed vs fp32) but runs 23% slower
(33 tok/s vs 43 tok/s cuBLAS). 100% GPU utilization (compute-bound) vs 7% with
cuBLAS (memory-bound). Loss is lower (15.80 vs 16.80) — possibly from higher
precision in the fused dequant path. Kernel needs tensor core integration and
better tiling to beat cuBLAS — filed as trueno#234.

### P3: Cross-platform parity

1. **wgpu/burn real model loading** — burn can't load HF safetensors/APR yet.
   Current 6,730 tok/s is on synthetic MLP. Real Qwen training on Vulkan needs
   model loading support in burn or APR format reader.

2. **entrenar CUDA-free compilation** (aprender#564) — enables wgpu-only builds
   for AMD/Intel GPU training without CUDA toolkit.

3. **Metal backend** — Apple M-series GPU training via Metal Performance Shaders.

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-RD-01 | torch.compile >10% regression | FALSIFIED: -11% at canary length |
| F-RD-02 | DeepSpeed ZeRO-2 OOMs on yoga | Planned |
