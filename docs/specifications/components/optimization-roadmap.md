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

**63+ fixes landed** in entrenar/trueno/aprender. WGPU profiler deployed (PMAT-480 DONE): 100% GPU compute, 0% overhead (gx10). Yoga WGPU UNBLOCKED (PMAT-493 DONE, libvulkan1 installed). **First yoga WGPU measurement: ~191 tok/s (5 steps before buffer crash, PMAT-498).** gx10: 470 tok/s (async). Convergence defect filed (PMAT-497): loss 11.74 vs unsloth 0.45. Tier 2 (FP16), Tier 4 (fused kernels), Tier 4.7 (tensor cores) ALL SHIPPED and WIRED — per-layer profiler shipped (PMAT-480), tensor core GEMM wired into all 7 projections (PMAT-481). Two blockers found 2026-04-04: (1) `apr finetune` missing from binary (training feature dropped by trueno-gpu compile errors) — FIXED upstream, (2) GGUF tensor names (`token_embd.weight`) not mapped to HF names (`model.embed_tokens.weight`) — filed PMAT-489. APR metadata completeness enforced via architecture preset fallback (aprender@39d33259).

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
| 22 | CPU lm_head backward fallback (PMAT-471) | entrenar | Backward works on yoga 8GB |
| 23 | `set_fp16_weights()` — FP16 weight cast (PMAT-470) | entrenar | FP16 GEMM path was DEAD CODE, now functional |
| 24 | cuBLAS workspace pre-alloc (PMAT-063) | entrenar | CUDA graph capture unblocked on sm_89 |
| 25 | FP16 backward GEMM + fp32 drop (PMAT-472) | entrenar | 2.6 GB freed, GPU lm_head enabled |
| 26 | FP16 path bugfixes: buffer resize + K/V guard + alloc reuse (PMAT-474) | entrenar | 3 crash bugs fixed before first measurement |

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
| paiml/entrenar#323 | CPU lm_head backward fallback (PMAT-471) | **FIXED** (2026-04-03, fix #22) |
| paiml/entrenar#324 | FP16 weight cast `set_fp16_weights()` (PMAT-470) | **FIXED** (2026-04-03, fix #23) |
| paiml/entrenar#325 | cuBLAS workspace pre-alloc (PMAT-063) | **FIXED** (2026-04-03, fix #24) |
| paiml/entrenar#326 | FP16 backward GEMM + fp32 drop (PMAT-472) | **FIXED** (2026-04-03, fix #25) |
| paiml/entrenar#327 | FP16 path crash bugs: resize + guard + alloc (PMAT-474) | **FIXED** (2026-04-03, fix #26) |
| paiml/qwen-train-canary#22 | NF4 kernel fusion (PMAT-475) | **SHIPPED** — RMSNorm+NF4 GEMV + Gate+Up NF4 GEMM in trueno. Saves 336 MB/step DRAM. |
| paiml/qwen-train-canary#23 | Backward graph capture (PMAT-464/477) | **SHIPPED** — fused clip + backward_graph.rs + instruct_pipeline split (12 files ≤500) |
| paiml/qwen-train-canary#24 | FP16 throughput measurement (PMAT-476) | Open — canary-apr-fp16 never executed |
| paiml/qwen-train-canary#27 | Training step profiling (PMAT-480) | Open — BrickProfiler integration for per-layer training timing |
| paiml/entrenar#328 | Wire BrickProfiler into training forward+backward | Open — upstream dependency for PMAT-480 |
| paiml/entrenar#329 | Wire NF4 tensor core GEMM into training forward | **FIXED** (2026-04-03, fix #37) — PMAT-481 |
| paiml/entrenar#330 | Fused backward GEMM for Gate+Up and K+V | Open — PMAT-482 |
| aprender@39d33259 | Fix APR metadata fallback — 1.5B preset + architecture+hidden_size match (GH-376) | **FIXED** (2026-04-04, fix #54) — PMAT-490 |
| PMAT-489 | GGUF tensor name mapping in apr finetune (token_embd vs model.embed_tokens) | **FIXED** (2026-04-04) — 11 unit tests verify complete mapping |
| PMAT-490 | APR v2 metadata completeness — GGUF imports missing num_heads/num_layers | **FIXED** via fallback (2026-04-04) — provable-contract enforcement TBD |
| aprender (local) | Respect `--gpu-backend wgpu` in WGPU routing condition | **FIXED** (2026-04-04, fix #63) — `(!cuda_ok \|\| gpu_backend == "wgpu")` |
| yoga Vulkan | `libvulkan.so.1` missing — wgpu "Parent device is lost" | **FIXED** (2026-04-05) — `apt install libvulkan1`, Vulkan verified, first WGPU measurement |
| PMAT-495 | gx10 binary rebuild blocked — alimentar+trueno code gen (generated_contracts) | **FIXED** (2026-04-05) — binary built, WGPU training working |
| PMAT-497 | APR convergence defect: loss 18.9 vs ref 1.51 (12.5x gap) | **OPEN** (critical) — Q4K dequant code looks correct (dequant_q4k.rs, extract_scale_min), layout comment confirms no-transpose. Per-op trace shows `attention=0ms` on yoga — suspect WGSL forward attention returns zeros. Reference: HF fp32 loss = 1.5126 on same input. |
| PMAT-498 | Yoga WGPU crash after 5 steps: Buffer label invalid | **OPEN** (critical) — wgpu buffer validation in loss readback |
| gx10 ARM dequant | Q4K→F32 CPU dequant takes 100+ min on GB10 ARM (vs 20 min x86_64) | **MEASURED** — WGPU fast path essential for ARM targets |

### Upstream Fixes (2026-04-03, fixes #27-28)

| # | Fix | Repo | Impact |
|---|-----|------|--------|
| 27 | Fused LoRA gradient clipping (PMAT-477) | entrenar | 168 D2H sync points → 0 per backward; enables CUDA graph capture |
| 28 | Fused RMSNorm + NF4 GEMV kernel (PMAT-475) | trueno | Eliminates global memory roundtrip between RMSNorm and NF4 GEMV |
| 29 | Backward graph capture infra (PMAT-464) | entrenar | backward_graph.rs: capture/replay for full backward loop |
| 30 | instruct_pipeline.rs → 12 files ≤500 lines | entrenar | Toyota Way: eliminated 4114-line monolith, all files ≤500 |
| 31 | Fused NF4 Gate+Up GEMM kernel (PMAT-475) | trueno | Shared input load: 336 MB/step DRAM eliminated for FFN (2/3 compute) |
| 32 | Wire fused Gate+Up into training forward (PMAT-475) | entrenar | NF4_FUSED_GEMM=1 uses single fused kernel instead of 2× gemm_nf4_forward |
| 33 | Fused K+V NF4 GEMM for GQA attention (PMAT-478) | entrenar | Reuses Gate+Up kernel for K+V (same dim in GQA). 352 MB/step saved. |
| 34 | NF4 tensor core GEMM — WMMA 16×16×16 (PMAT-479) | trueno | Dequant NF4→FP16 in SHMEM + WMMA mma.sync. 5-40x compute vs naive. |
| 35 | Training step profiling contract (PMAT-480) | qwen-train-canary | 12 falsification tests for scientific per-layer training profiling. |
| 36 | Per-layer training profiler (PMAT-480) | entrenar | begin_layer/end_layer_fwd/end_layer_bwd wired into forward+backward. |
| 37 | NF4 tensor core GEMM wiring (PMAT-481) | entrenar | NF4_TC_GEMM=1 dispatches to WMMA kernel in all 7 forward projections. |

### Contracts

| Contract | Status |
|----------|--------|
| cuda-training-forward-v1.yaml | 5/5 falsified, 3 resolved |
| chunked-lm-head-v1.yaml | Design complete, superseded by CPU lm_head |
| nf4-fused-rmsnorm-gemv-v1.yaml | Designed, 4 falsification tests PENDING (PMAT-475) |
| nf4-fused-gate-up-swiglu-v1.yaml | Designed, 4 falsification tests PENDING (PMAT-475) |
| cuda-graph-backward-v1.yaml | Designed, 4 falsification tests PENDING (PMAT-477) |
| training-step-profiling-v1.yaml | Designed, 12 falsification tests PENDING (PMAT-480) |

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

### Parity Roadmap: 421 → 5,262 tok/s (13x gap, WGPU path)

**CRITICAL FINDING (2026-04-05, PMAT-496):** APR GPU compute is ALREADY 5.7x FASTER
than unsloth (30,016 vs 5,262 tok/s at step level). The 13x wall-clock gap is entirely
**inter-step overhead OUTSIDE train_step()** — 98.3% of wall time is between calls.

**REVISED profiling (2026-04-05, v6.5.0):** Real measurement from existing binary
```
15 steps profiled (1 epoch):
  In-step total:   2,001ms  (1.7% of wall)    → avg 68ms/step (steady state)
  Inter-step:    113,099ms  (98.3% of wall)    → avg 7,540ms between steps
  Wall clock:    115,100ms  (training phase only, excludes 5s pipeline init)
```

**F-PROF-002 FALSIFIED:** Buffer allocation INSIDE train_step is NOT the bottleneck.
The overhead is OUTSIDE train_step — in `pipeline.encode()` (tokenization, 2×/sample),
the epoch iterator, and/or `queue.submit` serialization between steps.

**Revised fix priorities (re-ordered by measurement):**

**v6.5.0 profiler result (1-epoch, gx10 GB10):**
```
50 steps, 57.7s wall:
  sync:         92.5%  (53.4s — device.poll(Wait) in read_loss)
  gpu_compute:   7.4%  (4.3s — actual GPU forward+backward+optimizer)
  overhead:      0.1%
  coverage:    100.0%
```

1. **Async loss readback (P0)** — `read_loss()` calls `device.poll(Maintain::Wait)` which
   blocks until ALL async GPU dispatches complete. This serializes the pipeline.
   Fix: read loss once per epoch (not per step), or use non-blocking poll with loss
   accumulation buffer. **Expected: 1150ms → ~90ms/step = 5,600+ tok/s (parity).**

2. **Pre-tokenize corpus (DONE)** — eliminated 800 tokenizer calls per training.
   Saved 3ms total (was not the bottleneck).

3. **Epoch-level loss reporting** — currently reads and prints loss after EVERY sample.
   Each readback forces a full GPU sync. Moving to epoch-level reporting = 8 syncs
   instead of 400.

4. **Buffer pre-allocation** — secondary: reduces the 90ms GPU compute time further.

**Key insight:** The async GPU pipeline works perfectly — 7.4% GPU compute for 50
training steps. But `device.poll(Wait)` after each step destroys the pipeline's
asynchronous advantage by forcing CPU-GPU synchronization 400 times per training run.

Previous five-whys (CUDA path): entrenar uses per-GEMM cuBLAS calls with fp32 dequantized
weights (9.4 MB/GEMM at 256 GB/s = memory-BW bound), while unsloth uses fused Triton
kernels that stay entirely on GPU with fp16 tensor core compute.

| Tier | Fix | Expected | Measured | Status |
|------|-----|----------|----------|--------|
| **1** | `cuMemsetD32Async` (GPU-side zero) | 186→300 | **194** (canary, 2026-04-02) | DONE — zeroing was NOT the bottleneck |
| **1.5** | CPU lm_head backward fallback (PMAT-471) | enables training | — | **SHIPPED** — entrenar@de2ad7e1, entrenar#323. Without this, backward NEVER ran on yoga 8GB. |
| **2** | FP16 weights + cuBLAS fp16 GEMM (tensor cores) | →390 | — | **SHIPPED** — forward + backward FP16 GEMM, fp32 weights DROPPED. FP16_GEMM=1 frees ~2.6 GB VRAM (entrenar#324, #326). |
| **2.5** | cuBLAS workspace pre-alloc (PMAT-063) | unblocks CUDA graphs | — | **SHIPPED** — entrenar@de2ad7e1, entrenar#325. 32 MB pre-alloc before graph capture. |
| **2.7** | FP16 backward + fp32 drop (PMAT-472) | GPU embeddings fit | — | **SHIPPED** — entrenar@435e9762, entrenar#326. Backward uses tensor cores. fp32 dropped → 2.6 GB freed → GPU lm_head. |
| **3** | CUDA graphs (capture 28-layer forward, replay) | →1200 | — | **FORWARD SHIPPED** — PMAT-464. Backward DEFERRED: optimizer step dependency + gradient clipping sync blockers. |
| **4** | Fused NF4 Gate+Up GEMM (shared input load) | →3000 | — | **SHIPPED** — PMAT-475. Eliminates 336 MB/step DRAM for FFN. |
| **4.5** | Fused K+V NF4 GEMM (GQA attention reuse) | →3500 | — | **SHIPPED** — PMAT-478. Saves 352 MB/step. 688 MB total DRAM eliminated. |
| **4.7** | NF4 tensor core GEMM (WMMA 16×16×16) | →4000 | — | **WIRED** — PMAT-479+481. Dequant NF4→FP16 in SHMEM + WMMA mma.sync. All 7 projections. |
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

### P4: 100x Throughput Path (arXiv-informed, 2026-04-04)

Research-backed roadmap for reaching unsloth parity and beyond.
Combined target: 194 tok/s → 22,500+ tok/s (116x improvement).

**Tier 7: CUDA Graph Backward Capture (PMAT-488)** — 6.5x
- Forward graph shipped (PMAT-464), backward unblocked by fused clip (PMAT-477)
- Eliminates 84.6% kernel launch overhead (89,756µs → ~50µs per step)
- Reference: PyGraph (arXiv:2503.19779) — >2x benefit in PyTorch training
- Reference: CUDA Graph Batching (arXiv:2501.09398) — >1.4x from optimal batch
- Canary: `CUDA_GRAPH=1` env var, `canary-apr-graph` target
- Status: **READY TO IMPLEMENT** (all blockers resolved)

**Tier 8: Flash Attention Integration** — 2.5x additional
- Attention is 71.4% of forward time (420 separate kernel launches)
- Fuse Q@K^T + softmax + @V into single kernel (420 → 28 launches)
- trueno has FlashAttention + Tensor Core variant (forward only, issue #85 for backward)
- Reference: FlashAttention-3 (arXiv:2407.08608) — 740 TFLOPs/s (75% H100 utilization)
- Reference: FlashAttention-4 (arXiv:2603.05451) — asymmetric hardware pipelining
- Status: **PLANNED** (trueno forward exists, backward needed)

**Tier 9: Mirage-style Persistent Megakernel** — 1.7x additional
- Compile entire transformer block as single persistent kernel
- SM-level pipelining across layers (no kernel boundaries)
- trueno already has `TransformerBlockMegakernel` as starting point
- Reference: Mirage Persistent Kernel (arXiv:2512.22219) — entire model as one kernel
- Reference: Hazy Research (2025) — Llama-1B megakernel, 78% memory BW on H100
- Status: **RESEARCH** (architecture design needed)

**Combined: Tier 7 × Tier 8 × Tier 9 = 6.5 × 2.5 × 1.7 = 27.6x → ~5,300 tok/s minimum**
With NaN fix (2.9x) and TC GEMM (2.0x): 27.6 × 2.9 × 2.0 = **160x → ~31,000 tok/s**

### P5: Parity Profiling System (PMAT-487)

Scientific measurement infrastructure for cross-runtime comparison.

1. **torch.profiler in PyTorch + unsloth canaries** — per-kernel CUDA timing
2. **parity-profile-v1 schema** — common JSON format across all runtimes
3. **renacer CUPTI training traces** — ground-truth GPU kernel timing for APR
4. **probar parity scorecard** — automated cross-runtime gap analysis
5. **scripts/parity-report.py** — side-by-side Markdown comparison

Reference: SKIP framework (arXiv:2504.11750) — CUPTI-based GPU kernel profiling

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-RD-01 | torch.compile >10% regression | FALSIFIED: -11% at canary length |
| F-RD-02 | DeepSpeed ZeRO-2 OOMs on yoga | Planned |
| F-RD-03 | CUDA graph backward captures correctly | Planned (PMAT-488) |
| F-RD-04 | Parity profiling shows APR attention >5x slower than unsloth | Expected (no Flash Attention) |
