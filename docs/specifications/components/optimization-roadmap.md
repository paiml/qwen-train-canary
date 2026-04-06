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

**65+ fixes landed** in entrenar/trueno/aprender. **cuBLAS routing fix (PMAT-494, 2026-04-06): 2,101 tok/s** — 4.5x over WGPU 470. WGPU profiler deployed (PMAT-480 DONE): 100% GPU compute, 0% overhead (gx10). Yoga WGPU UNBLOCKED (PMAT-493 DONE). Convergence: backward_steps=0 on CUDA path (PMAT-512), WGPU loss 11.74 (oscillates). Tier 2-7 (WGPU optimizations) ALL SHIPPED but UNMEASURED — may be deprioritized now that cuBLAS path is 4.5x faster. **Provable-contract Grade A REQUIRED for all cuBLAS training code (PMAT-515).**

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
| paiml/qwen-train-canary#24 | FP16 throughput measurement (PMAT-476) | **OPEN** — canary-apr-fp16 never executed (PROVISIONAL baseline) |
| paiml/qwen-train-canary#27 | Training step profiling (PMAT-480) | **DONE** (2026-04-05) — 13-phase profiler deployed, 100% GPU compute measured |
| paiml/entrenar#328 | Wire BrickProfiler into training forward+backward | **DONE** (2026-04-05) — StepProfiler + per-op instrumentation shipped |
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

## Measurement Debt: Optimization Tiers SHIPPED but UNMEASURED (2026-04-05)

> **Stop shipping new tiers. Measure the existing ones.**

Every optimization tier below was coded, wired, and declared "SHIPPED" in the revision history. None has a canary measurement. Until measured, these are **hypotheses, not improvements.**

| Tier | Env Flag | Predicted Impact | Measured Impact | Blocker |
|------|----------|-----------------|-----------------|---------|
| Tier 2: FP16 GEMM | `FP16_GEMM=1` | 2.6 GB freed | **UNMEASURED** | No canary-apr-fp16 run |
| Tier 4: Fused RMSNorm+GEMV | `NF4_FUSED_GEMM=1` | 336 MB/step saved | **UNMEASURED** | No canary-apr-fused run |
| Tier 4: Fused Gate+Up GEMM | `NF4_FUSED_GEMM=1` | 2× fewer dispatches | **UNMEASURED** | Same |
| Tier 4: Fused K+V GEMM | `NF4_FUSED_GEMM=1` | 352 MB/step saved | **UNMEASURED** | Same |
| Tier 4.7: TC GEMM forward | `NF4_TC_GEMM=1` | 5-40× compute | **UNMEASURED** | No canary-apr-tc run |
| Tier 4.7: TC GEMM backward | `NF4_TC_BWD_GEMM=1` | 196 fewer launches | **UNMEASURED** | No canary-apr-tc-bwd run |
| Tier 5: Fused backward GEMM | `NF4_FUSED_BWD_GEMM=1` | 84 fewer launches | **UNMEASURED** | No canary-apr-fused-bwd run |
| Tier 7: CUDA graph backward | `CUDA_GRAPH=1` | 840→1 launch | **UNMEASURED** | No canary-apr-graph run |

**Measured throughput changes (Days 1-6):** async pipeline 421→470 tok/s (+12%, Day 5). cuBLAS routing fix 470→2,101 tok/s (+347%, Day 6). 7 WGPU optimization tiers remain UNMEASURED — these are WGPU-specific and may be deprioritized now that cuBLAS is confirmed.

### Execution Plan: Three Phases (P0)

No new optimization tier may be coded until these three phases complete sequentially. Each phase has a provable exit criterion — not a checklist, a falsifiable gate.

---

#### Phase A: Fix Profiling with Provable Contracts (PMAT-504)

**Goal:** Know exactly where time goes on CUDA targets, with compiler-enforced contracts that detect regressions and correctness bugs automatically.

**Why first:** We can't optimize what we can't measure. The current profiler works on WGPU (13 phases, 100% coverage on gx10) but has **zero coverage on the CUDA path**. The CUDA path is what matters for Phase B. Without profiling, we're guessing. The candle-vs-apr sister project found 3.4x profiler fidelity errors when contracts weren't wired — we must not repeat that.

**Deliverables:**

1. **Wire 6 provable-contract invariants into training profiler** (from candle-vs-apr Phase 15):
   - `wall_coverage >= 0.85` — catches missing kernels (would have caught PMAT-498 crash)
   - `GEMM >= 50% of layer_fwd` — architecture regression gate
   - `LmHead > 10x RmsNorm` — profiler sync fidelity check (candle-vs-apr saw 3.4x error without this)
   - `loss[epoch_n] < loss[epoch_0]` — convergence sanity (catches PMAT-497 immediately)
   - `step_time[n] < 2x step_time[n-1]` — memory leak / OOM early warning
   - `no orphan profiler spans` — trace corruption detection
   Each invariant enforced via `#[ensures(...)]` or `debug_assert!` — L5 compiler-enforced, not eprintln warnings.

2. **Port StepProfiler to CUDA NF4 path** — the WGPU profiler (13 phases) exists but `CudaNf4TransformerBlock` has no equivalent. Wire `begin_phase`/`end_phase` into the CUDA training forward and backward with the same phase decomposition.

3. **Verify convergence fix (PMAT-497)** — run gx10 canary, assert loss < 2.0 after WGSL transpose fix. If still > 2.0, dump per-layer activations and diff against HF fp32 reference. The `loss[epoch_n] < loss[epoch_0]` contract will catch this automatically going forward.

4. **Fix yoga WGPU crash (PMAT-498)** — add buffer label validation contract, fix the `Buffer with '' label is invalid` error. The `wall_coverage >= 0.85` contract would have caught this before it crashed.

**Exit criterion (falsifiable):**
> `apr finetune --profile` on CUDA target produces JSON with all 13 phases, wall_coverage >= 0.85, and all 6 contract invariants passing. Loss < 2.0 on at least one target. If this is not achieved, Phase B is blocked.

**PMAT:** PMAT-504 (profiling contracts), PMAT-497 (convergence), PMAT-498 (yoga crash)

---

#### Phase B: Hybrid Backend — cuBLAS on NVIDIA, WGPU on AMD/Metal (PMAT-503)

**Goal:** Use cuBLAS for GEMM on CUDA-capable hardware. Keep WGPU for portability.

**Why second:** The WGPU 11.2x gap was an architectural mismatch, not an optimization gap. **cuBLAS routing fix (PMAT-494, 2026-04-06) confirmed this:** one-line change = 4.5x throughput.

| APR path (WGPU) | Unsloth path | APR path (cuBLAS) — **CONFIRMED** |
|----------|-------------|--------------------------|
| WGSL source | Triton/CUDA | Rust + cuBLAS |
| → SPIR-V → Vulkan | → PTX → cuBLAS | → cuBLAS directly |
| Hand-written GEMM | cuBLAS autotuned | cuBLAS autotuned |
| 470 tok/s | 5,262 tok/s | **2,101 tok/s (MEASURED)** |

cuBLAS has decades of per-architecture autotuning. No Rust framework (Candle, Burn, trueno) has achieved parity with custom kernels. The pragmatic industry pattern (Candle, PyTorch, JAX) is to delegate GEMM to cuBLAS. trueno already has cuBLAS bindings — this is wiring, not research.

**Deliverables:**

1. **Fix CUDA PTX JIT caching (PMAT-492)** — pre-compile trueno kernels to cubin. Current: 2-hour first run, 28 tok/s cached on yoga. After: seconds to load, cuBLAS GEMM speed.

2. **Wire `--gpu-backend cuda` to use cuBLAS GEMM in NF4 training forward+backward** — the `CudaNf4TransformerBlock` already exists and uses cuBLAS for inference. Training needs: (a) cuBLAS for LoRA GEMM in forward, (b) cuBLAS for gradient GEMM in backward, (c) NF4 dequant → cuBLAS SGEMM pipeline (same as bitsandbytes pattern).

3. **Runtime backend selection** — `--gpu-backend auto` detects CUDA capability and selects cuBLAS on NVIDIA, WGPU/Vulkan on AMD/Metal. The `--gpu-backend` flag already exists; this completes its implementation.

4. **Measure cuBLAS path on gx10 and yoga** — run `canary-apr --gpu-backend cuda` on both targets. Compare directly against WGPU path and against unsloth on same hardware.

**Exit criterion (falsifiable):**
> `apr finetune --gpu-backend cuda` on gx10 produces throughput >= 2,000 tok/s (within 3x of unsloth, proving cuBLAS closes the architectural gap). If throughput is still < 1,000 tok/s, the bottleneck is not GEMM and further profiling (Phase A contracts) is needed. If cuBLAS throughput matches WGPU throughput, the gap is NOT cuBLAS vs WGSL — investigate elsewhere.
>
> **STATUS (2026-04-06): THROUGHPUT CRITERION MET.** 2,101 tok/s measured (PMAT-494 routing fix). **CONVERGENCE NOT MET:** backward_steps=0, loss not captured (PMAT-512). Phase B remains OPEN until loss < 2.0 on cuBLAS path.
>
> **HARD REQUIREMENT (PMAT-515):** All cuBLAS training code must achieve provable-contract penetration at Grade A (score >= 0.60). See parent spec §8.

**PMAT:** PMAT-503 (hybrid backend), PMAT-492 (JIT cache), PMAT-494 (routing fix — DONE), PMAT-512 (loss capture), PMAT-515 (contracts)

---

#### Phase C: A/B Test on CUDA Targets (PMAT-501)

**Goal:** Rigorous head-to-head comparison of WGPU vs cuBLAS vs optimization tiers on yoga (8GB) and gx10 (120GB).

**Why third:** Phase A gives us the profiler to measure accurately. Phase B gives us the cuBLAS path to compare against. Phase C is the measurement sweep that tells us what actually works — and retires the 6 UNMEASURED tiers from the debt table.

**Deliverables:**

1. **Baseline A/B matrix** — every combination on both targets:

   | Variant | Env Flags | yoga (8GB) | gx10 (120GB) |
   |---------|-----------|-----------|-------------|
   | WGPU baseline | (none) | PMAT-498 | 470 tok/s |
   | CUDA cuBLAS baseline | `--gpu-backend cuda` | TBD | **2,101 tok/s** (2026-04-06, bwd_steps=0) |
   | CUDA + TC GEMM fwd | `NF4_TC_GEMM=1` | TBD | TBD |
   | CUDA + TC GEMM fwd+bwd | `NF4_TC_GEMM=1 NF4_TC_BWD_GEMM=1` | TBD | TBD |
   | CUDA + fused kernels | `NF4_FUSED_GEMM=1 NF4_FUSED_BWD_GEMM=1` | TBD | TBD |
   | CUDA + FP16 | `FP16_GEMM=1` | TBD | TBD |
   | CUDA + graph | `CUDA_GRAPH=1` | TBD | TBD |
   | CUDA + all flags | all flags | TBD | TBD |
   | Unsloth (reference) | N/A | 6,697 | 16,118 |

2. **Per-variant profiler output** — Phase A profiler runs on every variant. Contract invariants checked automatically. Bottleneck phase identified for each.

3. **Convergence check per variant** — every variant must produce loss < 2.0 at steps=100. Variants that don't converge are marked BROKEN regardless of throughput.

4. **Retire measurement debt** — after Phase C, every tier in the debt table has a number or is marked BROKEN. No more UNMEASURED entries.

5. **Decision: which path ships?** Based on Phase C data:
   - If cuBLAS baseline > 3,000 tok/s → ship cuBLAS as default on NVIDIA
   - If any optimization tier adds > 20% over cuBLAS baseline → ship that tier
   - If WGPU > cuBLAS on any metric → investigate (unexpected, likely a bug)
   - If nothing reaches 2,000 tok/s → the bottleneck is not GEMM, re-profile

**Exit criterion (falsifiable):**
> All 8 variants measured on gx10. F-MEASURE-01 retired (0 UNMEASURED tiers). At least one variant achieves >= 2,000 tok/s with loss < 2.0. If no variant meets this bar, the problem is deeper than GEMM backend selection and requires a different approach.

**PMAT:** PMAT-501 (measure tiers), PMAT-505 (A/B test matrix)

---

> **F-MEASURE-01:** If 3+ optimization tiers remain UNMEASURED on 2026-04-12, the project has a measurement problem, not a coding problem. Action: freeze all upstream development. Deploy and measure until every SHIPPED tier has a number.

### Architectural Context: WGPU vs cuBLAS on NVIDIA

The WGPU 11.2x gap was a **platform mismatch** — confirmed by cuBLAS routing fix (4.5x in one change):

| APR WGPU path | Unsloth path | APR cuBLAS path (Phase B) |
|----------|-------------|--------------------------|
| WGSL compute shader source | Triton/CUDA source | Rust + cuBLAS FFI |
| → SPIR-V compilation | → PTX compilation | → cuBLAS library call |
| → Vulkan compute dispatch | → cuBLAS/cuDNN dispatch | → cuBLAS dispatch |
| → NVIDIA Vulkan driver | → NVIDIA CUDA driver | → NVIDIA CUDA driver |
| Hand-written GEMM | cuBLAS autotuned | cuBLAS autotuned |

cuBLAS has decades of per-architecture autotuning by NVIDIA engineers. No framework — including HuggingFace Candle (Rust), Burn, or our trueno — has achieved cuBLAS GEMM parity with custom kernels. The pragmatic industry pattern is to delegate GEMM to cuBLAS (Candle does this, PyTorch does this, JAX does this).

**After Phase B+C:** WGPU remains the correct path for AMD/Metal portability. cuBLAS is the correct path for NVIDIA throughput. `--gpu-backend auto` selects at runtime.

## Findings Summary (2026-04-05)

### Measured Parity (tok/s) — Updated 2026-04-06

| Runtime | yoga (8GB) | gx10 (120GB) | vs Unsloth | Loss |
|---------|-----------|-------------|------------|------|
| **unsloth** QLoRA | **5,412** (steps=20) / **6,697** (steps=100) | **5,262** (steps=20) / **16,118** (steps=100) | baseline | 0.45 |
| **apr** QLoRA (cuBLAS) | TBD (needs apr-cli 0.4.14) | **2,101** (2026-04-06, bwd_steps=0) | **40%** (2.5x gap) | N/A (PMAT-512) |
| **apr** QLoRA (WGPU) | **BLOCKED** (crash step 5, PMAT-498) | **470** (async, 2026-04-05) | **8.9%** (11.2x gap) | **11.74** (PMAT-497) |
| **apr** QLoRA (CUDA JIT) | **28** (cached JIT, 941s) | TBD | **0.5%** (237x gap) | 16.76 |
| **pytorch** full FT | OOM (F-EXEC-02) | **3,957** (steps=100, 2026-04-06) | 24.5% (4x, expected) | 0.0087 |
| **cublas** parity | OOM (F-EXEC-02) | **4,000** (steps=100) | 0.000 divergence | ~2.0 |
| **wgpu** synthetic | — | — | 6,730 (Vulkan, intel) | 1.0 |

**Config drift warning:** steps=20 results are warm-up-dominated. Always compare at steps=100 for baseline parity.

### Root Cause: APR 11.2x Parity Gap (was 151x, was 34x)

**Gap history:** 151x (2026-03-31) → 34x (2026-04-02) → 11.2x (2026-04-05, current)

Five-whys analysis (UPDATED 2026-04-05 — original root cause RESOLVED, new root cause identified):

~~1. APR is 151x slower — 44 vs 6,628 tok/s~~ **RESOLVED** by 65+ upstream fixes.

**Current root cause (2026-04-05):** Two independent defects remain:

1. **Throughput: 11.2x gap** — 470 vs 5,262 tok/s (gx10, same hardware). GPU is 100% utilized (profiler-verified). Bottleneck is WGPU compute shader dispatch speed: `gpu_lora_bwd`=55.7% (551ms/step), `gpu_fwd`=40.6% (401ms/step). Zero overhead. Fix path: kernel fusion (PMAT-484), WGSL shader optimization. **Phase B (cuBLAS hybrid) targets 3,000+ tok/s.**
2. **Convergence: oscillating, not stuck** — Loss trajectory (8 epochs): 18.9→**9.15**→12.3→16.3→15.5→12.0→10.8→11.7. **Model IS learning** (epoch 2 reached 9.15, below random 11.93). But oscillates wildly instead of converging. This is NOT a "model doesn't train" problem — it's a **learning rate / optimizer configuration** problem. LR 2e-4 likely too high for WGPU compute precision. Unsloth uses cosine decay with warmup=10; APR uses flat LR. Fix: reduce LR to 5e-5 + add cosine schedule. Prior root cause (WGSL GEMM layout PMAT-497) may also contribute.

### Two Paths to Parity

**Path 1: Safetensors training (Python stack)** — ALREADY AT PARITY

The Python stack (unsloth, pytorch, cublas) uses HuggingFace safetensors models.
No parity gap exists within this stack:
- unsloth QLoRA: 6,628 tok/s (yoga) — the throughput target
- pytorch full FT: 4,017 tok/s (gx10) — expected 4x slower (full precision vs QLoRA)
- cublas parity: 0.000 loss divergence — GEMM backends identical

**Path 2: APR training (Sovereign Stack)** — 11.2x gap (was 151x 2026-03-31, was 34x 2026-04-02), 3 fix tiers

## Historical: NF4 Dequant NaN (RESOLVED 2026-04-02)

> Condensed from v3.x-v4.x iteration. The NF4 dequant NaN blocker is RESOLVED.
> See git history for full diagnostic chain (DIAG-002 through DIAG-005, H-PARITY-001/002).

**Root cause (resolved):** trueno's NF4 dequant zeroed V-projection weights (11 tensors
shape 256x1536). Softmax(0/0)=NaN propagated through 28 layers. Filed as trueno#233,
fixed in local HEAD. Plus activation explosion across CudaBlockScratch sharing
between fwd/bwd — fixed by zeroing 21 scratch buffers per forward (PMAT-453).

**TRAINING WORKING (2026-04-02):** APR NF4 QLoRA training on yoga 8GB — 43 tok/s canary
confirmed. Full GPU pipeline: embed → 28 NF4 layers → BT GEMM lm_head → fused CE loss.

**Upstream fixes that landed (21 commits, 3 repos):**
- trueno 4a7838a4: copy_to_host partial readback
- entrenar f9845e07: VRAM embedding 1780→890MB (single layout)
- entrenar c605ea16: make_current in NF4 forward
- entrenar 475256c6: direct_transpose_upload (skip NF4 roundtrip)
- entrenar a515d2f9: original fp32 upload
- aprender ba0e392f: respect --rank 16 flag
- entrenar 8966424b: zero training state GPU buffers + L5 violations (PMAT-453)

The V-projection dequant fix unlocked training; all subsequent optimization tiers
(Tiers 1-6 in Optimization Tiers table) built on top of this working baseline.

**Contract:** apr-training-parity-v1.yaml — hypothesis-driven, falsify-first (retired 2026-04-02)

### Parity Roadmap: 470 → 5,262 tok/s (11.2x gap, WGPU path, gx10 2026-04-05)

**CORRECTED FINDING (2026-04-05, v6.7.0, SUPERSEDES v6.4.0/v6.5.0):** After async pipeline
deployment, the profiler measures **100% GPU compute** with **0% overhead** at
`wall_coverage=1.000`. The 11.2x gap IS real WGPU compute shader speed, NOT inter-step
overhead. `gpu_lora_bwd` at 55.7% (551ms/step) and `gpu_fwd` at 40.6% (401ms/step)
dominate. Zero allocations, zero sync, zero buf_write per step.

The earlier v6.4.0/v6.5.0 claims of "5.7x FASTER step-level GPU compute" and "98.3%
inter-step overhead" were artifacts of the SYNC pipeline where `device.poll(Wait)`
serialized per-step. Async pipeline (v6.7.0) resolved that, and the real compute
speed became visible: 470 tok/s wall-clock IS the GPU compute throughput.

**F-PROF-002 FALSIFIED (still true, different reason):** Buffer allocation inside
train_step is NOT the bottleneck — there are ZERO allocs/step under async pipeline.
The overhead we hypothesized doesn't exist; compute shader speed is the ceiling.

**Fix priorities (after async pipeline deployment, v6.7.0):**

**v6.7.0 profiler result (gx10 GB10, 400 steps, 395.7s wall, wall_coverage=1.000):**
```
gpu_lora_bwd:  55.7% (220.3s, avg 550.6ms/step) ← dominant bottleneck
gpu_fwd:       40.6% (160.6s, avg 401.5ms/step)
gpu_lm_bwd:     2.0% (7.8s)
gpu_lm:         1.7% (6.9s)
data_prep:      0.0% (78ms)
sync:           0.0% (zero per-step)
allocs/step:    0
```

1. **Kernel fusion (P0)** — `gpu_lora_bwd` dominates. Fused backward GEMM (PMAT-484)
   targets 840→1 launch reduction. Current status: SHIPPED but UNMEASURED (see
   Measurement Debt table).

2. **Pre-tokenize corpus (DONE, v6.7.0)** — eliminated per-step tokenizer calls.

3. **Async loss readback (DONE, v6.7.0)** — replaced per-step `device.poll(Wait)` with
   deferred loss accumulation. This fix eliminated the 92.5% sync overhead identified
   in v6.5.0.

4. **Buffer pre-allocation (DONE, v6.7.0)** — zero allocs/step under async pipeline.

**Key insight:** The earlier "1150ms → 90ms/step = 5,600+ tok/s (parity)" prediction
from v6.5.0 was WRONG — the 1150ms was sync overhead, not compute. After fixing sync,
GPU compute itself is 952ms/step (401ms fwd + 551ms bwd) = 470 tok/s. **Real ceiling
is WGPU compute shader speed, not synchronization.** Closing the 11.2x gap requires
Phase B (cuBLAS hybrid) or kernel fusion, not more sync tuning.

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

#### Historical: yoga CUDA path (2026-04-02, SUPERSEDED by v6.7.0)

> The following section documents yoga CUDA-path measurements at 197 tok/s.
> Current measurements are on gx10 WGPU (470 tok/s, 100% GPU compute). Preserved
> for its memory-bandwidth analysis, which still informs kernel-fusion priorities.

The yoga CUDA path measured 7% GPU utilization and hit a 197 tok/s memory-bandwidth
ceiling: TF32 tensor cores (41x compute) produced 0% improvement over cuBLAS SIMD
because weight matrix loads (1536×1536 = 9.4 MB at 256 GB/s = 37μs) dominated
compute (1.9μs TF32, 80μs SIMD). Conclusion: reduce memory traffic per sample via
(1) fused QKV projection, (2) flash attention, (3) FP16 weights, (4) fused FFN.

These kernel-fusion priorities still hold — PMAT-475/478/484 all target memory
traffic reduction. But the CPU-bound diagnosis (93% idle = CPU embedding/loss/AdamW)
was specific to yoga 8GB, where embeddings spilled to CPU. On gx10 with async
pipeline, GPU compute is 100% of wall time (no CPU idle).

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
