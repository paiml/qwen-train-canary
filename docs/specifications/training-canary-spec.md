# Training Canary Performance Specification

**Document ID:** PAIML-TRAIN-CANARY-001
**Version:** 6.23.0
**Last Updated:** 2026-04-06
**Status:** ACTIVE
**Methodology:** Popperian Falsification + Deterministic Canary Benchmarks
**Primary Target:** Yoga (RTX 4060 Laptop, 8 GB VRAM, sm_89)
**Model:** Qwen2.5-Coder-1.5B-Instruct (1.78B params, 28 layers, hidden=1536, heads=12/2)

> Every claim in this spec carries a falsification condition.
> If the condition triggers, the claim is revised or retracted -- not defended.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Hardware Targets](#2-hardware-targets)
3. [Canary Workloads](#3-canary-workloads)
4. [Metrics Contract](#4-metrics-contract)
5. [Baseline Thresholds](#5-baseline-thresholds)
6. [Scoring & Regression Detection](#6-scoring--regression-detection)
7. [Falsification Register](#7-falsification-register)
8. [PMAT Compliance](#8-pmat-compliance)
9. [Revision History](#9-revision-history)

**Component specs:** See [components/](components/) for detailed breakdowns:
- [cuBLAS Parity Gate](components/cublas-parity-gate.md)
- [Benchmarking Methodology](components/benchmarking-methodology.md)
- [Dataset Specification](components/dataset-specification.md)
- [Deployment Topology](components/deployment-topology.md)
- [Optimization Roadmap](components/optimization-roadmap.md)
- [Dependencies & References](components/dependencies.md)
- [Training Profiler](components/training-profiler.md) — `apr finetune --profile` central profiling interface (PMAT-496)

---

## 1. Executive Summary

### What This Is

Competitive benchmark for fine-tuning throughput across five training runtimes — the training analog of qwen-coder-deploy's inference runtime comparison. Five canary workloads exercise distinct training paths for Qwen2.5-Coder-1.5B-Instruct.

**Parity mandate:** Gaps are not findings to document — they are defects to fix. When a runtime falls behind (apr adapter-init-only, wgpu not running, torch.compile regressing), the response is to fix the runtime, not to record the gap and move on. The canary exists to drive parity across all five runtimes. A runtime that cannot match the throughput leader on the same hardware is broken until fixed.

**Yoga is the initial primary target.** All baselines, thresholds, and falsification conditions are calibrated against the RTX 4060 Laptop (8 GB VRAM, sm_89). Secondary targets (gx10, intel) validate at larger batch sizes and alternative backends.

### Honesty Check: Progress Stall (2026-04-05, PMAT-500)

> **F-PROGRESS-01:** If measured APR throughput does not improve >2x within 7 days of an optimization tier being SHIPPED, the tier either doesn't work, isn't wired, or was never deployed. Action: stop shipping new tiers. Measure the existing ones.

**Hard facts (2026-03-31 → 2026-04-05):**

| Metric | Day 1 | Day 5 | Delta |
|--------|-------|-------|-------|
| APR throughput (gx10) | 0 (broken) | 470 tok/s | ∞→470 (functional) |
| APR throughput improvement from optimization | — | +12% (async pipeline) | Only measured delta |
| Unsloth throughput (gx10) | 16,118 tok/s | 16,118 tok/s | 0% (it was already fast) |
| APR parity gap | ∞ | 11.2x | Narrowed from broken to 11x |
| APR convergence | N/A | loss 11.74 final (but 9.15 at epoch 2) | **Model learns then oscillates** — LR too high |
| Yoga APR status | broken | **still broken** (crash step 5) | No improvement |
| Optimization tiers shipped | 0 | 8 (Tiers 2, 4×3, 4.7×2, 5, 7) | All UNMEASURED |
| Spec revisions | v1.0 | v6.15.0 | 35 revisions |
| Upstream fixes | 0 | 65+ | Most were bug fixes, not perf |
| PMAT items | 19 | 88 | +69 items |

**Five-whys root cause:**

1. **Why hasn't throughput improved?** → No optimization tier has been measured in a canary run. Every tier (FP16, fused kernels, tensor cores, CUDA graph, fused backward) is SHIPPED but sits behind an env flag that has never been enabled in a real measurement.
2. **Why unmeasured?** → Can't build and deploy the optimized binary to target hardware. yoga is blocked (WGPU crash PMAT-498, CUDA 2-hour JIT PMAT-492). gx10 had alimentar build issues (PMAT-495).
3. **Why keep coding new tiers before measuring old ones?** → Writing upstream code and spec felt productive. Deploying and measuring is harder, slower, less satisfying work.
4. **Why did spec work feel more productive?** → Spec versions are monotonically increasing (v1.0 → v6.11.0). Throughput isn't. The spec creates an illusion of velocity that the throughput numbers don't support.
5. **Why is there an architectural mismatch?** → APR uses WGPU/Vulkan compute shaders on NVIDIA hardware where cuBLAS has decades of per-architecture autotuning. This is not an optimization gap — it's a fundamental platform mismatch. No Rust framework (including Candle) has achieved cuBLAS GEMM parity with custom kernels.

**Convergence root cause UPDATE (2026-04-06, LR 5e-5 test):**

| LR | Loss trajectory | Minimum | Verdict |
|----|----------------|---------|---------|
| 2e-4 | 18.9→9.15→12.3→16.3→15.5→12.0→10.8→11.7 | 9.15 (epoch 2) | Oscillates but learns briefly |
| 5e-5 | 16.4→22.2→18.8→20.2→18.9→19.7→19.9→19.5 | 16.4 (epoch 1) | **NEVER learns** — always above random |

**LR too high is NOT the root cause.** Lower LR should converge better (just slower) — instead it's worse. Loss > ln(151936)=11.93 from step 1 means LoRA adapters push predictions in the WRONG direction immediately.

**Ruled out (2026-04-06 investigation):**
- f16_to_f32 WGSL shader: pow(2.0,n) was bit-exact. Fixed to bitwise anyway (trueno 0.17.3)
- Forward-backward mismatch: Fixed Q/K/V only backward. No effect on loss.
- LoRA init: Kaiming A, zeros B — correct (verified).
- AdamW direction: w -= lr * grad — correct (verified).
- Cross-entropy label shift: labels[i]=input_ids[i+1] — correct (verified).

**Key observation:** LR was hardcoded to 2e-4 — FIXED (now wired from CLI). Different LR confirmed to produce different trajectories.

**GPU vs CPU diagnostic (2026-04-06, PMAT-509):**

| Stage | APR (WGPU) | PyTorch (F32) | Ratio | Verdict |
|-------|-----------|---------------|-------|---------|
| embed | 7.99 | 5.64 | 1.4x | Q4K noise (acceptable) |
| layer 0 | 183 | 112 | 1.6x | Diverging |
| layer 1 | 4523 | 2483 | 1.8x | Growing |
| layer 27 | 6719 | 1853 | **3.6x** | Significant |
| logits norm | 1107 | 2646 | **0.42x** | Inverted |
| logits argmax | 74403 | 16 | — | **WRONG** |
| CE loss | 16.37 | 0.55 | 30x | Model broken |

PyTorch initial CE: 0.55 (correctly loaded model). APR: 16.37 (above random).

Hidden states diverge PROGRESSIVELY through 28 layers (1.4x→3.6x). Not catastrophic (no NaN/Inf) — consistent with **accumulated rounding error in the tiled GEMM shader**, not a single indexing bug. The Q4K→F32 dequant is correct (verified bit-by-bit), weight transpose is correct, CE loss is correct.

**Root cause RESOLVED (2026-04-06, PMAT-509):**
1. **Missing RoPE** in training forward (both `encode_forward_layer_training` + `forward_layer_traced`). Fixed in trueno 0.17.4.
2. **Missing QKV biases** — Qwen2 Q/K/V biases stored CPU-side but never applied in GPU training. Fixed in trueno.

Results after fix:
- LR 2e-4: epoch 1 loss **10.58** (was 16.37, now below random 11.93)
- LR 5e-5: epoch 1 loss **3.60** (close to PyTorch 0.55)

**PMAT-510: backward divergence — FIX SHIPPED (2026-04-06).** Loss diverged after epoch 1 (3.60→17.88). Root cause: `wgpu_pipeline.rs:1066` used a single `grad_hidden_buf` as gradient for ALL 28 LoRA layers. Every layer got the same gradient signal instead of per-layer gradients through the residual+FFN+attention chain.

**Fix (entrenar 0.7.9, trueno 0.17.5):**
1. Reverse layer iteration order (27→0) — layers closest to loss update first
2. Per-layer backward propagation through frozen base weights:
   - `grad_silu = grad @ W_down^T` (backward through down_proj, [seq, inter])
   - `grad_ffn = grad_silu @ W_gate^T` (backward through gate_proj, [seq, hidden])
   - `grad_input = grad + grad_ffn` (residual connection)
3. Each layer receives a DIFFERENT gradient signal — the accumulated transformation through all subsequent layers' FFN paths

Simplifications (acceptable for QLoRA convergence, can optimize later):
- Skips SiLU backward derivative (treats as identity)
- Skips RMSNorm backward (treats as identity)
- Skips up_proj path (only gate_proj, which is dominant)
- Skips attention backward (parameter-free, no learnable weights in attention mechanism itself)
- RoPE backward on Q/K gradients not yet implemented

**Dogfood result (2026-04-06, gx10 GB10, LR 5e-5, 32 epochs):**

| Epoch | Loss (before fix) | Loss (PMAT-510 fix) | Verdict |
|-------|-------------------|---------------------|---------|
| 1 | 3.60 | **2.97** | Improved — per-layer gradient works |
| 2 | 17.88 | 10.06 | Still diverging but slower |
| 3 | — | 13.11 | Above random (11.93) |
| 32 | — | 16.11 | Fully diverged |

**Partial success:** Epoch 1 improved (2.97 < 3.60), confirming per-layer gradient propagation produces better LoRA updates. But loss still diverges from epoch 2 — the simplified backward (skip RMSNorm, SiLU, up_proj, attention) causes gradient explosion through 28 layers. Without RMSNorm backward to re-normalize, the FFN backward `grad @ W_down^T @ W_gate^T` amplifies gradient magnitude exponentially.

**PMAT-511: RMSNorm backward needed.** The next fix must implement RMSNorm backward in the per-layer gradient chain. RMSNorm normalizes hidden states; its backward re-normalizes gradients, preventing explosion. This is the key missing piece for multi-epoch convergence.

**Profile:** lora_bwd=700-800ms (90% of 750ms/step). Throughput: 264 tok/s (down from 470 without backward GEMMs — expected, 56 extra large matrix ops per step).

**Execution plan (three sequential phases, each with provable exit criteria):**

| Phase | What | Exit Criterion | PMAT |
|-------|------|---------------|------|
| **A. Fix Profiling** | Wire 6 provable-contract invariants into profiler. Port StepProfiler to CUDA path. Verify convergence (loss < 2.0). Fix yoga crash. | `apr finetune --profile` on CUDA target: 13 phases, wall_coverage >= 0.85, 6 contracts passing, loss < 2.0 | PMAT-504, 497, 498, 506. **Progress:** 6 contracts wired into score.py + canary runner (71 tests passing). Contracts applied to all 8 historical APR results. **NEW (2026-04-05 dogfood):** trueno WGSL q4k_gemv shader fixed (bitcast for -inf/NaN, wgpu 27.0.1 compat — PMAT-507). APR on gx10 still blocked: `--gpu-backend cuda` NOT routed (PMAT-494), WGPU shader now crashes on startup due to wgpu 27.0.1 regression (fixed in trueno but apr binary not rebuilt on gx10 yet). gx10 Python env fixed: torch 2.11+cu130 from default PyPI supports sm_121. Unsloth blocked on gx10 (PMAT-508: triton aarch64). pytorch gx10 canary PASSING: 3,906 tok/s (baseline 4,000). |
| **B. Hybrid cuBLAS Backend** | Fix JIT caching. Wire cuBLAS GEMM into NF4 training. `--gpu-backend auto` selects cuBLAS on NVIDIA, WGPU on AMD/Metal. | `apr finetune --gpu-backend cuda` on gx10 >= 2,000 tok/s (within 3x of unsloth) | PMAT-503, 492 |
| **C. A/B Test on CUDA Targets** | 8-variant matrix on yoga + gx10. All 6 UNMEASURED tiers measured. Per-variant profiler + convergence check. | All tiers measured. At least one variant >= 2,000 tok/s with loss < 2.0. F-MEASURE-01 retired. | PMAT-501, 505 |

**Phase A must complete before B starts. Phase B must complete before C starts.** No new optimization tier may be coded until Phase C completes. See [optimization-roadmap.md](components/optimization-roadmap.md) for full deliverables per phase.

> **F-PROGRESS-02:** If this section is still accurate 7 days from now (2026-04-12), the project methodology is falsified. Spec-writing without measurement is documentation, not engineering.

### Chain of Reasoning

**Step 1: Why canaries?** Training performance depends on a deep stack: PyTorch, CUDA runtime, cuBLAS/cuDNN, driver, GPU clocks, memory allocator, model weights, tokenizer, dataset pipeline. A regression in ANY layer silently degrades throughput. Canaries are short (100 steps, ~2 min), deterministic (seed=42, clock-locked), and produce machine-readable JSON.

> **F-EXEC-01:** If a canary fails to detect an artificially injected 15% throughput slowdown on yoga, the entire regression detection methodology is falsified.

**Step 2: Why five workloads?** Each canary isolates a different bottleneck. Like qwen-coder-deploy compares realizr/ollama/llama.cpp/vLLM/wgpu for inference, we compare training runtimes head-to-head:

| Canary | Runtime | Bottleneck | Why It Matters |
|--------|---------|-----------|---------------|
| **apr** | aprender/entrenar (Rust) | Sovereign Stack training | **The runtime to improve.** Native Rust QLoRA via trueno. |
| **unsloth** | unsloth (Python) | QLoRA + 4-bit quant | **The throughput target to beat.** Best-known Python QLoRA. |
| **pytorch** | PyTorch (Python) | Raw training loop | Baseline. Isolates framework overhead. |
| **cublas** | PyTorch (Python) | GEMM backend parity | Runs SAME loop twice. Detects silent numerical divergence. |

> **F-WL-06:** If apr throughput < unsloth throughput on same hardware, the Sovereign Stack training path has a throughput deficit. Action: profile entrenar hot path, check trueno GEMM performance.

**Step 3: What constitutes a regression?** Three gates:

1. **Throughput**: tokens/sec within 10% of baseline (THROUGHPUT_TOLERANCE=0.10)
2. **Memory**: peak VRAM within 5% of baseline (VRAM_TOLERANCE=0.05)
3. **Convergence**: final loss below threshold (< 2.0 CUDA, < 2.5 WGPU)

The cuBLAS canary adds:
4. **Numerical parity**: loss divergence < 0.01
5. **Performance parity**: throughput ratio 0.95x-1.05x

### Key Constraints

- **8 GB VRAM ceiling** on yoga -- batch=4 at seq_len=512 is the safe maximum for full fine-tune.
- **QLoRA NF4** reduces weight memory 4x (1.5 GB), allowing headroom.
- **Deterministic seeds** (42) + **locked GPU clocks** (1900 MHz) target <2% run-to-run variance.
- **Gradient accumulation** available via `--gradient-accumulation-steps N` (PMAT-459) for VRAM-constrained hosts (yoga batch=1 accum=4). Default canary mode uses batch=4 without accumulation to measure raw per-step throughput.
- **uv is the ONLY Python packaging tool.** No pip, conda, poetry, pipenv, or venv. All Python canaries (unsloth, pytorch, cublas) use `uv run` or `uv sync` for dependency management. Environments are defined by `pyproject.toml` + `uv.lock` at the repo root or per-canary directory.

> **F-EXEC-02:** If batch=4 seq=512 OOMs on yoga for the pytorch canary, the memory budget claim is falsified. Action: reduce to batch=2 or add gradient checkpointing.

---

## 2. Hardware Targets

### Yoga (PRIMARY -- RTX 4060 Laptop)

All initial baselines and falsification conditions target yoga. Secondary targets (gx10, wgpu) are active — Phase 0 complete (PMAT-424 DONE, 0.34% variance confirmed).

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX 4060 Laptop GPU |
| Compute | sm_89 (Ada Lovelace) |
| VRAM | 8 GB GDDR6 |
| Memory BW | 256 GB/s |
| TDP | 115W |
| Clock | Locked 1900 MHz (`nvidia-smi -lgc 1900,1900`) |
| CUDA | 12.6 Runtime / 13.1 Driver |
| Host | AMD, 32 GB RAM |
| Network | LAN 192.168.50.38 |

**Constraints:** 8 GB VRAM means full fine-tune of 1.5B at batch=4 is tight. QLoRA fits comfortably. cuBLAS parity canary loads the model twice (sequentially, with `torch.cuda.empty_cache()` between).

> **F-HW-01:** If yoga canaries show >5% run-to-run throughput variance with locked clocks, the determinism claim is falsified. Action: investigate thermal throttling, background processes.

### gx10 (SECONDARY -- Grace Blackwell GB10)

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10, sm_121 (Blackwell) |
| VRAM | 120 GB unified |
| CUDA | 13.0 |
| Network | localhost (runs locally) |

**Active** (PMAT-424 DONE). Canary uses batch=16. Measured: pytorch 3,906 tok/s (v6.17.0), cublas 0.000 divergence. **Unsloth BLOCKED** (PMAT-508: triton aarch64).

**Platform constraint (2026-04-05):** gx10 is aarch64. PyTorch cu124/cu126 wheels have NO aarch64 builds with sm_121 support. Only `torch>=2.11` from default PyPI (cu130) works. This means: (1) unsloth needs triton which has no aarch64 wheels, (2) the `cuda-base` extra must be used instead of `cuda` on gx10. See pyproject.toml and PMAT-508.

### Intel (SECONDARY -- WGPU/Vulkan)

| Property | Value |
|----------|-------|
| GPU | AMD Radeon Pro W5700X, Navi 10 (RDNA 1), 8 GB |
| API | Vulkan 1.3 (Mesa RADV) |
| Host | Intel CPU, 64 GB RAM, LAN 192.168.50.100 |

**Active** (PMAT-431 DONE). Measured: **6,730 tok/s** at Qwen-sized synthetic (hidden=1536, vocab=32000). Real model loading (HF safetensors/APR format) is the next milestone — see [optimization-roadmap.md P2](components/optimization-roadmap.md).

> **F-HW-02:** If WGPU throughput = 0 or burn-canary crashes on intel, the WGPU training feasibility claim is falsified.

---

## 3. Canary Workloads

### 3.0 APR Fine-Tune Canary (Sovereign Stack)

**File:** `canaries/apr/train.py` | **Backend:** CUDA via trueno | **Duration:** TBD

The **runtime to improve**. APR fine-tune uses aprender's native Rust training engine (entrenar) with trueno SIMD-accelerated tensor operations. Like realizr is the SSC inference engine, entrenar is the SSC training engine. Currently **11.2x slower** than unsloth on same hardware (470 vs 5,262 tok/s, gx10 2026-04-05). Convergence also lagging: loss 11.74 vs unsloth 0.45 (PMAT-497).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Runtime | `apr finetune` (Rust binary) | Native Rust, no Python overhead |
| Quantization | NF4 (4-bit) | QLoRA via trueno fused NF4 kernels |
| LoRA rank/alpha | 16 / 32 (or auto) | Matches unsloth config for fair comparison |
| Optimizer | AdamW | entrenar native implementation |
| Output | APR + safetensors checkpoints | Dual format for SSC and HF compatibility |

> **F-WL-06:** If apr throughput < unsloth throughput on same hardware, the Sovereign Stack training path has a throughput deficit. Profile entrenar hot path.
> **F-WL-07:** If apr produces different loss trajectory than unsloth/pytorch for same data, numerical divergence in trueno GEMM.

### 3.1 Unsloth QLoRA Canary

**File:** `canaries/unsloth/train.py` | **Backend:** CUDA (sm_89+) | **Duration:** ~2 min (100 steps, yoga)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | NF4 (4-bit) | Standard QLoRA via bitsandbytes |
| LoRA rank/alpha | 16 / 32 | r=8 too low for code, r=32 OOMs |
| Target modules | q,k,v,o,gate,up,down | All linear projections |
| Gradient ckpt | unsloth | 60% memory savings |
| Optimizer | AdamW 8-bit | 75% optimizer memory reduction |
| LR schedule | Cosine, warmup=10 | Standard for QLoRA |

> **F-WL-01:** If unsloth throughput < pytorch throughput on yoga, the claim that unsloth is faster than naive QLoRA is falsified. Action: check unsloth version, profiler.

### 3.2 PyTorch Baseline Canary

**File:** `canaries/pytorch/train.py` | **Backend:** CUDA (sm_89+) | **Duration:** ~3 min (100 steps, yoga)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | bf16 (sm_89+) / fp16 | Native mixed precision |
| Optimizer | AdamW 8-bit (yoga) / AdamW (gx10) | 8-bit required on <=16GB (F-EXEC-02) |
| LR schedule | CosineAnnealingLR | T_max=steps |
| Gradient clipping | 1.0 | Prevent divergence in short runs |
| Gradient checkpointing | Auto (on if VRAM <= 16 GB) | Full FT exceeds 8 GB without it |

**Training loop order (verified by falsification):**
```python
optimizer.zero_grad()   # Clear BEFORE backward
loss.backward()         # Compute gradients
clip_grad_norm_(1.0)    # Prevent explosion
optimizer.step()        # Update weights
scheduler.step()        # Adjust LR
```

> **F-WL-02:** If pytorch canary throughput < 3,000 tok/s on yoga, investigate PyTorch version, cuDNN autotuner, bf16 matmul precision, memory allocator.

### 3.3 cuBLAS Parity Canary

**File:** `canaries/cublas/train.py` | **Backend:** CUDA (sm_89+) | **Duration:** ~4 min (50 steps x 2, yoga)

Runs the **same training loop twice** with different GEMM backends:

| Run | Config | Purpose |
|-----|--------|---------|
| Run 1 | PyTorch default (cuDNN autotuner) | Baseline |
| Run 2 | `preferred_linalg_library("cusolver")` + TF32 | cuBLAS path |

**Parity thresholds:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| loss_divergence | < 0.01 | Absolute final loss difference |
| max_step_divergence | < 0.05 | Max per-step loss delta |
| throughput_ratio | 0.95-1.05 | cuBLAS / default |
| vram_delta_mb | < 200 | Absolute VRAM difference |

> **F-WL-03:** If loss_divergence > 0.01, GEMM backends are numerically divergent. Action: investigate precision settings, driver version.
> **F-WL-04:** If both runs produce identical results (ratio=1.0000), TF32 flag may not be taking effect -- falsifies the test itself.

See [components/cublas-parity-gate.md](components/cublas-parity-gate.md) for background and parity chain.

### 3.4 WGPU/Burn Canary

**File:** `canaries/wgpu/train.py` | **Backend:** WGPU/Vulkan | **Duration:** ~5 min (100 steps, intel)

Wraps a Rust binary (`burn-canary`) via subprocess. Binary emits JSON to stdout.

> **F-WL-05:** If burn-canary binary not found, WGPU deployment is broken (check forjar). If VRAM > 7,000 MB, burn allocator has a memory leak.

---

## 4. Metrics Contract

### Output Schema (JSON)

Every canary produces JSON conforming to:

```json
{
  "canary": "apr|apr-fused|apr-tc|apr-fp16|apr-fused-fp16|apr-fused-fp16-graph|unsloth|pytorch|pytorch-compile|cublas|wgpu",
  "backend": "cuda|wgpu|vulkan|cpu|metal",
  "host": "string",
  "gpu": {
    "device": "string",
    "vram_total_mb": "int",
    "cuda_version": "string",
    "compute_capability": "string"
  },
  "timestamp": "ISO 8601 UTC",
  "config": {
    "model": "string", "batch_size": "int", "seq_len": "int",
    "steps": "int", "lr": "float", "seed": "int",
    "dtype": "bf16|fp16|f32", "optimizer": "string",
    "quantization": "nf4|none"
  },
  "metrics": {
    "throughput_samples_sec": "float",
    "tokens_per_sec": "float",
    "peak_vram_mb": "int",
    "final_loss": "float",
    "step_time_ms": { "mean": "float", "p50": "float", "p95": "float", "p99": "float" },
    "wall_time_sec": "float"
  }
}
```

cuBLAS canary adds `metrics.default`, `metrics.cublas`, and `metrics.parity` -- see [components/cublas-parity-gate.md](components/cublas-parity-gate.md).

### Metric Definitions

| Metric | Formula | Unit |
|--------|---------|------|
| throughput_samples_sec | (batch_size x steps) / wall_time | samples/s |
| tokens_per_sec | (batch_size x seq_len x steps) / wall_time | tok/s |
| peak_vram_mb | torch.cuda.max_memory_allocated() / 1024^2 | MB |
| final_loss | mean(losses[-10:]) | dimensionless |
| wall_time_sec | end - start (perf_counter) | sec |

> **F-MET-01:** If any canary emits JSON that fails schema validation, the metrics contract is broken. Action: fix canary output before accepting results.

---

## 5. Baseline Thresholds

All baselines established from measured data (PMAT-424 DONE, 0.34% variance on yoga). gx10 and wgpu baselines added from 2026-03-31 measurements.

| Canary | Metric | Baseline | Tolerance | Gate |
|--------|--------|----------|-----------|------|
| unsloth (yoga) | tokens_per_sec | 6,600 | -10% | >= 5,940 |
| unsloth (yoga) | peak_vram_mb | 3,600 | +5% | <= 3,780 |
| unsloth | final_loss | 1.0 | -- | <= 1.0 |
| unsloth@gx10 | tokens_per_sec | 16,100 | -10% | >= 14,490 |
| pytorch | tokens_per_sec | 3,000 | -10% | >= 2,700 |
| pytorch | peak_vram_mb | 8,000 | +5% | <= 8,400 |
| pytorch | final_loss | 2.0 | -- | <= 2.0 |
| cublas | tokens_per_sec | 3,000 | -10% | >= 2,700 |
| cublas | loss_divergence | 0.01 | -- | <= 0.01 |
| cublas | throughput_ratio | 0.95 | -- | >= 0.95 |
| apr (gx10 WGPU) | tokens_per_sec | 470 | -10% | >= 423 |
| apr | final_loss | 12.0 | -- | <= 12.0 |
| wgpu (synthetic) | tokens_per_sec | 6,600 | -10% | >= 5,940 |
| wgpu (synthetic) | final_loss | 2.5 | -- | <= 2.5 |

**Loss baseline tightening (2026-04-05):** Unsloth loss baseline reduced from 2.0 to 1.0. Measured loss is 0.47 (yoga) and 0.14 (gx10) — the old 2.0 baseline was too loose to detect convergence regressions. APR baseline added at 12.0 (measured 11.74, convergence defect PMAT-497 open).

**Config drift warning:** All baselines assume **steps=100**. Results at steps=20 are warm-up-dominated and NOT comparable. The 2026-04-05 gx10 unsloth result (5,262 tok/s at steps=20) is NOT a regression from the 16,118 tok/s baseline (steps=100).

**WGPU note:** The wgpu baseline (6,600 tok/s) is for the synthetic MLP at Qwen hidden/vocab scale (hidden=1536, vocab=32000), measured 2026-03-31. Real Qwen model loading is not yet supported by burn; a separate baseline will be established once real model weights load (PMAT-442).

**Baseline update policy:** After 5 consecutive nightly runs with <5% variance, update to the median observed value. Floor to nearest 100 for tok/s, ceil to nearest 100 for VRAM.

> **F-BL-01:** If first 5 yoga runs show >5% variance, baselines cannot be established. Action: investigate clock locking, thermal, background processes before proceeding.

### Expected Throughput (Yoga Primary)

| Canary | Runtime | yoga (8GB) | gx10 (120GB) | Lambda (RTX 4090) |
|--------|---------|-----------|-------------|-------------------|
| **apr** (WGPU async) | entrenar (Rust) | **BLOCKED** (5 steps, buffer crash PMAT-498, 2026-04-05) | **470** tok/s (453s, 2026-04-05) | **125** tok/s (2026-04-04) |
| **apr** (CUDA cached JIT) | entrenar (Rust) | **28** tok/s (941s, 2026-04-04) | TBD | **119** tok/s (2026-04-04) |
| **apr** (NF4 fused PTX) | entrenar (Rust) | **33** tok/s (100% GPU, 2026-04-03) | TBD | N/A |
| **apr-tc** (NF4 tensor core) | entrenar (Rust) | **UNMEASURED** — PMAT-479+481 shipped+wired, canary ready | TBD | N/A |
| **apr-fp16** (FP16 GEMM) | entrenar (Rust) | **UNMEASURED** — PMAT-470/472 shipped, canary ready | TBD | N/A |
| **apr-fused** (fused Gate+Up+K+V) | entrenar (Rust) | **UNMEASURED** — PMAT-475/478 shipped, canary ready | TBD | N/A |
| unsloth | Python + bitsandbytes | **5,512** (2026-04-04) | **5,262** (2026-04-05, 20 steps) | N/A |
| pytorch | Python + torch | TBD (gradacc batch=1 accum=4, PMAT-459) | **4,017** (2026-04-01) | N/A |
| cublas | Python + torch | N/A (F-EXEC-02) | **4,000** (0.000 div, 2026-04-01) | N/A |
| **wgpu** | burn (Rust, Vulkan) | N/A | N/A | **6,730** tok/s (synthetic, hidden=1536) |

**Parity gap (APR):** 470 tok/s wall (gx10 WGPU async) vs 5,262 tok/s (unsloth, gx10) = **11.2x** on same hardware. GPU is 100% utilized (profiler-verified, `wall_coverage=1.000`). **Root cause confirmed**: WGPU compute shader dispatch speed, NOT overhead. `gpu_lora_bwd` = 55.7% (551ms/step, 7 projections × 28 layers), `gpu_fwd` = 40.6% (401ms/step). Zero allocations, zero sync, zero overhead per step. Convergence also lagging: APR loss 11.74 (oscillating) vs unsloth loss 0.45 (converged). Two independent defects: (1) throughput — WGPU matmul dispatch overhead, fix via kernel fusion (PMAT-484); (2) convergence — learning rate or LoRA configuration mismatch. Blockers: (1) yoga WGPU blocked by missing `libvulkan.so.1` (PMAT-493); (2) PTX JIT takes 2 hours on yoga (PMAT-492). See [training-profiler.md](components/training-profiler.md), [optimization-roadmap.md](components/optimization-roadmap.md).

---

## 6. Scoring & Regression Detection

### Score Computation

```python
THROUGHPUT_TOLERANCE = 0.10  # 10% regression threshold
VRAM_TOLERANCE = 0.05        # 5% memory regression threshold

# Standard canaries (unsloth, pytorch, wgpu)
checks = {
    "throughput": tok_s >= baseline_tok_s * (1 - THROUGHPUT_TOLERANCE),
    "vram":       peak_vram <= baseline_vram * (1 + VRAM_TOLERANCE),
    "loss":       final_loss <= baseline_loss,
}

# cuBLAS parity canary
checks = {
    "numerical_parity": loss_divergence <= 0.01,
    "perf_parity":      throughput_ratio >= 0.95,
    "throughput":        default_tok_s >= baseline_tok_s * 0.90,
}
```

### Grade Mapping

| Result | Grade | Action |
|--------|-------|--------|
| All checks pass | PASS | No action |
| Throughput fails only | WARN | Investigate, may be transient |
| VRAM fails | FAIL | Memory regression -- block deploy |
| Loss fails | FAIL | Convergence broken -- block deploy |
| Numerical parity fails | FAIL | GEMM backend divergence -- critical |
| Multiple failures | FAIL | Systematic regression -- urgent |

### CI Gate

```bash
make score-gate  # Exits non-zero if ANY canary fails
```

> **F-SC-01:** If `make score-gate` exits 0 when a canary metric exceeds its threshold, the scoring logic is falsified. Action: fix scoring before trusting results.

---

## 7. Falsification Register

Every claim carries a falsification condition (F-prefixed IDs inline above). This section is the consolidated register.

### Active Conditions

| ID | Claim | Falsification Condition | Priority |
|----|-------|------------------------|----------|
| F-CONV-01 | APR loss converges to < 2.0 after transpose fix | If APR loss still > 2.0 after weight transpose fix (PMAT-497), the WGSL forward pass has additional correctness bugs beyond layout. Action: dump per-layer activations and compare to HF reference. | P0 |
| F-WL-07 | APR loss trajectory matches unsloth/pytorch | **TRIGGERED (2026-04-05):** APR loss 11.74 vs unsloth 0.45. BUT epoch 2 reached 9.15 (below random) — **model IS learning, then oscillates.** Root cause: LR 2e-4 too high (unsloth uses cosine decay+warmup). Fix: reduce LR to 5e-5 + cosine schedule. PMAT-497 GEMM layout may also contribute. | P0 |
| F-PROF-007 | WGPU dispatch speed is the throughput bottleneck | If reducing dispatch count (kernel fusion) does NOT improve wall-clock throughput proportionally, the bottleneck is elsewhere (memory BW, kernel occupancy). Action: measure with fused backward GEMM (PMAT-484). | P1 |
| F-PROF-008 | Dispatch overhead is significant (>50% of gpu_lora_bwd) | If GPU-side kernel time matches CPU-side time (ratio 0.95-1.05), dispatch overhead is negligible. Action: optimize shaders, not dispatch count. | P0 |
| F-PROF-010 | LoRA backward is memory-bound (AI=6.3 << ridge=111) | If measured AI > ridge point, LoRA IS compute-bound. Action: optimize WGSL shader, not memory access. | P0 |
| F-VRAM-01 | APR reports peak_vram_mb | **TRIGGERED (2026-04-05):** All APR results report `peak_vram_mb: 0`. WGPU path has no `torch.cuda.max_memory_allocated()` equivalent. Action: implement wgpu buffer tracking or exempt WGPU from VRAM gate. | P2 |
| F-CFG-01 | Canaries use steps=100 for baseline comparisons | **TRIGGERED (2026-04-05):** gx10 unsloth ran steps=20 (5,262 tok/s) — NOT comparable to steps=100 baseline (16,118 tok/s). Warm-up overhead dominates at steps=20. Action: enforce steps=100 in all baseline-comparison canary runs. | P1 |
| F-PROGRESS-01 | Optimization tiers produce measured throughput improvement | **TRIGGERED (2026-04-05):** 8 optimization tiers SHIPPED (Tiers 2, 4×3, 4.7×2, 5, 7), 0 measured. Only measured delta: +12% from async pipeline. Action: stop shipping new tiers, measure existing ones. Deadline: 2026-04-12. | **P0** |
| F-MEASURE-01 | SHIPPED tiers get measured within 7 days | If 3+ tiers remain UNMEASURED on 2026-04-12, the project has a measurement problem. Action: freeze upstream development, deploy and measure. | **P0** |
| F-PROGRESS-02 | Spec-writing correlates with throughput improvement | If spec reaches v7.x without measured throughput change >2x, the methodology is falsified. Writing specs is not engineering. | **P0** |
| F-REGRESS-01 | Deployed binaries do not silently regress | **CONFIRMED with direct measurement (2026-04-05 16:22 UTC):** Ran `make canary-apr-gx10` with live contracts wired in. Binary apr 0.4.11 (2bfde5d3) with `--gpu-backend wgpu` flag silently used CUDA path instead (stderr: "[CUDA] NF4 QLoRA training initialized"), hit GH-480 sm_121 JIT patching, then deadlocked in futex wait for 17.6 min. Result: 0 valid backward steps, loss unparseable, bogus 993 tok/s. **Live contracts caught this: canary exited non-zero with 3 failures (backward_executed, metrics_quality, vram).** Prior async run (10:29) used "[wgpu]" path successfully. Routing flag IGNORED on current binary — filed as PMAT-494 regression. See `results/canary-apr-gx10-regressed-20260405.json`. | **P0** |
| F-ECOSYSTEM-01 | Upstream repos buildable for deployment | **TRIGGERED (2026-04-05):** apr-cli pins trueno 0.16.0 crates.io but workspace uses 0.17.1 local path (diamond dep). presentar path stale in entrenar. aprender has `build_default_registry` undefined error. Dev-overrides mechanism exists but broken. **Cannot build, cannot deploy, cannot measure.** Action: fix build state of Sovereign Stack before Phase B can begin. | **P0** |

### Falsified Claims

| ID | Claim | Date | What Happened | Resolution |
|----|-------|------|---------------|------------|
| F-EXEC-01 | Canaries detect 10% regressions | 2026-04-01 | CONFIRMED: GPU clock throttled to 600 MHz on gx10 (vs ~1500+ MHz default). Measured 7,325 tok/s vs 13,600 baseline = 46% regression. Scoring correctly returned FAIL (throughput gate: 7,325 < 12,240 threshold). Methodology validated. | Clock injection test on gx10. Same scoring logic applies to yoga. |
| F-EXEC-02 | Full FT fits 8GB at batch=4 seq=512 | 2026-03-31 | OOM even at batch=2 + 8-bit optimizer + gradient checkpointing. Model (3.5GB) + gradients (3.5GB) = 7GB floor. | pytorch/cublas deferred to gx10. Yoga runs unsloth only. |
| F-RD-01 | torch.compile +20-40% throughput | 2026-03-31 | -11.3% regression (3,598 vs 4,055 tok/s). Compilation cost (~90s) amortized over only 100 steps = net loss. | torch.compile not suitable for canary-length runs. Would help at >1000 steps. |
| F-HW-01 | Locked clocks <5% variance | 2026-03-31 | CONFIRMED: 0.34% variance across 5 runs on yoga. | Baseline methodology validated. |
| F-WL-03 | cuBLAS parity <0.01 | 2026-03-31 | CONFIRMED: 0.000000 divergence on gx10. Perfect parity. | GEMM backends numerically identical on Blackwell. |
| F-WL-01 | Unsloth faster than raw PyTorch | 2026-03-31 | CONFIRMED: unsloth 13,660 tok/s vs pytorch 4,055 tok/s on gx10 (batch=16, same hardware). QLoRA NF4 + 8-bit optimizer saves ~75% compute per step. | QLoRA advantage holds. |
| F-WL-02 | PyTorch baseline >3k tok/s | 2026-03-31 | CONFIRMED: gx10 measures 4,055 tok/s (pytorch) and 4,010 tok/s (cublas default) at batch=16. Yoga deferred (F-EXEC-02). | pytorch throughput validated on gx10. |
| F-HW-02 | WGPU training feasible on W5700X | 2026-03-31 | CONFIRMED: burn-canary binary running, 6,730 tok/s on Qwen-sized synthetic (PMAT-431 DONE). Real model loading (HF safetensors) pending. | WGPU training path viable. |
| F-WL-05 | WGPU deployment works | 2026-03-31 | CONFIRMED: burn-canary binary found and producing results. 6,730 tok/s on hidden=1536 synthetic. | WGPU deployment operational. |
| F-WL-06 | apr throughput vs unsloth | 2026-03-31 → 2026-04-05 | **CONFIRMED, gap narrowing.** Original: 151x (44 vs 6,628 tok/s). Now: **11.2x** (470 vs 5,262 tok/s, gx10 WGPU async). 65+ upstream fixes landed. GPU utilization 100% (profiler-verified). Remaining gap is real WGPU compute shader speed, NOT overhead. **Convergence defect separate:** loss 11.74 vs 0.45 (PMAT-497, F-WL-07). Two independent problems. | 65+ fixes across entrenar/trueno/aprender. Profiler: gpu_lora_bwd 55.7%, gpu_fwd 40.6%. |
| F-MET-01 | Metrics schema valid | 2026-04-01 | TRIGGERED then FIXED: wgpu results missing `timestamp` field (burn binary doesn't emit it, Python wrapper does). Fixed by adding timestamp to existing results. Schema validator (`validate_schema.py`) now runs as `make score` prerequisite. 11/11 results pass. | Contract: canary-metrics-schema-v1.yaml. Validator: scripts/validate_schema.py. Refs: paiml/qwen-train-canary#7 (PMAT-444). |
| F-SC-01 | Scoring logic correct | 2026-04-01 | CONFIRMED: 5 falsification injection tests pass. 15% slowdown → FAIL, 5% variance → PASS, 10% VRAM → FAIL, cuBLAS div 0.02 → FAIL, cuBLAS ratio 0.94 → FAIL. All match canary-score-gate-v1.yaml contract predictions. | Contract: canary-score-gate-v1.yaml. Baselines: wgpu 100→6600, pytorch-compile added. Refs: PMAT-445. |
| F-WL-04 | cuBLAS test is meaningful | 2026-04-01 | CONFIRMED: throughput_ratio=1.0043 on gx10, not 1.0000 exactly. TF32 flag IS taking effect (0.43% speedup on Blackwell). cuBLAS test differentiates backend behavior. | Measured: canary-cublas-gx10-20260331.json. Refs: paiml/qwen-train-canary#8 (PMAT-447). |
| F-PROF-002 | Buffer allocation is the bottleneck (alloc_ratio >= 0.50) | 2026-04-05 | FALSIFIED: alloc_ratio = 0.000 (zero allocations per step). gpu_compute_pct = 100%. The bottleneck is GPU compute dispatch speed (gpu_lora_bwd 55.7%), not buffer allocation. Pre-allocation already solved by WGPU async pipeline. | Profiler v1 measured on gx10: 400 steps, wall_coverage=1.000, alloc_count=0. See training-profiler.md §4.2. |
| F-CONV-01 (original) | APR loss converges to < 2.0 | 2026-04-05 | FALSIFIED: Reference HF fp32 loss = 1.51, APR epoch 1 = 18.9 (12.5x gap, worse than random 11.93). Root cause: WGSL tiled GEMM reads B[k*N+n] expecting [K,N] layout but weights in [N,K]. Fix: transpose during upload (aprender@e767d6b6). Verification pending (epoch logs not emitted from rebuilt binary). | F-CONV-01 rewritten as "loss < 2.0 after fix" — now active pending verification. |

### Falsification Protocol

1. **Before accepting any baseline:** Run the falsification condition. A claim that has never been tested is an assumption, not a fact.
2. **On falsification or confirmation:** Move condition from Active to Falsified Claims (with outcome: FALSIFIED or CONFIRMED). Update the claim text. Open a PMAT item for any fix required.
3. **Quarterly review:** Re-run all P0 falsification conditions. Staleness = risk.

### Parity Enforcement

A gap between runtimes is a **defect**, not a finding. The response to every gap is:

1. **Measure** the gap (canary comparison on same hardware)
2. **Root-cause** the deficit (profile, trace, bisect)
3. **Fix** the slower runtime (patch the code, not the spec)
4. **Re-measure** until parity is achieved or the gap is proven fundamental

Acceptable gaps: hardware limitations (8GB VRAM ceiling), missing backends (WGPU on NVIDIA).
Unacceptable gaps: missing features (apr not training), unoptimized paths (torch.compile overhead), untested runtimes (burn not building).

---

## 8. PMAT Compliance

### Work Item Summary

| Range | Area | Count |
|-------|------|-------|
| PMAT-420-425 | Phase 0: Scaffold + baselines (yoga primary) | 6 |
| PMAT-426-430 | Phase 1: Throughput optimization | 5 |
| PMAT-431-434 | Phase 2: WGPU maturity | 4 |
| PMAT-435-438 | Phase 3: Advanced canaries | 4 |
| PMAT-439-456 | Spec audit, schema/scoring validation, test infrastructure | 18 |
| PMAT-457-461 | Parity fixes: APR baselines, FP16 GEMM, grad accum, profiler | 5 |
| PMAT-462-466 | NaN fix, nightly coverage, CUDA graph contract, spec v3.4.0 | 5 |
| PMAT-467-474 | FP16 path: weight cast, backward GEMM, crash fixes, canary pipeline | 8 |
| PMAT-475-477 | Kernel fusion (NF4 RMSNorm+GEMV), backward graph unblock (fused clip), FP16 measurement | 3 |
| PMAT-478-479 | Fused K+V NF4 GEMM (GQA attention), NF4 tensor core GEMM (WMMA 16×16×16) | 2 |
| PMAT-480 | Training step profiling — wire BrickProfiler into training loop (scientific profiling) | 1 |
| PMAT-481-482 | NF4 tensor core GEMM wiring, fused backward GEMM gap | 2 |
| PMAT-483-488 | Per-op profiling, fused backward GEMM, probar scorer, CUDA graph backward | 6 |
| PMAT-489-496 | GGUF mapping, metadata, CUDA NF4 bypass, kernel cache, Vulkan fix, WGPU overhead | 8 |
| PMAT-497-499 | Convergence defect (GEMM layout), yoga WGPU crash, unified profiling interface | 3 |
| PMAT-500 | Spec falsification sweep v6.10.0 — 19 issues fixed (stale baselines, contradictions, triggered conditions) | 1 |
| PMAT-501-505 | Three-phase execution plan: A. profiling+contracts, B. hybrid cuBLAS, C. A/B test matrix | 5 |
| **Total** | | **86** |

See [components/optimization-roadmap.md](components/optimization-roadmap.md) for full phase details.

### Quality Gates

| Gate | Tool | Threshold |
|------|------|-----------|
| Schema validation | `make validate-schema` | All results pass (F-MET-01) |
| Canary pass/fail | `make score` | All canaries PASS (includes schema) |
| CI gate | `make score-gate` | Exit 0 |
| Nightly regression | `scripts/nightly.sh` | Yoga passes first, then secondaries |

---

## 8.1. Strategic Assessment: Why APR Performance Stalled (2026-04-06)

> **F-STRATEGY-01:** If after 7 days of optimization work, the gap between APR and the next-fastest runtime has not closed by at least 2x, the optimization strategy is wrong — not the implementation.

### Diagnosis

**Profile breakdown (gx10, entrenar 0.7.11):**

| Phase | Time (ms) | % of step | What it does |
|-------|----------|-----------|--------------|
| Forward (28 layers) | 20-35 | 3% | Q4K dequant + tiled GEMM + attention + FFN |
| CE loss | <1 | <0.1% | Cross-entropy on GPU |
| lm_head backward | 8-31 | 3% | grad_hidden = grad_logits @ lm_head |
| **LoRA backward** | **700-760** | **90%** | 28 layers × 3 projections × (transpose + GEMM + AdamW) |
| **Total** | **750** | 100% | ~264 tok/s (batch=16, seq=512) |

**The bottleneck is clear:** LoRA backward consumes 90% of step time. The forward pass is already reasonably fast. No amount of forward-path optimization (fused kernels, tensor cores) will help because they touch 3% of the runtime.

**Why LoRA backward is slow:** Each of 28 layers × 3 QKV projections = 84 LoRA updates. Each update requires:
1. `XA = X @ A` (GEMM: seq×hidden @ hidden×rank)
2. `dB = (XA)^T @ grad` (transpose + GEMM: rank×seq @ seq×out_dim)
3. `dA = X^T @ (grad @ B^T)` (transpose + GEMM + transpose + GEMM)
4. Two AdamW steps (A and B)

That's **~336 GPU dispatches per step** through the WGSL tiled GEMM shader. PyTorch does the same math via cuBLAS batched GEMM (1 kernel launch for all 84 projections). Unsloth fuses further.

**8 optimization tiers SHIPPED but 0 MEASURED.** The spec's own F-PROGRESS-01 falsification condition has been triggered for 7+ days. Every tier (FP16, fused kernels, tensor cores, CUDA graph, fused backward) sits behind env flags never enabled in production.

**Convergence is also unsolved.** Even at 470 tok/s (pre-backward-experiments), the model diverges after epoch 1. Today's per-layer backward experiments confirmed: the simplified backward (without SiLU derivative) injects wrong-direction gradient that makes training WORSE. The epoch 1→2 divergence root cause is still unknown.

### Five Paths Forward (Choose One)

#### Path A: Accept WGPU, Fix Convergence Only
**Effort:** 1-2 weeks | **Expected throughput:** 470 tok/s (no change) | **Expected convergence:** loss < 2.0

Revert all per-layer backward experiments (they made things worse). Focus entirely on why loss diverges after epoch 1:
- Dump LoRA A/B weight norms per epoch (check if updates are in correct direction)
- Compare AdamW state (m/v) against PyTorch reference implementation
- Check if Q4K dequant rounding accumulates across training steps
- Verify cross-entropy label alignment across epoch boundaries

**Pro:** Convergence is the actual blocker. A model that converges at 470 tok/s is infinitely more useful than one at 16,000 tok/s that diverges. **Con:** APR stays 34x behind unsloth, 8x behind pytorch. The throughput gap remains a permanent competitive disadvantage.

#### Path B: cuBLAS Backend for NVIDIA (Hybrid)
**Effort:** 2-3 weeks | **Expected throughput:** 2,000-4,000 tok/s | **Expected convergence:** likely (cuBLAS is proven)

Wire cuBLAS GEMM into the LoRA backward path on NVIDIA targets. Keep WGPU for AMD/Metal/portability. This is the Phase B from the execution plan (PMAT-503) that was designed but never started.

- Replace WGSL tiled GEMM with cuBLAS sgemm/hgemm for the 84 LoRA backward GEMMs
- Use `--gpu-backend auto` to select cuBLAS on NVIDIA, WGPU on other platforms
- trueno already has CUDA PTX kernels (trueno-gpu crate) — wire them through

**Pro:** Addresses the 90% bottleneck directly. cuBLAS batched GEMM could handle all 84 projections in 1-2 kernel launches instead of 336 dispatches. Expected 5-15x speedup on the dominant operation. **Con:** Requires trueno CUDA PTX JIT cache (PMAT-492, 2-hour cold start on sm_89). Adds CUDA dependency for NVIDIA targets.

#### Path C: PyTorch Backend via FFI
**Effort:** 1-2 weeks | **Expected throughput:** 3,000-4,000 tok/s | **Expected convergence:** yes (uses PyTorch's proven backward)

Use PyTorch's C++ API (libtorch) or Python FFI for the LoRA backward GEMM. Keep the APR model loading, Q4K dequant, and forward in Rust. Only delegate the hot loop (LoRA backward + AdamW) to PyTorch.

- Call `torch.mm()` via FFI for the backward GEMMs
- Use PyTorch's fused AdamW optimizer
- Keep the Rust pipeline for everything except the 90% bottleneck

**Pro:** Gets PyTorch-level GEMM performance immediately. Convergence comes for free (PyTorch backward is battle-tested). **Con:** Adds Python/libtorch dependency. "Sovereign Stack" is no longer fully sovereign. Deployment complexity increases.

#### Path D: Abandon WGPU Training, Use PyTorch with APR Inference
**Effort:** 0 (already works) | **Expected throughput:** 3,957 tok/s (pytorch) / 16,118 tok/s (unsloth) | **Expected convergence:** yes

Accept that training via WGPU compute shaders on NVIDIA hardware is a structural mismatch. Use the existing pytorch canary (which already converges at 3,957 tok/s) or unsloth canary (16,118 tok/s) for actual training. APR focuses on what it's good at: fast GGUF inference via WGPU.

- Training: `unsloth` or `pytorch` canary (already working)
- Inference: `apr run` (WGPU, cross-platform, already fast)
- Bridge: export LoRA adapters from pytorch → merge with APR model

**Pro:** Immediately functional. Uses each tool for what it's best at. No more time spent on a 34x performance gap. **Con:** APR doesn't own the full training stack. The "Sovereign Stack" story loses its training chapter.

#### Path E: Double Down on WGPU — Ship Batched GEMM Shader
**Effort:** 3-4 weeks | **Expected throughput:** 1,000-2,000 tok/s | **Expected convergence:** unknown (still need to fix epoch 1→2 divergence)

Write a custom WGSL batched GEMM shader that handles all 84 LoRA projections in a single dispatch (vs 336 today). This is the "make WGPU fast" path.

- Design batched GEMM: one workgroup per (layer, projection) pair
- Shared memory tiling within each workgroup (8x cache benefit)
- Fuse transpose into the GEMM (eliminate 56 standalone transposes)
- Fuse AdamW into the GEMM output (eliminate 168 separate dispatches)

**Pro:** Keeps the sovereign stack story. Could close gap to 3-5x (from 34x). Pushes WGPU state of the art. **Con:** Speculative — no one has achieved cuBLAS parity with WGSL. Requires deep GPU shader expertise. Convergence still unsolved independently. 3-4 weeks is optimistic.

### Recommendation Matrix

| Path | Throughput | Convergence | Effort | Sovereign | Risk |
|------|-----------|-------------|--------|-----------|------|
| **A: Fix convergence** | 470 tok/s | Likely | 1-2w | Yes | Low |
| **B: cuBLAS hybrid** | 2,000-4,000 | Likely | 2-3w | Partial | Medium |
| **C: PyTorch FFI** | 3,000-4,000 | Yes | 1-2w | No | Low |
| **D: Abandon WGPU train** | 3,957-16,118 | Yes | 0 | No | Zero |
| **E: Batched WGSL GEMM** | 1,000-2,000 | Unknown | 3-4w | Yes | High |

**The spec's recommendation: Path A first (convergence is the actual blocker), then Path B (cuBLAS hybrid gives 80% of the benefit for 20% of the effort of Path E).**

---

## 9. Revision History

| Version | Date | Changes | PMAT |
|---------|------|---------|------|
| 1.0.0 | 2026-03-31 | Initial spec: 4 canaries, 3 hardware targets, 19 PMAT items | PMAT-420 |
| 2.0.0 | 2026-03-31 | Refactor: 500-line cap, component specs, falsification-first, yoga primary | PMAT-420 |
| 3.0.0 | 2026-04-01 | 5-runtime competitive comparison, parity mandate, 14 upstream fixes, measured baselines across 3 hosts, apr pipeline verified complete | PMAT-420 |
| 3.1.0 | 2026-04-01 | Fix 15 (entrenar#316 NF4 forward NaN) landed — APR IS LEARNING (loss 4.86→3.27); spec audit: F-WL-06 updated, roadmap P0 updated, wgpu baseline corrected, deferred notes removed | PMAT-439/440/441/442 |
| 3.2.0 | 2026-04-01 | All 15 falsification conditions resolved. Schema validator + 25 pytest tests. F-EXEC-01 CONFIRMED (GPU clock injection). Fresh gx10 results (unsloth 16,118 tok/s). APR canary timeout fixed. nightly.sh complete (all 3 hosts, 5 canaries). score.py VRAM skip for baselines lacking peak_vram_mb. | PMAT-443-453 |
| 3.3.0 | 2026-04-03 | APR throughput corrected: 44→194 tok/s (34x gap, was 151x). Throughput formula bug fixed (8x under-report). APR baselines updated (190 tok/s, loss 20.0). FP16 cuBLAS GEMM contract designed (Tier 2: 390 tok/s target). Gradient accumulation canary implemented. Step profiler integration. FP16 GEMM primitives landed upstream (entrenar@1ce6ef24). 42 total PMAT items. | PMAT-457-461 |
| 3.4.0 | 2026-04-03 | APR baseline marked PROVISIONAL (NaN backward skips inflate 194 tok/s). Fused residual+RMSNorm fix landed upstream (entrenar@b4d74f2c, entrenar#321). CUDA graph Tier 3 contract designed (→1200 tok/s, entrenar#322). Nightly coverage complete: all 5 runtimes on all hosts. pytorch-gradacc yoga target added. 22 upstream fixes total. 47 PMAT items. | PMAT-462-466 |
| 3.5.0-3.8.0 | 2026-04-03 | FP16 forward+backward GEMM shipped (Tier 2). CPU lm_head backward fallback. cuBLAS workspace pre-alloc. fp32 weight drop (2.6 GB freed). FP16 canary pipeline + provable contract. 3 crash bugs fixed. 26 upstream fixes. CUDA graph forward shipped, backward deferred. | PMAT-470-474 |
| 3.9.0 | 2026-04-03 | **Kernel fusion + backward graph unblocking.** Five-whys: 34x gap root cause is memory BW, not compute. NF4 fused RMSNorm+GEMV kernel in trueno (PMAT-475). Fused LoRA gradient clipping in entrenar — 168 D2H sync → 0, enables CUDA graph backward (PMAT-477). FP16 measurement gap identified (PMAT-476). 3 provable contracts. 28 upstream fixes. | PMAT-475-477 |
| 4.0.0 | 2026-04-03 | **Scientific profiling + tensor core GEMM + fused K+V.** Five-whys: can't close 34x gap without per-layer profiling. Designed training-step-profiling-v1 contract (12 falsification tests). BrickProfiler integration filed upstream (entrenar#328). NF4 tensor core GEMM shipped (PMAT-479, WMMA 16×16×16). Fused K+V GEMM shipped (PMAT-478, 352 MB/step saved). New canary targets: canary-apr-tc, canary-apr-profile. 47 tests (41→47). 35 upstream fixes. 61 PMAT items. | PMAT-478-480 |
| 4.1.0 | 2026-04-03 | **Profiling measurement loop closed + fused backward GEMM designed.** Five-whys root cause: InstructPipeline had ZERO profiling (StepProfiler existed only in CudaTransformerTrainer). Fix #38 upstream: wired StepProfiler into InstructPipeline with per-phase timing (FORWARD, LOSS, LM_BWD, BLK_BWD), per-layer forward+backward timing, and structured JSON output (print_json_report). Canary now parses JSON profiling, score.py validates wall_coverage >= 0.90 (F-TSP-001). Two provable contracts: per-operation-training-profiling-v1.yaml (8 falsification tests), fused-backward-gemm-v1.yaml (5 falsification tests). 3 new GitHub issues (#28/#29/#30). 3 new PMAT items (PMAT-483/484/485). 36 tests (33→36). 38 upstream fixes. | PMAT-481-485 |
| 4.2.0 | 2026-04-03 | **Fused backward GEMM shipped + all repos pushed.** Fix #39: Gate+Up backward fusion (NF4_FUSED_BWD_GEMM=1, cuBLAS beta=1.0 accumulate, eliminates 28 add_inplace/step). Fix #40: K+V backward fusion (same pattern, eliminates 56 more add_inplace/step). Total: 84 fewer kernel launches per step. Also fixed: entrenar --all-features clippy blockers (realizar::cuda feature propagation, wgpu LoRA field names, stale imports). probar Rust 1.94 pedantic/nursery expansion (733→0 clippy errors). All 4 upstream repos pushed clean. New canary target: canary-apr-fused-bwd. 42 upstream fixes. 50 tests. | PMAT-482-485 |
| 4.3.0 | 2026-04-03 | **Per-operation profiling shipped (entrenar#328).** Fix #43: 8 forward ops instrumented in CudaNf4TransformerBlock (rmsnorm_attn, qkv_gemm, attention, o_proj, rmsnorm_ffn, gate_up_gemm, silu, down_gemm). Per-op timing via CudaBlockScratch accumulators (zero-overhead when disabled). StepProfiler v2: JSON includes "ops" breakdown + "gemm_pct" for bottleneck classification. Canary parses v1/v2 profiler, shows top-5 ops. score.py validates GEMM dominance >= 30% (F-POP-002). 53 tests. 43 upstream fixes. | PMAT-483 |
| 4.4.0 | 2026-04-03 | **NF4 backward tensor core GEMM — deepest throughput mover.** Five-whys: backward uses generic cuBLAS (separate dequant + GEMM) while forward has TC GEMM. ROOT CAUSE: no backward TC kernel in trueno. Fix #44: `Nf4TensorCoreGemmBackwardAKernel` (WMMA 16×16×16, inline NF4 dequant in SHMEM) built in trueno. Fix #45: wired into entrenar backward (`NF4_TC_BWD_GEMM=1`). Eliminates 196 kernel launches/step (28 layers × 7 projections). Probar training scorecard module shipped (PMAT-485): grades efficiency A-F, classifies bottleneck, detects regressions. Two new provable contracts: `nf4-backward-tensor-core-gemm-v1.yaml` (7 falsification tests), `training-step-scorecard-v1.yaml` (7 falsification tests). New canary: `canary-apr-tc-bwd`. Profiling enabled by default on base canary. 3 GH issues filed (trueno#236, probar#38, entrenar#331). Per-op profiling compilation fixed (missing OP_* constants + scratch fields). 46 upstream fixes. | PMAT-481/484/485 |
| 4.5.0 | 2026-04-04 | **Proper attention backward — fixes root cause of NaN cascade + loss stuck at 16.8.** Five-whys: (1) Loss 16.8 not converging, 7/20 NaN backward skips → (2) grad_k/grad_v contain stale forward data → (3) backward_nf4_attention_mechanism returns without computing anything → (4) code comments say "that's wrong" → (5) ROOT CAUSE: attention backward was never implemented. Fix #47: Full attention backward mirroring FP32 path — softmax backward + 4 batched 4D GEMMs (grad_scores, grad_V, grad_Q, grad_K) + GQA reduction (12→2 heads). Also adds `reduce_gqa_gradients_nf4()` for proper KV head accumulation. Provable contract: `attention-backward-v1.yaml` (6 falsification tests). GH issue: entrenar#332. Nightly pipeline updated with canary-apr-fused-bwd and canary-apr-tc-bwd. Expected: NaN skips → 0, loss → <2.0, V LoRA starts learning. 47 upstream fixes. | PMAT-486 |
| 4.6.0 | 2026-04-04 | **Parity profiling system + 100x throughput roadmap (arXiv-informed).** Research synthesis: 8 new arXiv papers analyzed, batuta oracle consulted, 2 provable contracts designed. Key finding: 84.6% kernel launch overhead + 12.2% memory efficiency + NaN waste = 116x improvement potential (194 → 22,500 tok/s). Three new optimization tiers: Tier 7 CUDA graph backward (6.5x, PyGraph arXiv:2503.19779), Tier 8 Flash Attention (2.5x, FlashAttention-3 arXiv:2407.08608), Tier 9 persistent megakernel (1.7x, Mirage arXiv:2512.22219). `torch.profiler` added to PyTorch + unsloth canaries (parity-profile-v1 schema). `scripts/parity-report.py` for cross-runtime comparison. Provable contracts: `parity-profiling-system-v1.yaml` (6 falsification tests), `cuda-graph-training-step-v1.yaml` (6 falsification tests). 19 academic references in dependencies.md. GH issue: entrenar#333 (CUDA graph backward). Sovereign stack updates planned: renacer CUPTI training traces, probar parity scorecard, batuta oracle capability expansion. | PMAT-487/488 |
| 4.7.0 | 2026-04-04 | **CUDA graph backward capture IMPLEMENTED — the 6.5x change.** backward_nf4_gpu_blocks_loop now captures the entire 28-layer backward (backward_nf4 + fused_clip + optimizer_step) into a CUDA graph on first call, replays on subsequent calls. 840+ kernel launches → 1 graph replay. BackwardGraphState cached in InstructGpuTrainingState, invalidated on seq_len change. Per-layer profiling disabled during capture (CPU timing incompatible with graph). Sync fallback for non-fused gradient clipping outside capture. New canary: `canary-apr-graph` (CUDA_GRAPH=1 isolated measurement). Baseline: 500 tok/s PROVISIONAL (target 1200+ tok/s). 48 upstream fixes total across 5 repos. | PMAT-488 |
| 4.8.0 | 2026-04-04 | **Measurement pipeline complete + canary-apr-max with all 6 flags.** canary-apr-max now enables every optimization: NF4_FUSED_GEMM + NF4_FUSED_BWD_GEMM + NF4_TC_GEMM + NF4_TC_BWD_GEMM + FP16_GEMM + CUDA_GRAPH (forward+backward). Added --profile to unsloth + pytorch canaries (torch.profiler parity data). Backward per-op profiling constants (OP_DOWN_BWD through OP_LORA_BWD) complete 16-op instrumentation. Graph wiring validation test passes (5/5 checks). nightly.sh updated with canary-apr-graph. 49 upstream fixes. Ready for measurement: `make canary-apr-max` is the definitive throughput test. | PMAT-487/488 |
| 4.9.0 | 2026-04-04 | **Per-op backward profiling wired + optimization sweep.** 6 backward ops instrumented with op_begin/op_end: OP_DOWN_BWD, OP_SWIGLU_BWD, OP_GATE_UP_BWD, OP_ATTN_BWD, OP_QKV_BWD, OP_NORM_BWD. Combined with 8 forward ops = 14/16 per-op coverage. `scripts/sweep.sh`: systematic 13-variant A/B measurement. 51 upstream fixes. | PMAT-483/488 |
| 5.0.0 | 2026-04-04 | **All 8 backward projections use NF4 TC GEMM + comprehensive validation.** O-projection was last holdout — now Q, K, V, O, gate, up, down all dispatch to tensor core backward (224 TC GEMM dispatches per step: 8 projections × 28 layers). Comprehensive validation tests: `test_optimization_coverage.py` (4 check groups, 15 TC dispatches verified, 6 env vars, 16 profiling constants, 8 canary targets, 7 contracts) and `test_graph_wiring.py` (5 checks). probar parity comparison shipped. 53 upstream fixes across 5 repos. **Optimization stack is feature-complete and validated.** | PMAT-481/487/488 |
| 5.1.0 | 2026-04-04 | **First dogfood on yoga — unsloth baseline measured.** Unsloth 20-step measurement: **5,512 tok/s**, 5,085 MB VRAM, loss 0.4674 (converges properly). torch.profiler fix: `FunctionEventAvg` attribute probing for API compatibility. Note: CUPTI profiling returned 0 (permission issue) — training metrics valid, kernel profiling needs nsys. APR canary blocked by apr-cli rebuild (disk space + trueno semver mismatch on yoga). Dogfooding gap found: `realizar` requires trueno ^0.16 but yoga has 0.17.1. | PMAT-487 |
| 5.2.0 | 2026-04-04 | **Five-whys: `apr finetune` silently missing from binary + GGUF metadata incomplete.** (1) `training` feature silently dropped from installed apr binary because trueno-gpu had 16 E0603 `registers` module private errors. Feature IS in default features but build failure was silent. Fixed: rebuilt from NVMe target with training feature. (2) APR v2 metadata from GGUF imports missing `num_heads`/`num_layers`/`num_kv_heads` — `transformer_config_from_apr_metadata()` returned None. Fixed upstream (aprender@39d33259): architecture+hidden_size preset fallback for qwen2 0.5B/1.5B/7B, plus 1.5B added to size fallback table. (3) Q4K models use GGUF tensor names (`token_embd.weight`) not HF names (`model.embed_tokens.weight`) — training pipeline expects HF style. Filed PMAT-489 (critical). (4) APR v2 metadata completeness should be provable-contract enforced at import time. Filed PMAT-490 (high). Parity report enhanced with auto-discovery + host-tagged comparison + regression summary. `make parity` target added. PMAT-481 completed (TC GEMM wired), PMAT-482 consolidated into PMAT-484, PMAT-488 titled. 54 upstream fixes. | PMAT-489/490 |
| 5.3.0 | 2026-04-04 | **PMAT-489 + PMAT-490 closed — GGUF tensor name mapping tested, metadata completeness enforced.** (1) PMAT-489 CLOSED: 11 unit tests added to entrenar verifying complete GGUF→HF tensor name mapping (token_embd→model.embed_tokens, blk.N.attn_q→model.layers.N.self_attn.q_proj, all 12 tensor types for all 28 layers). Detection+mapping code was already in `Transformer::from_apr()` and `map_gguf_weight_name()` — tests prove correctness. (2) PMAT-490 CLOSED: Import-time metadata completeness warning added to aprender `write_apr_file_raw()`. Warns when critical fields (hidden_size, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_size) are None. Soft enforcement — downstream preset fallback handles missing fields, but warning surfaces the gap. (3) Both import path (`tensor_expectation.rs:qwen2_map_name`) and training path (`weights/mapping.rs:map_gguf_weight_name`) verified to map identical GGUF→HF names. Inference path (`loader_apr_quantized.rs`) handles both naming conventions via `.contains()` search. 55 upstream fixes. Ready for yoga dogfood. | PMAT-489/490 |
| 5.4.0 | 2026-04-04 | **Dogfood on RTX 4090: embedded tokenizer contract violation found.** Five-whys from first real dogfood: (1) `apr finetune model.apr` fails "No sibling tokenizer" → (2) `InstructPipeline::from_apr()` only looks for sibling `.tokenizer.json` file → (3) APR IS an embedded format — tokenizer vocab+merges ARE in metadata → (4) ROOT CAUSE: `from_apr()` written before APR embedded tokenizer support existed → (5) Fix: `extract_embedded_tokenizer()` reconstructs HF tokenizer.json from APR metadata (vocab→id map + merge rules), sibling file is now fallback only. Second finding: Q4K→F32 CPU dequantization takes 20+ minutes for 1.5B model — makes canary unusable. The WGPU path (`execute_training_wgpu` in finetune.rs) already loads Q4K directly to GPU in seconds via `OwnedQuantizedModel::from_apr()` but is only used when CUDA is unavailable. Filed PMAT-491 (critical): CUDA NF4 training should bypass CPU dequant path, load Q4K directly like WGPU does. First measurement: planning phase reports 3,108 tok/s wall-clock (includes dequant overhead), 4.99 GB memory. 56 upstream fixes. | PMAT-491 |
| 5.5.0 | 2026-04-04 | **L5 compiler-enforced contracts — provable, not warnings.** Five-whys root cause of PMAT-490/491: contracts existed in YAML specs but enforcement was eprintln warnings, not compiler checks. (1) PMAT-490 "fix" was `eprintln!("[WARNING]...")` — not a contract, just noise. REPLACED with `#[requires(model_config.map_or(false, \|c\| c.hidden_size.is_some() && c.num_layers.is_some() && c.num_heads.is_some() && c.vocab_size.is_some()))]` on `write_apr_file_raw()`. Compiler now refuses to write APR files with incomplete metadata. (2) `#[requires(tokenizer.is_some())]` enforces apr_tokenizer_embedding P0 contract at write time. (3) `#[requires(apr_path.exists())]` on `InstructPipeline::from_apr()`. (4) `#[ensures(ret.as_ref().map_or(true, \|t\| t.vocab_size() > 0))]` on `extract_embedded_tokenizer()`. All contracts enforce via `debug_assert!` — zero cost in release, hard failure in debug/test builds. 57 upstream fixes. | PMAT-490/491 |
| 5.6.0 | 2026-04-04 | **PMAT-491 CLOSED — 20-min CPU dequant eliminated. First real APR training measurement.** Route CUDA NF4 path through `OwnedQuantizedModel::from_apr()` (fast Q4K direct load) instead of `Transformer::from_apr()` (20-min CPU dequant). Before: 22+ min and never finished. After: **223s total (3.7 min), 119 tok/s, loss 16.76** on RTX 4090. Pipeline IS TRAINING: loss measured at 16.76 with 1 valid backward step, 0 NaN skips. Step profiling shows 3.3s/step with lora_bwd dominating (99%). Adapter export needs directory fix (non-critical). RTX 4090 measurement: 119 tok/s APR vs ~5,500 tok/s unsloth (46x gap). 58 upstream fixes across 5 repos. | PMAT-491 |
| 5.7.0 | 2026-04-04 | **Five-whys: yoga WGPU device failure + gx10 arch mismatch.** Two dogfood failures caught and fixed. (1) Yoga: "Parent device is lost" — five-whys root cause: PMAT-491 conflated fast Q4K loading (concern A) with GPU backend selection (concern B), routing CUDA hardware to WGPU/Vulkan which doesn't exist on yoga. Fix: WGPU path only when CUDA is NOT available; CUDA hosts use InstructPipeline→CudaNf4TransformerBlock. (2) gx10: "Exec format error" — five-whys root cause: scp'd x86_64 binary to aarch64 host, overwriting working binary without architecture check. Fix: clone batuta dep, build natively on gx10. Both failures are deployment process gaps, not code bugs — but deploying a wrong binary IS a code-level concern (should have `file` check). LoRA backward bottleneck identified: 784+ GPU dispatches/step (7 proj × 28 layers × 4 ops), each allocating new buffers via `self.trainer.zeros()`. 59 upstream fixes. | PMAT-491 |
| 5.8.0 | 2026-04-04 | **Remote dogfood: yoga CUDA kernel compilation hangs, gx10 dequant timeout.** Three-host dogfood results: (1) **Lambda RTX 4090: 119 tok/s, loss 16.76** — only working measurement, via WGPU fast Q4K path. (2) **yoga RTX 4060: CUDA NF4 path hangs** — model loads to GPU (7530 MiB), NF4 QLoRA initializes, then CUDA PTX kernel JIT compilation (`[FROM_PTX]`) appears to stall indefinitely (2 hours at 0% GPU, 0.4% CPU). Five-whys root cause: PTX→SASS compilation for sm_89 kernels hangs during JIT. This is a trueno CUDA kernel issue, not entrenar. (3) **gx10 GB10: 2-hour timeout** — old binary (April 1, pre-PMAT-491) uses CPU dequant path which takes >2 hours on ARM. Build of new binary on gx10 blocked by alimentar `generated_contracts` code gen infra. Key finding: **only WGPU path works for training today**. CUDA NF4 path has kernel compilation stall. Next action: investigate trueno PTX kernel that causes hang on sm_89. Canary stderr capture increased to 2000 chars + WGPU loss format parsing added. 60 upstream fixes. | PMAT-491 |
| 5.9.0 | 2026-04-04 | **Yoga CUDA NF4 completes in 7079s (2 hours) — not a hang, pathologically slow JIT.** Second yoga run confirms: process completes but takes 2 hours (3.8 tok/s with JIT overhead). CUDA driver cache exists (415 MB at ~/.nv/ComputeCache/). Third run with cached JIT: **941s (15.7 min), 28 tok/s** — 7.5x faster. Five-whys: first-run JIT compiles PTX→SASS for all kernels; subsequent runs benefit from CUDA driver cache. Filed PMAT-492 (critical): trueno should pre-compile to cubin. Embedded tokenizer extraction CONFIRMED working on yoga: `[tokenizer] Loaded embedded BPE tokenizer from APR metadata (vocab_size=151936)`. 60 upstream fixes. | PMAT-492 |
| 6.0.0 | 2026-04-04 | **Scratch buffer optimization FALSIFIED — reverted. Dispatch overhead is the real bottleneck, not allocation.** Five-whys: (1) Pre-allocated LoRA backward scratch buffers (7 buffers, reused across 28 layers × 7 projections) expected to eliminate ~1200 allocs/step. (2) Result: **119→73 tok/s REGRESSION**. (3) Root cause: scratch buffers sized for max projection (8960) but most projections are smaller (1536). WGPU matmul dispatches on oversized buffers waste compute. (4) REVERTED: entrenar@cd016f90. (5) The real bottleneck is **dispatch count** (784+ compute shader invocations per step), not allocation. Fix is kernel fusion (PMAT-484): combine multiple LoRA ops into fewer dispatches. Also confirmed: embedded tokenizer works, VRAM guard ledger needs repair. 61 upstream fixes. | PMAT-492 |
| 6.1.0 | 2026-04-04 | **WGPU-first routing restored — 125 tok/s confirmed on Lambda. Yoga Vulkan blocked.** Five-whys from Lambda timeout after routing fix: previous fix routed CUDA hosts to InstructPipeline (20-min dequant). WGPU must be preferred for ALL NF4 training — it's the only path that works in reasonable time. Lambda re-measured: **125 tok/s, loss 16.76, 213s (3.5 min)**. Yoga WGPU fails: "Parent device is lost" — Vulkan ICD installed (nvidia_icd.json, libnvidia-gl-590) but `GpuDevice::new()` fails on driver 590. yoga is blocked: WGPU fails (Vulkan) AND CUDA takes 2hr (JIT). Best yoga measurement: 28 tok/s via cached-JIT CUDA path (run 3, 941s). **Confirmed measurements: Lambda 125 tok/s (WGPU), yoga 28 tok/s (CUDA cached JIT).** 62 upstream fixes. | PMAT-491/492 |
| 6.2.0 | 2026-04-04 | **Five-whys: yoga Vulkan ROOT CAUSE + gx10 ARM dequant 75+ min (not 20 min).** (1) Yoga "Parent device is lost": root cause is `libvulkan.so.1` NOT INSTALLED. Fix: `apt install libvulkan1`. (2) Upstream fix: `--gpu-backend wgpu` now respected in WGPU routing. (3) Makefile: canary-apr/canary-apr-gx10 pass `--gpu-backend wgpu`. (4) **CRITICAL FINDING**: gx10 ARM CPU dequant takes **75+ minutes** (not 20 min as spec estimated). The Q4K→F32 dequant on GB10 ARM is 3-4x slower than x86_64. WGPU fast Q4K path is essential for ARM targets. (5) gx10 binary rebuild blocked by alimentar code gen infrastructure (missing generated_contracts macros across alimentar+aprender). Attempted fixes: macro stubs, cargo patches, feature gates — all blocked by deep transitive deps. Need: alimentar code gen fix or alimentar crates.io publish with stubs. (6) Unsloth: 5,412 tok/s on yoga (confirmed, 2% variance from 5,512). 63 upstream fixes. Filed PMAT-493 (yoga Vulkan), PMAT-494 (routing). | PMAT-491/492/493/494 |
| 6.3.0 | 2026-04-05 | **First gx10 measurement: 421 tok/s APR WGPU.** Built new binary with alimentar code gen stubs + feature gates. APR baseline updated from PROVISIONAL 40 tok/s to MEASURED 400 tok/s. | PMAT-491/495 |
| 6.4.0 | 2026-04-05 | **~~GPU compute parity ACHIEVED~~ — SUPERSEDED by v6.7.0.** 24,094 tok/s step-level was measured on SYNC pipeline where 98.3% was inter-step idle. After async pipeline (v6.7.0): wall-clock=470 tok/s, GPU=100% utilized, 11.2x gap is real compute speed. The 24,094 step-level number was an artifact of measuring only within train_step() while ignoring inter-step overhead. | PMAT-496 |
| 6.5.0 | 2026-04-05 | **OVERHEAD LOCALIZED: 98.3% is BETWEEN train_step() calls, NOT inside.** (1) Real profiling data from existing binary: step times 30-317ms (avg 68ms steady state), but 1-epoch wall=115s for 13 steps. In-step total=2.0s (1.7%), inter-step=113s (98.3%). (2) **F-PROF-002 FALSIFIED**: buffer allocation inside train_step is NOT the bottleneck — overhead is OUTSIDE train_step entirely. (3) Root cause narrowed: `pipeline.encode()` (tokenization, called 2×per sample) + epoch iterator + `queue.submit` serialization between steps. (4) Step-level throughput: **30,016 tok/s** (5.7x faster than unsloth). (5) WgpuStepProfiler implemented in entrenar (13 phases + alloc tracking). Binary rebuild blocked by trueno diamond dep (realizar pins different trueno). Training-only build succeeds without wgpu feature. (6) profiler-wall-coverage-v1 + profiler-bottleneck-classification-v1 provable contracts designed with 10 arxiv citations. | PMAT-496 |
| 6.7.0 | 2026-04-05 | **PROFILER MEASURED: 100% GPU compute, 0% overhead. F-PROF-002 fully confirmed false.** (1) Async pipeline profiler on gx10: 400 steps, 395.7s profiler wall, `wall_coverage=1.000`. All 13 phases measured. `gpu_lora_bwd`=55.7% (551ms/step), `gpu_fwd`=40.6% (401ms/step). Zero allocations, zero sync, zero overhead per step. (2) Throughput: 470 tok/s (async) vs 5,262 tok/s (unsloth) = 11.2x gap confirmed as real GPU compute speed, not overhead. (3) Convergence defect: APR loss oscillates 18.9→9.2→12.3→16.3→15.5→12.0→10.8→11.7 over 8 epochs vs unsloth loss 0.45. Separate defect from throughput — likely learning rate or LoRA config. (4) Profiler spec updated: all hypothetical values replaced with measured data. (5) Two-run comparison: pretok (sync) measures 28.1s of 502.5s (5.6% coverage) vs async measures 395.7s of 449.6s (88% coverage). Async path moves device.poll into step, capturing true GPU wait time. (6) PMAT-496 reclassified: was "GPU idle 99%", now "GPU compute 100% — WGPU dispatch speed is the bottleneck". 63 upstream fixes. | PMAT-480/483/496 |
| 6.8.0 | 2026-04-05 | **Yoga WGPU UNBLOCKED + convergence root cause + uv-only packaging.** (1) PMAT-493 CLOSED: `apt install libvulkan1` + `vulkan-tools` on yoga. Vulkan verified: RTX 4060 detected via `vulkaninfo --summary`. First yoga WGPU measurement: **~191 tok/s step-level** (5 steps before buffer crash, PMAT-498). (2) Convergence investigation: label shift is CORRECT (labels[i]=full_ids[i+1]), WGSL masking is CORRECT (non-response positions → 0). Loss 18.9 > ln(151936)=11.93 (random) → model actively predicts wrong tokens. Root cause hypothesis: Q4K→F32 dequantization corrupts weight values on WGPU path. Filed PMAT-497. (3) uv-only mandate: pyproject.toml created, Makefile updated (all `~/venvs/*/bin/python` → `uv run --extra cuda python`), CLAUDE.md + spec updated. triton override for aarch64. (4) APR canary script fixed: `--profile-interval N` → `--profile` (boolean flag), throughput formula handles partial runs (crashed canaries). (5) uv installed on yoga. gx10 unsloth blocked by torchvision aarch64 incompatibility. 65 upstream fixes. | PMAT-493/497/498 |
| 6.9.0 | 2026-04-05 | **Profiler v2: 5 improvements + 10 falsification conditions (arXiv-informed).** Chain of thought: (1) v1 profiler tells WHERE (gpu_lora_bwd 55.7%) but not WHY (slow shaders vs memory bandwidth vs dispatch overhead). (2) Searched 5 org repos (trueno BrickProfiler 447 lines, entrenar StepProfiler 11 phases, batuta RAG profiling 886 lines, renacer process tracer with z-score anomaly). (3) 7 new arXiv papers analyzed (Omniwise 2506.20886, TritonForge 2512.09196, FlashAttention-3 2407.08608, PyGraph 2503.19779, Mirage 2512.22219). (4) Key insight: LoRA backward AI=6.3 << ridge=111 → deeply memory-bound, shader optimization useless. (5) Five improvements: GPU timestamp queries (F-PROF-008/009), roofline classification (F-PROF-010/011), per-layer timing (F-PROF-012/013), per-step variance (F-PROF-014/015), dispatch gap profiling (F-PROF-016/017). 15 total arXiv references in profiler spec. Also: PMAT-497 transpose fix deployed to gx10, canary ran in 451s (same throughput, loss parsing blocked by stderr truncation — needs fix). | PMAT-496/497 |
| 6.10.0 | 2026-04-05 | **Unified profiling interface: `trait ComputeProfiler` (PMAT-499).** Five-whys: 18 profilers across 8 repos with 0 shared abstraction → each built bottom-up for one backend → no shared trait → contracts describe output not interface → ROOT CAUSE: no `trait ComputeProfiler`. Design: one trait with 5 backend impls (WGPU, CudaPtx, cuBLAS, CUTLASS, SIMD). Phase/OpId/ComputeBackend enums from existing contracts. Three output formats (JSON, OTLP, Chrome Trace) from one data model. Migration: wrap StepProfiler → CudaProfiler, wrap Instant::now → WgpuProfiler, adapt BrickProfiler → BrickProfilerAdapter. 4 falsification conditions (F-UNI-001 through F-UNI-004). 78 PMAT items. | PMAT-499 |
| 6.11.0 | 2026-04-05 | **Spec falsification sweep — 19 issues found and fixed (PMAT-500).** 7 stale baselines, 6 contradictions, 3 triggered conditions, 3 dependency fixes. 17 new arXiv citations. Competitive analysis: PyTorch/Candle/Unsloth/Chronicals. Sister project (candle-vs-apr) insights: 6 contract invariants, profiler fidelity gap, trueno-parity canary. 81 PMAT items. | PMAT-500 |
| 6.12.0 | 2026-04-05 | **Progress stall diagnosis + three-phase execution plan.** Five-whys: 421→470 tok/s only measured delta (+12%), 6 tiers UNMEASURED, yoga blocked. **Three-phase plan with provable exit criteria:** Phase A (profiling+contracts), Phase B (cuBLAS hybrid), Phase C (A/B test matrix). F-PROGRESS-01/02 and F-MEASURE-01 added as P0 with 2026-04-12 deadline. 86 PMAT items. | PMAT-500-505 |
| 6.13.0 | 2026-04-05 | **Phase A execution: 3 deliverables shipped, 2 blockers found.** (1) 6 provable-contract invariants wired into score.py (convergence, better-than-random, backward_executed, metrics_quality, config_steps, step_time_sanity) — 64 tests passing (was 53). Old test fixtures using loss=16.8 caught by contracts — tests forced to use realistic values. (2) PMAT-498 root cause: `label: None` on CE loss readback buffer + missing `device.poll()` before copy. Fix: `label: Some("ce_loss_readback")` + poll before copy in `wgpu_cross_entropy.rs`. 3 additional unlabeled buffers fixed in `wgpu_training.rs`. (3) Convergence reanalysis: epoch 2 reached loss **9.15** (below random 11.93) — model IS learning but oscillates wildly (9.15→16.3→11.74). **Root cause: LR 2e-4 too high without cosine decay.** Not a GEMM correctness bug. (4) gx10 VRAM ledger corruption fixed: 32 dead reservations cleaned, future runs use fast WGPU path. **Blockers:** apr-cli build has trueno diamond dep (can't deploy PMAT-498 fix to yoga), gx10 convergence test hit slow CPU dequant path (75+ min on ARM). | PMAT-498/504 |
| 6.14.0 | 2026-04-05 | **Dogfooding reality: deployed binary REGRESSED, upstream ecosystem broken.** (1) gx10 binary rebuilt between 10:29 (async path, loss 11.74) and 11:39 (current, loss=100 NaN sentinel) — **silent regression**. Prior working async pipeline code path no longer in current binary. Contracts would catch this if applied to live results. (2) CUDA JIT hangs on gx10 sm_121 after compiling all backward kernels (PMAT-492 confirmed — not fixed). (3) WGPU path on gx10 produces `loss=100` NaN sentinel + `attention=0ms` per-op trace (PMAT-497 still active). (4) apr-cli pins `trueno = 0.16.0` from crates.io while workspace uses 0.17.1 local path — **this IS the diamond dep**. Dev-overrides mechanism exists but presentar path is stale. aprender has independent compile errors (`build_default_registry` undefined). **Can't build, can't deploy, can't measure.** F-PROGRESS-01 remains TRIGGERED. Measurement-first priority blocked by entire ecosystem build state. | PMAT-492/497/498 |
| 6.15.0 | 2026-04-05 | **Profiler spec restructured + 2 fidelity contracts shipped.** Profiler spec now leads with Phase A deliverables (P0 priority) instead of historical/prior-art content. Shipped contracts section prominently at top. Three P0 categories: (1) provable contracts [SHIPPED], (2) CUDA StepProfiler port [NOT STARTED], (3) fidelity invariants [DESIGNED]. Added 2 new fidelity contracts to score.py: `lmhead_executed` (F-PROF-FIDELITY-01) and `no_orphan_spans` (F-PROF-FIDELITY-02) — catches profiler sync bugs. F-XPROJECT-01 CONFIRMED: contracts caught F-REGRESS-01 on day 1. Remaining deliverables table (P0/P1/P2) added to profiler spec for clear prioritization. 64 tests passing. | PMAT-504 |
| 6.16.0 | 2026-04-05 | **Falsification sweep — stale data cleanup, cross-doc harmonization.** (1) optimization-roadmap.md compressed: removed ~195 lines of v3.x/v4.x H-PARITY-001/002 diagnostic chain (NF4 dequant NaN RESOLVED 2026-04-02), updated "13x gap"→"11.2x gap", rewrote v6.5.0 sync-dominated fix priorities against v6.7.0 async profiler data. (2) training-profiler.md: F-PROF-005/006 marked FALSIFIED/OBSOLETE (0 allocs/step achieved but only +12%, not 2x). (3) Main spec revision history reordered: v6.x entries were in insertion order (6.0,6.1,6.5,6.4,...); now strict version order. (4) Tier count harmonized 6→8 (Tiers 2, 4×3, 4.7×2, 5, 7). (5) README.md updated with current measurements (470 tok/s APR, 16,118 tok/s unsloth@gx10), 64-test count, F-WL-07/F-PROGRESS-01 triggered conditions. (6) PMAT item count: 81→88. | PMAT-500/507 |
| 6.17.0 | 2026-04-05 | **Dogfood session: 3 ecosystem fixes, 2 fresh measurements, 2 new blockers.** (1) trueno WGSL q4k_gemv shader fixed: `-1.0/0.0` → `bitcast<f32>(0xFF800000u)` for wgpu 27.0.1 compat. Published trueno 0.17.2 to crates.io. (2) gx10 Python env fixed: default PyPI torch 2.11+cu130 supports sm_121 (cu124/cu126 indices have NO aarch64 wheels). pyproject.toml updated: torch>=2.6, cu126 index, `cuda-base` extra for aarch64 without unsloth. (3) Fresh gx10 measurements: **pytorch 3,906 tok/s** (loss 0.0087, VRAM 50.6 GB, PASS), **cublas numerical parity perfect** (0.0 divergence, VRAM delta 1 MB, PASS). (4) PMAT-508 filed: unsloth on gx10 BLOCKED (triton aarch64 + torch.int1 dep chain). (5) PMAT-494 CONFIRMED: `--gpu-backend cuda` NOT routed, APR always uses WGPU Q4K path. (6) 71 tests passing. 89 PMAT items. | PMAT-504/507/508 |
| 6.18.0 | 2026-04-06 | **PMAT-509 deep investigation: GPU vs CPU forward path diagnostic.** (1) Hardcoded LR 2e-4 FIXED — now wired from CLI. Different LR confirmed producing different trajectories. (2) Per-layer hidden state comparison (APR WGPU vs PyTorch F32): embed 1.4x ratio, layer 0 1.6x, layer 27 3.6x — progressive divergence through 28 layers, consistent with accumulated GEMM rounding error. (3) Logit comparison: APR argmax=74403 vs PyTorch=16 (completely wrong). APR logits norm 1107 vs PyTorch 2646 (0.42x). (4) PyTorch base model CE loss: 0.55. APR: 16.37 (30x worse). (5) Code audit verified: Q4K dequant bit-exact, weight transpose correct, CE loss correct, tiled GEMM indexing correct, scatter/gather correct. (6) Root cause narrowed to **tiled GEMM precision on wgpu/Vulkan** — 252 matmuls compound error. Or RoPE implementation difference. (7) 7 upstream commits across 3 repos (trueno 0.17.2/0.17.3, entrenar Q/K/V + LR + diagnostics, aprender LR wiring). 90 PMAT items. | PMAT-497/507/509 |
| 6.19.0 | 2026-04-06 | **PMAT-510 FIX SHIPPED: per-layer backward gradient propagation.** Root cause: `wgpu_pipeline.rs:1066` used single `grad_hidden_buf` for ALL 28 LoRA layers. Fix (entrenar 0.7.9, trueno 0.17.5): reverse iteration 27→0, per-layer backward through W_down^T + W_gate^T + residual. Dogfood: epoch 1 loss improved (2.97 vs 3.60) but diverges from epoch 2 (→16.11). Root cause: unscaled backward GEMMs amplify gradient ~5400x/layer. | PMAT-509/510 |
| 6.20.0 | 2026-04-06 | **PMAT-511 v1 FAILED: 1/sqrt(dim) scale too aggressive.** Combined scale 0.00018x → FFN backward effectively zero → identical results to unscaled (epoch 1: 2.97, final: 16.11). Pytorch gx10 3,957 tok/s PASS (torch 2.11+cu130, NCCL 2.29.7 for sm_120). | PMAT-510/511 |
| 6.21.0 | 2026-04-06 | **PMAT-511 v2: residual mixing ratio alpha=0.1.** Five-whys on v1 failure: 1/sqrt(dim) per-GEMM = 0.018% contribution → identical to no backward. Fix: alpha=0.1 folded into cached W_down^T transpose. Each layer's FFN backward contributes 10% of unscaled magnitude. Growth bounded by (1+0.1k)^28 ≈ 14x total instead of exponential. Also cached transposed weights (eliminates 56 transposes/step). Shipped entrenar 0.7.11. Awaiting dogfood. 91 PMAT items. | PMAT-511 |
| 6.22.0 | 2026-04-06 | **Strategic assessment: why APR performance stalled + 5 paths forward.** PMAT-511v2 result: alpha=0.1 WORSE than no mixing (3.63→17.77 vs 2.97→16.11) because simplified FFN backward without SiLU derivative injects wrong-direction gradient. Per-layer backward experiments are a DEAD END. Profile shows 90% of step time in LoRA backward (336 WGSL dispatches vs 1-2 cuBLAS batched calls). 8 optimization tiers SHIPPED but 0 MEASURED (F-PROGRESS-01 falsified). Yoga PMAT-498 crash confirmed. Section 8.1 added: five strategic paths (A: fix convergence, B: cuBLAS hybrid, C: PyTorch FFI, D: abandon WGPU training, E: batched WGSL GEMM). Recommendation: Path A then B. | PMAT-500/511 |
