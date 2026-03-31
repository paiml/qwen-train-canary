# Training Canary Performance Specification

**Document ID:** PAIML-TRAIN-CANARY-001
**Version:** 2.0.0
**Last Updated:** 2026-03-31
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

---

## 1. Executive Summary

### What This Is

Competitive benchmark for fine-tuning throughput across five training runtimes — the training analog of qwen-coder-deploy's inference runtime comparison. Five canary workloads exercise distinct training paths for Qwen2.5-Coder-1.5B-Instruct.

**Parity mandate:** Gaps are not findings to document — they are defects to fix. When a runtime falls behind (apr adapter-init-only, wgpu not running, torch.compile regressing), the response is to fix the runtime, not to record the gap and move on. The canary exists to drive parity across all five runtimes. A runtime that cannot match the throughput leader on the same hardware is broken until fixed.

**Yoga is the initial primary target.** All baselines, thresholds, and falsification conditions are calibrated against the RTX 4060 Laptop (8 GB VRAM, sm_89). Secondary targets (gx10, intel) validate at larger batch sizes and alternative backends.

### Chain of Reasoning

**Step 1: Why canaries?** Training performance depends on a deep stack: PyTorch, CUDA runtime, cuBLAS/cuDNN, driver, GPU clocks, memory allocator, model weights, tokenizer, dataset pipeline. A regression in ANY layer silently degrades throughput. Canaries are short (100 steps, ~2 min), deterministic (seed=42, clock-locked), and produce machine-readable JSON.

> **F-EXEC-01:** If a canary fails to detect an artificially injected 15% throughput slowdown on yoga, the entire regression detection methodology is falsified.

**Step 2: Why five workloads?** Each canary isolates a different bottleneck. Like qwen-coder-deploy compares realizr/ollama/llama.cpp/vLLM/wgpu for inference, we compare training runtimes head-to-head:

| Canary | Runtime | Bottleneck | Why It Matters |
|--------|---------|-----------|---------------|
| **apr** | aprender/entrenar (Rust) | Sovereign Stack training | Native Rust QLoRA. CUDA (trueno PTX) or wgpu (Vulkan). Target to beat. |
| **unsloth** | unsloth (Python) | QLoRA + 4-bit quant | Best-known Python QLoRA. Production fine-tuning path. |
| **pytorch** | PyTorch (Python) | Raw training loop | Baseline with no optimizations. Isolates framework overhead. |
| **cublas** | PyTorch (Python) | GEMM backend parity | Runs SAME loop twice. Detects silent numerical divergence. |
| **wgpu** | burn (Rust) | Non-NVIDIA training | Burn framework via Vulkan on AMD. Cross-platform feasibility. |

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
- **No gradient accumulation** in canary mode -- measures raw per-step throughput.

> **F-EXEC-02:** If batch=4 seq=512 OOMs on yoga for the pytorch canary, the memory budget claim is falsified. Action: reduce to batch=2 or add gradient checkpointing.

---

## 2. Hardware Targets

### Yoga (PRIMARY -- RTX 4060 Laptop)

All initial baselines and falsification conditions target yoga. Secondary targets are deferred until yoga Phase 0 completes (PMAT-424).

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

**Deferred until:** Yoga baselines established (PMAT-424 complete). Canary uses batch=16.

### Intel (SECONDARY -- WGPU/Vulkan)

| Property | Value |
|----------|-------|
| GPU | AMD Radeon Pro W5700X, Navi 10 (RDNA 1), 8 GB |
| API | Vulkan 1.3 (Mesa RADV) |
| Host | Intel CPU, 64 GB RAM, LAN 192.168.50.100 |

**Deferred until:** burn-canary binary is operational (PMAT-431). Expected: 50-200 tok/s.

> **F-HW-02:** If WGPU throughput = 0 or burn-canary crashes on intel, the WGPU training feasibility claim is falsified.

---

## 3. Canary Workloads

### 3.0 APR Fine-Tune Canary (Sovereign Stack)

**File:** `canaries/apr/train.py` | **Backend:** CUDA via trueno | **Duration:** TBD

The **target to beat**. APR fine-tune uses aprender's native Rust training engine (entrenar) with trueno SIMD-accelerated tensor operations. Like realizr is the SSC inference engine, entrenar is the SSC training engine.

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
  "canary": "unsloth|pytorch|cublas|wgpu",
  "backend": "cuda|wgpu|vulkan|cpu",
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

All baselines calibrated against **yoga** (RTX 4060 Laptop). To be updated after 5 consecutive nightly runs with <5% variance (PMAT-424).

| Canary | Metric | Baseline | Tolerance | Gate |
|--------|--------|----------|-----------|------|
| unsloth (yoga) | tokens_per_sec | 6,600 | -10% | >= 5,940 |
| unsloth (yoga) | peak_vram_mb | 3,600 | +5% | <= 3,780 |
| unsloth | final_loss | 2.0 | -- | <= 2.0 |
| pytorch | tokens_per_sec | 3,000 | -10% | >= 2,700 |
| pytorch | peak_vram_mb | 8,000 | +5% | <= 8,400 |
| pytorch | final_loss | 2.0 | -- | <= 2.0 |
| cublas | tokens_per_sec | 3,000 | -10% | >= 2,700 |
| cublas | loss_divergence | 0.01 | -- | <= 0.01 |
| cublas | throughput_ratio | 0.95 | -- | >= 0.95 |
| wgpu | tokens_per_sec | 100 | -10% | >= 90 |
| wgpu | peak_vram_mb | 7,000 | +5% | <= 7,350 |
| wgpu | final_loss | 2.5 | -- | <= 2.5 |

**Baseline update policy:** After 5 consecutive nightly runs with <5% variance, update to the median observed value. Floor to nearest 100 for tok/s, ceil to nearest 100 for VRAM.

> **F-BL-01:** If first 5 yoga runs show >5% variance, baselines cannot be established. Action: investigate clock locking, thermal, background processes before proceeding.

### Expected Throughput (Yoga Primary)

| Canary | Runtime | yoga (8GB) | gx10 (120GB) | intel (8GB) |
|--------|---------|-----------|-------------|------------|
| **apr** | entrenar (Rust) | **36** (CUDA fallback to CPU, 186x deficit) | building | TBD (wgpu) |
| unsloth | Python + bitsandbytes | **6,697** (measured) | **13,660** (measured) | N/A |
| pytorch | Python + torch | N/A (F-EXEC-02) | **4,055** (measured) | N/A |
| cublas | Python + torch | N/A (F-EXEC-02) | **4,010/4,027** | N/A |
| wgpu | burn (Rust) | N/A | N/A | **6,730** (measured, hidden=1536, Vulkan) |

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
| F-EXEC-01 | Canaries detect 10% regressions | Inject 15% slowdown on yoga -> must FAIL | P0 |
| F-EXEC-02 | Full FT fits 8GB at batch=4 seq=512 | OOM on yoga -> reduce batch or add grad ckpt | P0 |
| F-HW-01 | Locked clocks give <5% variance | 10x runs on yoga, if variance >5% -> clocks broken | P0 |
| F-HW-02 | WGPU training feasible on W5700X | burn-canary crash or 0 tok/s -> not ready | P1 |
| F-WL-01 | Unsloth faster than raw PyTorch | unsloth_tok_s < pytorch_tok_s -> overhead > savings | P0 |
| F-WL-02 | PyTorch baseline >3k tok/s on yoga | <3k -> investigate PyTorch/cuDNN/bf16 | P0 |
| F-WL-03 | cuBLAS parity (divergence <0.01) | >0.01 -> GEMM precision issue | P0 |
| F-WL-04 | cuBLAS test is meaningful | ratio=1.0000 exactly -> TF32 flag not effective | P1 |
| F-WL-05 | WGPU deployment works | binary missing or VRAM >7GB -> broken | P1 |
| F-BL-01 | Baselines achievable in 5 runs | >5% variance after 5 runs -> environment issue | P0 |
| F-MET-01 | Metrics schema valid | JSON validation failure -> fix canary output | P0 |
| F-SC-01 | Scoring logic correct | score-gate passes bad result -> fix scoring | P0 |

### Falsified Claims

| ID | Claim | Date | What Happened | Resolution |
|----|-------|------|---------------|------------|
| F-EXEC-02 | Full FT fits 8GB at batch=4 seq=512 | 2026-03-31 | OOM even at batch=2 + 8-bit optimizer + gradient checkpointing. Model (3.5GB) + gradients (3.5GB) = 7GB floor. | pytorch/cublas deferred to gx10. Yoga runs unsloth only. |
| F-RD-01 | torch.compile +20-40% throughput | 2026-03-31 | -11.3% regression (3,598 vs 4,055 tok/s). Compilation cost (~90s) amortized over only 100 steps = net loss. | torch.compile not suitable for canary-length runs. Would help at >1000 steps. |
| F-HW-01 | Locked clocks <5% variance | 2026-03-31 | CONFIRMED: 0.34% variance across 5 runs on yoga. | Baseline methodology validated. |
| F-WL-03 | cuBLAS parity <0.01 | 2026-03-31 | CONFIRMED: 0.000000 divergence on gx10. Perfect parity. | GEMM backends numerically identical on Blackwell. |
| F-WL-06 | apr throughput vs unsloth | 2026-03-31 | Three failures: (1) yoga sm_89: CPU fallback 36 tok/s [trueno#231]. (2) gx10 sm_121: NaN/Inf loss [aprender#563]. (3) intel: 147 compile errors without CUDA [aprender#564]. Contract: cuda-training-forward-v1.yaml (5/5 falsified). | Tracked: paiml/trueno#231, paiml/aprender#563, paiml/aprender#564. |

### Falsification Protocol

1. **Before accepting any baseline:** Run the falsification condition. A claim that has never been tested is an assumption, not a fact.
2. **On falsification:** Update this register, revise or retract the claim, open a PMAT item for the fix.
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
| **Total** | | **19** |

See [components/optimization-roadmap.md](components/optimization-roadmap.md) for full phase details.

### Quality Gates

| Gate | Tool | Threshold |
|------|------|-----------|
| Canary pass/fail | `make score` | All yoga canaries PASS |
| CI gate | `make score-gate` | Exit 0 |
| Nightly regression | `scripts/nightly.sh` | Yoga passes first, then secondaries |

---

## 9. Revision History

| Version | Date | Changes | PMAT |
|---------|------|---------|------|
| 1.0.0 | 2026-03-31 | Initial spec: 4 canaries, 3 hardware targets, 19 PMAT items | PMAT-420 |
| 2.0.0 | 2026-03-31 | Refactor: 500-line cap, component specs, falsification-first, yoga primary | PMAT-420 |
