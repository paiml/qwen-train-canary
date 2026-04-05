# Training Profiler Specification

**Parent:** [Training Canary Spec](../training-canary-spec.md) Section 8
**PMAT:** PMAT-504 (Phase A contracts), PMAT-506 (live contract enforcement), PMAT-496 (overhead), PMAT-480 (BrickProfiler), PMAT-483 (per-op), PMAT-499 (unified interface)
**Status:** ACTIVE — v2 contracts-enforced (v6.14.0)

---

## Priority: Phase A Deliverables (PMAT-504)

The profiler is only valuable if it catches regressions automatically. Phase A has three deliverables, in priority order:

### P0.1 — Provable-Contract Invariants (SHIPPED v6.14.0)

Seven compiler-enforced contracts wired into `score.py` + `canaries/apr/train.py` runner. Every APR canary result is scored against these contracts; the canary exits non-zero if any contract fails (PMAT-506).

| # | Contract | Rule | Rationale |
|---|----------|------|-----------|
| 1 | `convergence` | `final_loss <= 2.0` (CUDA) or `<= 2.5` (WGPU) | F-CONV-01: model must converge |
| 2 | `better_than_random` | `final_loss < ln(vocab_size)` (11.93 for Qwen2.5) | F-CONV-02: worse than random = broken |
| 3 | `loss_improved` | `min(loss_trajectory) < 0.8 * trajectory[0]` | F-CONV-03: >=20% improvement somewhere — distinguishes learning-but-oscillating from stuck |
| 4 | `backward_executed` | `valid_backward_steps > 0` | F-BWD-01: training must happen |
| 5 | `metrics_quality` | `_metrics_quality == "measured"` | F-MET-02: loss parsed, not estimated |
| 6 | `config_steps` | `steps >= 100` | F-CFG-01: warm-up drift prevention |
| 7 | `step_time_sanity` | `avg_fwd + avg_bwd < 10000ms` | F-PROF-STEP: catch hangs/leaks |

**Live application:** Applied to all 7 historical APR results. **2 of 7 pass `better_than_random`** (async + profile at loss=11.74). The other 5 (losses 12.8-16.8) never crossed random baseline → correctly flagged as regressions. **F-CONV-03 retrofitted:** async/profile pass (ratio 0.484 = 52% improvement), pretok passes (ratio 0.745 = 26%), runs with <2 epochs skipped.

> **F-REGRESS-01 (TRIGGERED 2026-04-05):** The gx10 binary regressed silently between 10:29 (loss=11.74 working) and 11:39 (loss=100 NaN sentinel). Contracts exist but weren't applied to live results. **Fix (PMAT-506 SHIPPED):** `canaries/apr/train.py` now calls `score_result()` after every run and exits non-zero on contract failure.

### P0.2 — Port StepProfiler to CUDA Path (NOT STARTED)

The 13-phase WGPU profiler (`wgpu_pipeline.rs`) has 100% wall_coverage. The CUDA path (`CudaNf4TransformerBlock`) has **zero StepProfiler instrumentation** — we can't measure where CUDA training time goes, blocking Phase B (cuBLAS hybrid).

**Deliverable:** Wire `begin_phase`/`end_phase` into CUDA forward + backward with identical phase decomposition (`gpu_fwd`, `gpu_lm`, `gpu_ce`, `gpu_lm_bwd`, `gpu_lora_bwd`). Report per-phase ms via the same JSON schema.

**Exit criterion:** `apr finetune --gpu-backend cuda --profile` produces `wall_coverage >= 0.85` with 13 phases identical to WGPU output.

### P0.3 — Profiler Fidelity Invariants (candle-vs-apr Insights)

candle-vs-apr found BrickProfiler's `Deferred` sync mode had **3.4x fidelity error**. Apply the same invariants to StepProfiler:

| # | Contract | Rule | Status |
|---|----------|------|--------|
| 7 | `lmhead_executed` | `gpu_lm_ms > 0` | **SHIPPED v6.15.0**. Simplified form — full `lmhead_vs_rmsnorm_ratio` (>10x) requires rmsnorm as its own phase, deferred to CUDA StepProfiler port. |
| 8 | `gemm_dominance` | `gemm_pct >= 30%` | **SHIPPED** (preexisting F-POP-002). Architecture regression detection. |
| 9 | `no_orphan_spans` | All phases with total_ms > 0 must have avg_ms > 0 | **SHIPPED v6.15.0**. Trace corruption detection. |

---

## Historical: FALSIFIED Hypothesis (2026-04-05)

APR WGPU training on gx10 GB10 was originally measured at **421 tok/s wall-clock** with **24,094 tok/s step-level GPU compute** — suggesting 99% GPU idle time. The hypothesis was buffer allocation overhead (F-PROF-002).

**FALSIFIED (2026-04-05):** After async pipeline deployment, the profiler measured **100% GPU compute** with **0% overhead**. The 11x gap vs unsloth (470 vs 5,262 tok/s) is real WGPU compute shader speed, not inter-step overhead. `gpu_lora_bwd` at 55.7% (551ms/step) and `gpu_fwd` at 40.6% (401ms/step) dominated. Zero allocations, zero sync, zero buf_write per step. Profiler captured 395.7s of 449.6s training (88% coverage).

**NOTE (2026-04-05):** The binary producing this data is no longer deployed — see F-REGRESS-01. Current gx10 binary produces `loss=100` NaN sentinel via a different code path that lacks the async pipeline.

### Prior Art

**Chopper** (arXiv:2512.08242) demonstrates the correct methodology: multi-level decomposition (kernel → operation → layer → phase → iteration → GPU) applied to LLM training. Their finding that DVFS overhead dominates on MI300X parallels our finding that buffer allocation overhead dominates on WGPU.

**EROICA/PerfTracker** (arXiv:2506.08528) provides a priority taxonomy for root-cause localization: GPU compute > memory ops > collectives > Python overhead. Our profiler adapts this to WGPU: GPU compute > buffer alloc > queue submit > CPU sync > data prep.

**STAlloc** (arXiv:2507.16274) demonstrates the fix: static memory pool with pre-allocated address ranges eliminates cudaMalloc/cudaFree overhead, reducing fragmentation by 79.2%. The WGPU equivalent is pre-allocated buffer pools, as Burn discovered independently (burn-compute chunk/slice architecture reduced peak memory 40-50%).

**CPU-Induced Slowdowns** (arXiv:2603.22774) traces how CPU oversubscription creates GPU idle time via kernel launch latency — directly applicable to our WGPU `queue.submit()` overhead.

**Time is Not Compute** (arXiv:2603.28823) derives wall-clock scaling laws showing dual U-shape: short training budgets are compute-bottlenecked, long budgets are data-bottlenecked. Our 100-step canary sits in the short-budget regime where overhead dominates.

**Measuring GPU Utilization One Level Deeper** (arXiv:2501.16909) goes beyond SM-occupancy to decompose idle time into per-resource contention — the methodology our profiler must adopt for WGPU dispatch overhead.

### Gap in Literature

No arxiv papers address WGPU/Vulkan compute profiling for ML training workloads (TapML arXiv:2404.09151 covers inference only). This profiler fills that gap by adapting CUDA-era decomposition (Chopper, PerfTracker) to wgpu timestamp queries and Instant-based CPU phase timing.

## 2. Decision: `apr finetune --profile` as Central Interface

Profiling lives inside the training binary because inter-step overhead can ONLY be measured from inside the training loop. The `[PROFILE]` infrastructure already exists in `wgpu_pipeline.rs` — it captures GPU dispatch time but misses the 99% CPU/sync overhead.

**Rejected alternatives:**
- `cgp profile training` — layer violation (cgp is trueno, profiling needs entrenar internals)
- External profiler (Tracy, Nsight) — can't decompose Rust-side overhead without source instrumentation; Tracy integration adds dependency; Nsight is NVIDIA-only
- `probar score training` — probar is GUI testing; training scoring is a role mismatch

**Measurement pipeline:** `apr --profile` (measurement) → `canary runner` (contract gating, PMAT-506) → `score.py` (cross-run comparison) → `spec` (analysis)

### 2.1 Live Contract Enforcement (PMAT-506, SHIPPED v6.14.0)

The canary runner (`canaries/apr/train.py`) now calls `score_result()` on its own output JSON after every run and exits non-zero on contract failure. This catches silent regressions at measurement time — not hours later during spec review.

```python
# canaries/apr/train.py (simplified)
score = score_result(output, baseline)
for check, info in score["checks"].items():
    status = "PASS" if info["pass"] else "FAIL"
    print(f"  [{status}] {check}: {info['value']}")
if not score["pass"]:
    sys.exit(1)  # CI gate fires here
```

**Why this matters:** The gx10 binary silently regressed between 10:29 (loss=11.74) and 11:39 (loss=100) without any alarm. With live contracts, the 11:39 run would have failed 4 contracts (loss, convergence, better_than_random, backward_executed) and exited non-zero immediately.

## 3. Profiling Phases

### 3.1 Hierarchical Decomposition

Following Chopper's multi-level methodology (arXiv:2512.08242), the training step decomposes into phases that sum to 100% of wall time. Every nanosecond between `step_start` and `step_end` is attributed to exactly one phase.

| Phase | What | Current (gx10) | Category |
|-------|------|-----------------|----------|
| `data_prep` | Tokenize + build input_ids + prompt/response split | 0.2ms (0.0%) | CPU |
| `embed` | CPU embedding lookup (vocab × hidden matmul) | 0.0ms (0.0%) | CPU |
| `buf_alloc` | `create_buffer` + `zeros()` calls | 0.0ms (0.0%) | Memory |
| `buf_write` | `queue.write_buffer` (CPU → GPU staging) | 0.0ms (0.0%) | Transfer |
| `fwd_encode` | Command encoder recording for 28-layer forward | 0.0ms (0.0%) | CPU |
| `fwd_submit` | `queue.submit()` for forward pass | 0.0ms (0.0%) | Dispatch |
| `gpu_fwd` | GPU compute: 28-layer forward + LoRA inline | **401ms (40.6%)** | Compute |
| `gpu_lm` | GPU compute: RMSNorm + lm_head GEMM | 17ms (1.7%) | Compute |
| `gpu_ce` | GPU compute: cross-entropy fwd + bwd | 0.1ms (0.0%) | Compute |
| `gpu_lm_bwd` | GPU compute: lm_head backward GEMM | 19ms (2.0%) | Compute |
| `gpu_lora_bwd` | GPU compute: LoRA backward + AdamW | **551ms (55.7%)** | Compute |
| `sync` | `device.poll` / `map_async` — GPU→CPU loss readback | 0.0ms (0.0%) | Sync |
| `overhead` | Residual (Rust runtime, loop bookkeeping) | 0.0ms (0.0%) | Other |

**Measured 2026-04-05 on gx10 GB10 (async pipeline).** 400 steps, 395.7s profiler wall, 100% GPU compute. All overhead phases eliminated by async device.poll.

**Priority taxonomy** (adapted from PerfTracker arXiv:2506.08528):
Compute → Memory → Transfer → Dispatch → Sync → CPU → Other

### 3.2 Allocation Tracking

Per-step count and cumulative time for buffer operations (motivated by STAlloc arXiv:2507.16274 and GMLake arXiv:2401.08156 showing allocation overhead as primary throughput bottleneck):

| Counter | What |
|---------|------|
| `alloc_count` | Number of `create_buffer` / `zeros()` calls per step |
| `alloc_time_ms` | Cumulative time in buffer allocation per step |
| `write_count` | Number of `queue.write_buffer` calls per step |
| `write_time_ms` | Cumulative time in buffer writes per step |
| `submit_count` | Number of `queue.submit()` calls per step |
| `submit_time_ms` | Cumulative time in queue submission per step |

### 3.3 Per-Layer Timing (Optional)

When `--profile-layers` is set, report per-layer forward and backward timing:

```
layer_fwd_ms: [l0_ms, l1_ms, ..., l27_ms]
layer_bwd_ms: [l0_ms, l1_ms, ..., l27_ms]
```

## 4. Provable Contracts

### 4.1 Contract: `profiler-wall-coverage-v1.yaml`

```yaml
metadata:
  version: 1.0.0
  created: '2026-04-05'
  author: PAIML Engineering
  description: >
    Wall coverage contract — ensures profiler instrumentation accounts
    for >= 95% of measured step time across all training phases.
    Methodology: Chopper multi-level decomposition (arXiv:2512.08242).
    Measurement: Instant-based wall-clock timing (Hoefler & Belli SC'15).
  references:
    - 'arXiv:2512.08242 — Chopper: Multi-Level GPU Characterization'
    - 'arXiv:2506.08528 — EROICA/PerfTracker: Online Performance Troubleshooting'
    - 'arXiv:2501.16909 — Measuring GPU Utilization One Level Deeper'
  registry: false
  enforcement_level: standard

equations:
  wall_coverage:
    formula: wall_coverage = sum(phase_times[0..N]) / wall_clock_step_time
    domain: phase_times > 0 (microseconds), wall_clock_step_time > 0
    codomain: '[0, 1]'
    invariants:
      - wall_coverage >= 0.95 (phases account for >=95% of wall time)
      - wall_coverage <= 1.0 (phases are subsets of wall time)
    preconditions:
      - all phase times >= 0
      - all phase times finite (not NaN/Inf)
      - wall_clock_step_time > 0
    postconditions:
      - bottleneck phase identified (max pct)
      - sum(phase_pct) in [0.95, 1.00]

proof_obligations:
  - type: bound
    property: wall_coverage >= 0.95
    formal: sum(phase_times) / wall_clock >= 0.95
    tolerance: 0.02

falsification_tests:
  - id: F-PROF-001
    rule: wall_coverage >= 0.95
    prediction: >
      With 13 phases instrumented, sum of phase times accounts for
      >= 95% of measured wall-clock step time on a 10-step run.
    test: >
      Run 10-step training with profiler enabled on gx10.
      Compute wall_coverage = sum(phases) / wall_time.
      Assert wall_coverage >= 0.95.
    if_fails: >
      Missing instrumentation. An untracked code path consumes > 5%
      of wall time. Add phase to decomposition and re-measure.

  - id: F-PROF-003
    rule: profiler does not perturb measurement
    prediction: >
      Profiler overhead < 2% of uninstrumented wall time.
      Instant::now() cost ~25ns × 26 calls/step = ~650ns << 8850ms.
    test: >
      Run same workload with profiler enabled vs disabled.
      Assert wall_time_profiled / wall_time_unprofiled < 1.02.
    if_fails: >
      Profiler instrumentation introduces significant overhead.
      Reduce phase granularity or use cheaper timing.

kani_harnesses:
  - id: KANI-PROF-001
    obligation: wall_coverage in [0, 1]
    property: 0.0 <= wall_coverage && wall_coverage <= 1.0
    bound: 16
    strategy: stub_float
    solver: cadical
```

### 4.2 Contract: `profiler-bottleneck-classification-v1.yaml`

```yaml
metadata:
  version: 1.0.0
  created: '2026-04-05'
  author: PAIML Engineering
  description: >
    Bottleneck classification contract — validates the hypothesis that
    buffer allocation dominates WGPU inter-step overhead (PMAT-496).
    Uses falsification-first methodology: each hypothesis has a
    falsification condition that triggers investigation of alternatives.
  references:
    - 'arXiv:2507.16274 — STAlloc: Reducing GPU Memory Fragmentation'
    - 'arXiv:2401.08156 — GMLake: GPU Memory Defragmentation'
    - 'arXiv:2407.12117 — Memo: Fine-grained Tensor Management'
    - 'arXiv:2603.22774 — CPU-Induced Slowdowns in Multi-GPU Inference'
    - 'Burn-Compute: High Performance Async Backends (burn.dev/blog)'
  registry: false
  enforcement_level: standard

equations:
  alloc_ratio:
    formula: alloc_ratio = alloc_time / wall_clock_step_time
    domain: alloc_time >= 0, wall_clock > 0
    codomain: '[0, 1]'
    invariants:
      - If alloc_ratio >= 0.50, buffer allocation is the dominant bottleneck
      - If alloc_ratio < 0.10, buffer allocation is NOT the bottleneck

  gpu_compute_ratio:
    formula: gpu_ratio = sum(gpu_phases) / wall_clock_step_time
    domain: gpu_phases >= 0, wall_clock > 0
    codomain: '[0, 1]'
    invariants:
      - If gpu_ratio > 0.20, GPU IS compute-bound (contradicts PMAT-496)
      - If gpu_ratio < 0.05, GPU is severely underutilized

  batch_scaling_efficiency:
    formula: scaling = (tok_s_batch16 / tok_s_batch4) / 4.0
    domain: tok_s > 0
    codomain: '[0, 1]'
    invariants:
      - If scaling < 0.50, overhead dominates (batch increase doesn't help)
      - If scaling > 0.80, compute dominates (batch increase scales linearly)

proof_obligations:
  - type: bound
    property: alloc_ratio classification correct
    formal: (alloc_ratio >= 0.50) implies bottleneck == "buf_alloc"
    tolerance: 0.05

  - type: bound
    property: gpu_ratio < 0.20 (PMAT-496 hypothesis)
    formal: sum(gpu_phases) / wall_clock < 0.20
    tolerance: 0.05

falsification_tests:
  - id: F-PROF-002
    rule: buffer allocation is the bottleneck
    status: FALSIFIED (2026-04-05)
    prediction: >
      alloc_time >= 50% of wall-clock per step.
    result: >
      FALSIFIED: alloc_ratio = 0.000. Zero allocations per step.
      gpu_compute_pct = 100%. Async pipeline pre-allocation solved
      allocation overhead entirely. Bottleneck is GPU compute dispatch
      speed (gpu_lora_bwd 55.7%), not buffer allocation.

  - id: F-PROF-004
    rule: GPU compute is NOT the bottleneck
    status: FALSIFIED (2026-04-05)
    prediction: >
      gpu_compute_pct < 5% of wall time.
    result: >
      FALSIFIED: gpu_compute_pct = 100.0%. After async pipeline,
      GPU IS the bottleneck. The original 0.96% was measured on
      sync pipeline where 98.3% was inter-step overhead. Async
      pipeline moves device.poll into step, revealing true GPU time.

  - id: F-PROF-005
    rule: pre-allocation fixes the bottleneck
    status: FALSIFIED (2026-04-05)
    prediction: >
      After implementing buffer pool (ensure_training_activations +
      reuse zeros), wall-clock throughput improves >= 2x.
    result: >
      FALSIFIED: async pipeline (v6.7.0) achieved 0 allocs/step
      (measured alloc_count=0, wall_coverage=1.000). Wall-clock
      throughput improved from 421 to 470 tok/s = +12%, NOT 2x.
      Pre-allocation solved allocation overhead entirely but did
      NOT solve the throughput gap — bottleneck is GPU compute
      dispatch speed (gpu_lora_bwd 55.7%), not allocation.

  - id: F-PROF-006
    rule: batch scaling confirms overhead-dominated regime
    status: OBSOLETE (2026-04-05)
    note: >
      Batch scaling hypothesis was relevant under sync pipeline
      where 98.3% of wall was inter-step idle. Async pipeline
      moved device.poll into step, making GPU compute 100% of
      wall. Batch scaling efficiency on async pipeline has not
      been re-measured (batch=16 not re-run). If re-measured,
      efficiency should now be >0.50 (compute-bound regime).

kani_harnesses:
  - id: KANI-PROF-002
    obligation: alloc_ratio in [0, 1]
    property: 0.0 <= alloc_ratio && alloc_ratio <= 1.0
    bound: 8
    strategy: stub_float
    solver: cadical

qa_gate:
  id: C-PROF-002
  name: Bottleneck Classification
  checks:
    - alloc_ratio computed and classified
    - gpu_compute_ratio < 0.20
    - batch_scaling_efficiency measured
  pass_criteria: >
    All falsification tests produce clear pass/fail.
    Bottleneck correctly identified and actionable fix proposed.
```

## 5. Output Schema

`apr finetune --profile /tmp/profile.json` writes a JSON report after training:

```json
{
  "_profiler": "apr_training_profiler_v1",
  "_version": "1.0.0",
  "steps_profiled": 400,
  "wall_time_ms": 395661.2,
  "phases": {
    "data_prep":    { "total_ms": 78.5,     "pct": 0.0,  "avg_ms": 0.2 },
    "embed":        { "total_ms": 0.0,      "pct": 0.0,  "avg_ms": 0.0 },
    "buf_alloc":    { "total_ms": 0.0,      "pct": 0.0,  "avg_ms": 0.0 },
    "buf_write":    { "total_ms": 11.0,     "pct": 0.0,  "avg_ms": 0.0 },
    "fwd_encode":   { "total_ms": 0.0,      "pct": 0.0,  "avg_ms": 0.0 },
    "fwd_submit":   { "total_ms": 0.0,      "pct": 0.0,  "avg_ms": 0.0 },
    "gpu_fwd":      { "total_ms": 160613.9, "pct": 40.6, "avg_ms": 401.5 },
    "gpu_lm":       { "total_ms": 6901.3,   "pct": 1.7,  "avg_ms": 17.3 },
    "gpu_ce":       { "total_ms": 34.1,     "pct": 0.0,  "avg_ms": 0.1 },
    "gpu_lm_bwd":   { "total_ms": 7770.1,   "pct": 2.0,  "avg_ms": 19.4 },
    "gpu_lora_bwd": { "total_ms": 220252.2, "pct": 55.7, "avg_ms": 550.6 },
    "sync":         { "total_ms": 0.0,      "pct": 0.0,  "avg_ms": 0.0 },
    "overhead":     { "total_ms": 0.0,      "pct": 0.0,  "avg_ms": 0.0 }
  },
  "wall_coverage": 1.000,
  "bottleneck": "gpu_lora_bwd",
  "bottleneck_pct": 55.7,
  "alloc_stats": {
    "count_per_step": 0,
    "total_alloc_ms": 0.0,
    "avg_alloc_us": 0.0
  },
  "submit_stats": {
    "count_per_step": 0,
    "total_submit_ms": 0.0
  },
  "write_stats": {
    "count_per_step": 0,
    "total_write_ms": 0.0
  },
  "gpu_compute_ms": 395571.6,
  "gpu_compute_pct": 100.0
}
```

**Source:** `results/canary-apr-gx10-async-20260405.json` — measured on gx10 GB10, async WGPU pipeline, 8 epochs over 50 samples.

## 6. Cross-Project Insights (candle-vs-apr, added PMAT-500)

**Source:** `~/src/candle-vs-apr` — inference benchmark sister project (candle vs apr/realizr). Five-whys chain-of-thought analysis identified patterns directly applicable to training profiler.

### 6.1 Profiler Fidelity Gap (candle-vs-apr F10)

candle-vs-apr found BrickProfiler's `Deferred` sync mode had **3.4x fidelity error** on QkvProjection (reported 26µs vs actual 89µs, `performance.md` lines 340-361). The `apr profile` command conflated pipeline efficiency (includes idle between kernels, 83.8% overhead) with per-kernel efficiency. Fix: subtract `kernel_launch_overhead_pct` from roofline calculation (aprender c0953fd7).

**For training profiler:** Report per-layer timing **excluding launch overhead** separately from wall-clock throughput. Use `Immediate` sync mode by default. Add: `BrickProfiler.reported_time > 0.5x actual_time` invariant.

### 6.2 Contract Wiring Gap (candle-vs-apr Phase 15) — **SHIPPED v6.14.0**

candle-vs-apr discovered **11 YAML contract definitions with zero wired to profiler code** — causing SwiGLU graph recording bug (28 missing kernels) to go undetected for a week. Five-whys root cause: no `#[contract(...)]` macros enforce invariants at runtime.

**Status:** 7 contracts shipped — see **P0.1** at top of spec. Live enforcement via `canaries/apr/train.py` + `score.py`. 71 tests passing. 8 historical APR results scored (`results/contract-classification-20260405.json`). Contracts distinguish: **4/8 REGRESSED** (loss >= 11.93 random), **2/8 NOT-CONVERGED** (loss 11.74, < random, > 2.0 threshold), **2/8 CRASHED** (yoga PMAT-498 + gx10 PMAT-494 routing regression, 0 backward steps each).

### 6.3 Event-Based Sync (candle-vs-apr F12)

`compute_stream.synchronize()` cost 12.9% throughput in inference (20.1→22.7 tok/s, ITL 49.7→44.0ms). Fixed with `cuStreamWaitEvent`. The WGPU equivalent: explicit `vkCmdPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)` instead of implicit Vulkan synchronization.

**For PMAT-498 (WGPU crash after 5 steps):** Check burn-canary for explicit pipeline barrier placement after compute dispatches. Add monotonic frame counter contract to catch lost dispatches.

### 6.4 Trueno-Parity Canary (new, from chain-of-thought)

candle-vs-apr's cuBLAS parity gate (identical loop, different GEMM backend) directly inspires a **trueno-parity canary**: compare trueno NF4 GEMM output vs pure PyTorch matmul on isolated layers to diagnose PMAT-497 convergence defect.

> **F-XPROJECT-01 (CONFIRMED 2026-04-05):** The 6 wired contracts caught **F-REGRESS-01** on day 1 — gx10 binary regressed silently from loss=11.74 to loss=100, caught by contracts applied to live results. Contract enforcement pays for itself.

---

## 7. Remaining Deliverables (Priority Order)

### P0 (Phase A blockers)

| # | Deliverable | Status | Exit Criterion |
|---|-------------|--------|----------------|
| 1 | 6 provable contracts wired | **SHIPPED v6.14.0** | 64 tests passing |
| 2 | Live canary contract gating | **SHIPPED v6.14.0** (PMAT-506) | canary exits non-zero on violation |
| 3 | Port StepProfiler to CUDA path | **NOT STARTED** (blocked by F-ECOSYSTEM-01) | CUDA run produces 13-phase JSON with wall_coverage >= 0.85 |
| 4 | Profiler fidelity invariants (7-9) | **SHIPPED** (2 new + 1 preexisting, v6.15.0) | 3 contracts in score.py: lmhead_executed, gemm_dominance, no_orphan_spans |
| 5 | Trueno-parity canary | **DESIGNED** | Compare trueno GEMM vs PyTorch matmul on isolated layer, loss divergence < 0.01 |

### P1 (Phase B/C support)

| # | Deliverable | Rationale |
|---|-------------|-----------|
| 6 | Per-layer profiling (`--profile-layers`) | Identify slowest transformer layer (28 total) |
| 7 | GPU timestamp queries (not Instant::now) | F-PROF-008: measure actual GPU time, not dispatch wrapper |
| 8 | Roofline classification (AI vs ridge) | F-PROF-010: memory-bound vs compute-bound per op |
| 9 | Dispatch gap profiling | F-PROF-016: time between queue.submit() and first kernel |
| 10 | Per-step variance | F-PROF-014: detect thermal throttling, GC interference |

### P2 (Future — when Phase C complete)

- Unified `trait ComputeProfiler` (PMAT-499) — abstract across WGPU/CUDA/cuBLAS/CUTLASS/SIMD
- OTLP / Chrome Trace output formats
- Integration with qwen-coder-deploy inference profiler (cross-project comparison)

**Measured 2026-04-05 on gx10 GB10 (async pipeline, 400 steps, 8 epochs).** F-PROF-002 FALSIFIED: `buf_alloc` is 0%, not 82.6%. GPU compute is 100% of profiled wall time. Bottleneck is `gpu_lora_bwd` (55.7%) — 784 WGPU compute shader dispatches per step (7 projections × 28 layers × 4 ops each). The 11x gap vs unsloth (470 vs 5,262 tok/s) is real GPU compute speed, not overhead.

## 6. CLI Interface

```
apr finetune model.apr \
  --method qlora --quantize-nf4 \
  --data train.jsonl \
  --epochs 1 \
  --profile /tmp/profile.json       # Enables full profiling, writes JSON
  --profile-steps 5                  # Only profile first N steps (default: all)
  --profile-layers                   # Include per-layer timing (verbose)
```

When `--profile` is set:
1. Wrap every phase boundary with `Instant::now()`
2. Wrap every `create_buffer` / `zeros()` with allocation counter
3. Wrap every `queue.submit()` with submission counter
4. After training, compute percentages + bottleneck classification
5. Write JSON to the specified path
6. Print one-line summary to stderr

**One-line summary:**
```
[PROFILE] 13 steps, 115.0s wall, bottleneck=buf_alloc (82.6%), gpu_compute=0.9%, coverage=97.4%
```

## 7. Canary Integration

The canary script (`canaries/apr/train.py`) already parses profiler output. Extensions:

1. Pass `--profile /tmp/canary-profile.json` when profiling is requested
2. Embed the profiler JSON in the canary result under `"profiler"` key
3. `score.py` validates: `wall_coverage >= 0.90` (F-PROF-001)
4. `score.py` validates: `bottleneck` field present and non-empty

## 8. Implementation Plan

### Phase 1: Instrument `wgpu_pipeline.rs` (entrenar)

Replace the current `t0..t5` timing with a `WgpuStepProfiler` struct (modeled on the CUDA `StepProfiler` at `src/train/transformer_trainer/step_profiler.rs`):

```rust
struct WgpuStepProfiler {
    enabled: bool,
    phases: [Duration; NUM_WGPU_PHASES],
    phase_start: Option<Instant>,
    step_start: Option<Instant>,
    alloc_count: u32,
    alloc_time: Duration,
    submit_count: u32,
    submit_time: Duration,
    // ... accumulation across steps
}
```

### Phase 2: Instrument `WgpuTrainer::zeros()` (entrenar)

Wrap every `zeros()` and `create_buffer` call with allocation timing.

### Phase 3: Wire into `apr finetune` CLI (aprender)

Add `--profile` flag. Pass through to entrenar's `WgpuInstructPipeline`.

### Phase 4: Canary + Score integration

Update `train.py` to pass `--profile` flag and parse output JSON.

### Phase 5: Provable contract registration

**Status:** Contracts EXIST but are NOT BOUND.

Three profiling contracts exist in `provable-contracts/contracts/entrenar/`:
1. `training-step-profiling-v1.yaml` — per-layer decomposition (forward=sum(layer_forward[i]), backward=sum(layer_backward[i])), 12 falsification tests (F-TSP-001 through F-TSP-012), roofline + kernel launch overhead equations
2. `per-operation-training-profiling-v1.yaml` — per-op within each layer (5 forward GEMMs, 4 backward GEMMs, 2 norms, attention), invariant: gemm_time/layer_fwd >= 0.50
3. `kaizen/step-profiler-v1.yaml` — StepProfiler 11-phase contract for CUDA path

Additionally, `gpu-decode-profiling-v1.yaml` (v2.0.0) provides the inference analog: BrickProfiler with 15 invariants, SyncMode::Immediate/Deferred, and brick ordering proofs.

**Verification matrix (2026-04-05):**

| Contract | Implementation | Binding | Per-Layer | Per-Op | Any Model | JSON Output |
|----------|---------------|---------|-----------|--------|-----------|-------------|
| `training-step-profiling-v1` | StepProfiler (CUDA) | **NO** | **YES** (begin_layer/end_layer_fwd/bwd, MAX_LAYERS=64) | **YES** (16 ops) | **YES** (num_layers parametric) | **YES** (step_profiler_v2 JSON) |
| `per-operation-training-profiling-v1` | StepProfiler (CUDA) | **NO** | N/A | **YES** (16 ops: 9 fwd + 7 bwd) | **YES** (ops are architecture-independent) | **YES** (ops:{} in JSON) |
| `gpu-decode-profiling-v1` | BrickProfiler (trueno) | **YES** (binding.yaml) | **YES** (per-brick per-layer) | **YES** (23 BrickIds) | **YES** (BrickId enum is model-agnostic) | **YES** (JSON report) |
| WGPU training profiler | wgpu_pipeline.rs Instant::now | **NO** | **NO** (aggregate phases only) | **NO** (7 phase timestamps) | **YES** (uses num_layers) | **YES** (apr_training_profiler_v1 JSON) |

**Assessment: Layer-based tracing EXISTS for CUDA training and CUDA inference, but NOT for WGPU training.**

Chain of evidence:
1. **CUDA training path** (InstructPipeline → CudaNf4TransformerBlock): `StepProfiler` calls `begin_layer()`/`end_layer_fwd(i)`/`end_layer_bwd(layer_idx)` in `cuda_trainer.rs:1156-1657`. Per-op timing via `CudaBlockScratch::op_us[16]` fed through `record_layer_times()`. JSON output includes `per_layer:[{fwd_ms, bwd_ms}]` and `ops:{rmsnorm_attn, qkv_gemm, ...}`. Works for ANY model up to 64 layers.
2. **CUDA inference path** (realizar → trueno BrickProfiler): 23 BrickIds (RmsNorm, QkvProjection, AttentionScore, GateProjection, etc.), `SyncMode::Immediate` for real GPU timing, `SyncMode::Deferred` for low-overhead. BOUND in binding.yaml. Model-agnostic (BrickId enum covers standard transformer ops).
3. **WGPU training path** (WgpuInstructPipeline): Only `Instant::now()` around 7 aggregate phase boundaries (t0-t5). NO per-layer decomposition. NO per-op timing. The `[OP-TRACE]` line on yoga (layer.0 only) comes from a separate debug path, not the systematic profiler.

**Gap:** WGPU training has no per-layer profiling. The provable contracts (training-step-profiling-v1, per-operation-training-profiling-v1) describe what SHOULD exist but are NOT BOUND because the WGPU implementation doesn't have the instrumentation. The CUDA path DOES have it but can't run on gx10 (PTX JIT hang, PMAT-492).

**Next:** Port `StepProfiler::begin_layer()`/`end_layer_fwd()`/`end_layer_bwd()` into `WgpuInstructPipeline::train_step()` by wrapping `encode_forward_layer_training()` calls with layer-level timing. Then add bindings to `contracts/entrenar/binding.yaml`.

## 9. Profiler v2: Five Improvements (2026-04-05)

**Context:** The v1 profiler achieves 100% wall coverage with 13 phases and correctly identified `gpu_lora_bwd` (55.7%) and `gpu_fwd` (40.6%) as bottlenecks on gx10. But it cannot answer the critical question: **is the bottleneck slow shaders, memory bandwidth starvation at rank=16, or dispatch overhead from 784 small compute passes?** The following 5 improvements close that gap.

**Chain of thought:** (1) We measured WHERE time goes (phases) but not WHY. (2) The "why" requires three instruments we lack: GPU-side kernel timing, arithmetic intensity, and dispatch gap measurement. (3) Chopper (arXiv:2512.08242) showed that CPU-side timing conflates launch overhead with kernel execution — on MI300X, DVFS overhead was the dominant source, invisible to CPU timestamps. (4) Our 784 dispatches/step at batch=4 are in exactly the regime where launch overhead dominates (Chopper §5.2). (5) Without per-kernel roofline classification, we can't distinguish "fuse kernels" (dispatch-bound) from "optimize shaders" (compute-bound) from "reduce memory traffic" (bandwidth-bound).

### 9.1 GPU-side Timestamp Queries

**What:** Replace `Instant::now()` with `wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES` for GPU-side nanosecond timing around each compute dispatch.

**Why (chain of thought):** (1) Current CPU timing measures dispatch+wait, not kernel execution. (2) For async WGPU, CPU sees the full pipeline latency, not individual shader runtime. (3) The 551ms `gpu_lora_bwd` includes 784 dispatch_workgroups calls — we don't know if each takes 0.7ms (compute-bound) or 0.01ms compute + 0.69ms dispatch overhead. (4) GPU timestamps disambiguate. (5) The `wgpu-profiler` crate (github.com/Wumpf/wgpu-profiler) provides nested scopes with automatic query pooling and zero-stall deferred readback.

**Existing infrastructure:** trueno's `BrickProfiler` (`src/brick/profiler/mod.rs`, 447 lines) has `SyncMode::Immediate/Deferred` pattern and O(1) hot path via BrickId enums. Extend to WGPU timestamp queries.

> **F-PROF-008:** If GPU-side kernel time for `gpu_lora_bwd` matches CPU-side time (ratio 0.95-1.05), then dispatch overhead is negligible and the bottleneck is shader execution speed. Action: optimize WGSL shaders, not dispatch count.
> **F-PROF-009:** If GPU-side kernel time is <50% of CPU-side time, dispatch overhead dominates. Action: kernel fusion (PMAT-484) is the correct fix.

**Cite:** Chopper (arXiv:2512.08242) §4.2: "runtime traces alone cannot explain time differences... hardware performance counters needed." Found kernel launch overhead "relatively constant across configurations, causing it to occupy a larger percentage for small batch sizes."

### 9.2 Per-Kernel Arithmetic Intensity (Roofline Classification)

**What:** For each GEMM dispatch, compute arithmetic intensity `AI = 2*M*N*K / ((M*K + K*N + M*N) * 4)` FLOPs/byte and classify as compute-bound or memory-bound against device ridge point.

**Why (chain of thought):** (1) LoRA backward has rank=16, hidden=1536 — the B matrix is 16×1536 = 24K elements = 96 KB. (2) For a [seq=50, 16] × [16, 1536] GEMM: AI = 2×50×16×1536 / ((50×16 + 16×1536 + 50×1536)×4) = 2.4M / 380K = 6.3 FLOPs/byte. (3) GB10 Blackwell has ~900 GB/s bandwidth and ~100 TFLOPS, ridge point ~111 FLOPs/byte. (4) AI=6.3 << 111 → **LoRA backward is severely memory-bound**. (5) This means shader optimization is useless — need to reduce memory traffic (larger rank, fused reads, or batch multiple LoRA ops).

**Existing infrastructure:** trueno's `BrickStats` has per-brick counts. batuta's histogram infrastructure (886 lines, Prometheus-style buckets) can aggregate FLOPs/byte distributions.

> **F-PROF-010:** If LoRA backward AI > device ridge point, it IS compute-bound (contradicts the memory-bound hypothesis). Action: optimize WGSL shader, not memory access.
> **F-PROF-011:** If LoRA backward AI < 10% of ridge point, the operation is deeply memory-bound and no amount of shader optimization will help. Action: reduce memory traffic via batched LoRA or increased rank.

**Cite:** Omniwise (arXiv:2506.20886): empirical roofline achieves >98% kernel classification accuracy. "Measuring GPU Utilization One Level Deeper" (arXiv:2501.16909): per-resource contention decomposition beyond SM-occupancy.

### 9.3 Per-Layer Forward+Backward Timing

**What:** Wire layer index into WgpuStepProfiler phase timing. Output `layer_fwd_ms: [l0..l27]` and `layer_bwd_ms: [l0..l27]`.

**Why (chain of thought):** (1) The CUDA `StepProfiler` already has per-layer + per-op timing (16 ops, 28 layers). (2) The WGPU profiler only has 13 aggregate phases. (3) If layer 0 is 3x slower than layer 27 (due to RMSNorm shader JIT or cache effects), we're averaging away actionable information. (4) Per-layer timing enables: (a) detecting warmup effects, (b) identifying layers with anomalous projection sizes, (c) validating that all layers contribute equally to backward time. (5) Already spec'd in §3.3 but not implemented for WGPU.

**Existing infrastructure:** entrenar's `step_profiler.rs` (KAIZEN-047) measures 11 phases with per-layer arrays. Port to WGPU path.

> **F-PROF-012:** If per-layer variance < 5% (all layers within 5% of mean), layer-level profiling adds no actionable information beyond phase-level. Action: keep aggregate-only for WGPU (simpler, less overhead).
> **F-PROF-013:** If any single layer consumes >10% of total forward or backward time (vs expected 3.6% = 1/28), that layer has a specific bottleneck worth investigating.

**Cite:** PerfTracker (arXiv:2506.08528): "online performance troubleshooting requires per-operation localization." Chopper (arXiv:2512.08242): 7-level hierarchy includes per-layer decomposition.

### 9.4 Per-Step Variance + Anomaly Detection

**What:** Emit per-step phase times (not just cumulative totals). Compute coefficient of variation and flag outlier steps (>3x MAD).

**Why (chain of thought):** (1) Current profiler reports totals/averages over 400 steps. (2) If step 1 takes 17s (shader JIT) and steps 2-400 take 1s each, the average is misleading. (3) The yoga trace showed exactly this: RMSNorm 1671ms on step 1, ~50ms thereafter. (4) Thermal throttling on GB10 (passive cooling) could cause late-step slowdowns invisible to averages. (5) WGPU driver GC pauses (buffer deallocation) would appear as random spikes.

**Existing infrastructure:** renacer's process tracer has z-score anomaly detection with rate limiting. batuta's metrics has p50/p90/p99 histogram percentiles. Both patterns directly applicable.

> **F-PROF-014:** If coefficient of variation < 5% across all steps (excluding step 1 warmup), per-step variance adds no information. Action: averages are sufficient, skip per-step tracking.
> **F-PROF-015:** If p99/p50 ratio > 2.0 for any phase, there are significant outlier steps requiring investigation (thermal, GC, or driver).

**Cite:** EROICA (arXiv:2506.08528): 3D pattern vector (critical_path_occupancy, mean_utilization, stddev), anomaly threshold at 5x MAD. Chopper (arXiv:2512.08242): found DVFS frequency overhead was the dominant inefficiency on MI300X, exceeding MFMA utilization loss.

### 9.5 Dispatch Count + Inter-Dispatch Gap Profiling

**What:** Log `dispatch_count` per phase and measure inter-dispatch gaps using GPU timestamps between consecutive dispatches within the same compute pass.

**Why (chain of thought):** (1) We have 784 dispatches/step (7 projections × 28 layers × 4 ops) but zero visibility into the gap between them. (2) Each `dispatch_workgroups()` has WGPU encoder overhead (command buffer recording, bind group creation). (3) "CPU-Induced Slowdowns" (arXiv:2603.22774) shows CPU oversubscription creates GPU idle time via launch latency. (4) If inter-dispatch gaps are >1% of total pass time, fusing adjacent kernels (4 LoRA ops/layer → 1 fused dispatch) directly reduces overhead. (5) This is exactly the measurement needed for PMAT-484 (fused backward GEMM) — we can predict the speedup from fusion before implementing it.

**Existing infrastructure:** entrenar's WgpuStepProfiler already wraps `Instant::now()` around phases. Add a dispatch counter increment inside `encode_matmul` and `encode_lora_addmm`.

> **F-PROF-016:** If inter-dispatch gap time < 5% of total `gpu_lora_bwd` time, dispatch overhead is not the bottleneck and kernel fusion (PMAT-484) will NOT improve throughput significantly. Action: focus on shader optimization instead.
> **F-PROF-017:** If reducing dispatch count by 4x (fusion) reduces wall time by >20%, dispatch overhead IS significant. Action: prioritize PMAT-484 kernel fusion.

**Cite:** TritonForge (arXiv:2512.09196): profiling-guided kernel fusion achieves up to 5x improvement by eliminating inter-kernel launch latency. "CPU-Induced Slowdowns" (arXiv:2603.22774): kernel launch latency analysis.

---

## 10. References

| ID | Paper | Relevance | Used in |
|----|-------|-----------|---------|
| arXiv:2512.08242 | Chopper: Multi-Level GPU Characterization for LLM Training | Hierarchical decomposition (kernel→phase→iteration), DVFS overhead dominant | §3, §9.1, §9.3, §9.4 |
| arXiv:2506.08528 | EROICA/PerfTracker: Online Performance Troubleshooting | Priority taxonomy, 3D pattern vector anomaly detection (5x MAD) | §3, §9.3, §9.4 |
| arXiv:2507.16274 | STAlloc: Reducing GPU Memory Fragmentation | Pre-allocated memory pool (79% reduction), allocation overhead quantified | §3 (F-PROF-002 falsified) |
| arXiv:2401.08156 | GMLake: GPU Memory Defragmentation (ASPLOS'24) | Allocation overhead: 9.2 GB reclaimed | §3 |
| arXiv:2407.12117 | Memo: Fine-grained Tensor Management | Fragmentation → stall causal chain | §3 |
| arXiv:2603.22774 | CPU-Induced Slowdowns in Multi-GPU Inference | CPU oversubscription → GPU idle via launch latency | §3, §9.5 |
| arXiv:2603.28823 | Time is Not Compute: Scaling Laws for Wall-Clock Training | Dual U-shape: short budgets overhead-dominated | §3 |
| arXiv:2501.16909 | Measuring GPU Utilization One Level Deeper | Per-resource idle time beyond SM-occupancy | §3, §9.2 |
| arXiv:2410.07192 | PipeFill: Using GPUs During Bubbles | Pipeline bubble baselines (15-30%) | §3 |
| arXiv:2404.09151 | TapML: Emerging Platforms Meet Emerging LLMs | Vulkan compilation failures, platform testing | §1 |
| arXiv:2506.20886 | Omniwise: Predicting GPU Kernel Performance | Empirical roofline >98% accuracy, kernel classification | §9.2 |
| arXiv:2512.09196 | TritonForge: Profiling-Guided Kernel Optimization | Kernel fusion achieves up to 5x via launch elimination | §9.5 |
| arXiv:2407.08608 | FlashAttention-3: Fast Attention with Asynchrony | Attention optimization for Hopper, pipelining overlap | Tier 8 (roadmap) |
| arXiv:2503.19779 | PyGraph: CUDA Graph Capture for PyTorch Training | Graph capture eliminates launch overhead (6.5x) | Tier 7 (roadmap) |
| arXiv:2512.22219 | Mirage: Persistent Megakernel Optimization | Single-dispatch megakernel (1.7x) | Tier 9 (roadmap) |

## 9b. Bridging the PyTorch Profiler Gap (Five-Whys Analysis)

**Root cause chain:** (1) Can't compare APR to unsloth → (2) custom JSON vs Chrome Trace → (3) WGPU has no CUPTI → (4) WGPU HAS timestamp queries (unused) → (5) v1 focused on "where" not "why"

**How PyTorch measures:** CUPTI driver hooks → autograd graph linking → Chrome Trace Format → TensorBoard/Perfetto. Per-kernel GPU timing with nanosecond precision, zero training code modification.

**How unsloth measures:** HuggingFace Trainer callbacks → per-step `loss/lr/grad_norm` → WandB/TB export → `torch.cuda.max_memory_allocated()`. Simple wall-clock, no kernel profiling.

**How APR measures:** `Instant::now()` CPU timing → 13 phases → custom JSON → canary score.py. 100% wall coverage but CPU-side only, no per-kernel GPU timing, no per-step metrics, no standard export format.

### Five bridging improvements

| # | Gap | Fix | Falsification | Cite |
|---|-----|-----|---------------|------|
| 1 | No standard trace format | Chrome Trace export (`--profile-format chrome`) | Traces load in Perfetto alongside torch.profiler | Chopper (arXiv:2512.08242) uses Kineto traces |
| 2 | No per-step metrics | Emit loss/lr/grad_norm/step_ms per step (match Trainer.log) | APR loss curve matches unsloth within 0.1 on same data | PerfTracker (arXiv:2506.08528) per-op localization |
| 3 | No GPU-side kernel timing | WGPU timestamp queries via wgpu-profiler crate | GPU time matches CPU time → timestamps redundant | Chopper §4.2 |
| 4 | No memory waterfall | Track per-step WGPU alloc/peak/fragmentation | Peak matches planning estimate within 10% | STAlloc (arXiv:2507.16274) |
| 5 | No automated cross-runtime comparison | Parity scorecard: APR vs unsloth vs pytorch | All metrics within 10% = parity achieved | — |

## 10. Unified Profiling Interface (PMAT-499)

### 10.1 Problem: 18 Profilers, 0 Abstraction

Audit across 8 paiml repos found **18 distinct profilers** with no shared interface:

| Category | Backend-Specific | Abstract |
|----------|-----------------|----------|
| trueno | BrickProfiler (SyncMode) | AsyncTaskProfiler, BlisProfiler |
| entrenar | StepProfiler (CUDA), GpuProfiler | — |
| renacer | GpuTracerConfig (wgpu) | BrickTracer (OTLP), HPUProfiler, OtlpExporter |
| realizar | GpuProfile (CUDA cc dispatch) | ProfileReport (CPU wall-clock) |
| aprender/cgp | NcuProfiler, NsysProfiler, QuantProfiler | ProfilingCollector |
| batuta | — | Histogram (Prometheus) |

**Five-whys:**
1. Why can't we compare WGPU training profiler to CUDA training profiler? → Different structs, different output formats
2. Why different structs? → Each was built for one backend: StepProfiler for CUDA, Instant::now for WGPU
3. Why no shared trait? → Profilers were built bottom-up from implementation needs, not top-down from a contract
4. Why no contract? → The contracts exist (training-step-profiling-v1.yaml) but describe OUTPUT, not INTERFACE
5. **ROOT CAUSE:** No `trait ComputeProfiler` that abstracts over backend. Each profiler talks to its own GPU API directly.

### 10.2 Design: `trait ComputeProfiler`

One trait, implemented per backend. All paiml profiling converges here.

```rust
/// Unified profiling interface — PMAT-499
/// Backends: WGPU, CUDA/cuBLAS, CUTLASS, SIMD (CPU)
pub trait ComputeProfiler: Send + Sync {
    /// Backend identifier (for output tagging)
    fn backend(&self) -> ComputeBackend;

    // === Lifecycle ===
    fn begin_step(&mut self);
    fn end_step(&mut self);

    // === Phase-level (11 phases from StepProfiler) ===
    fn begin_phase(&mut self, phase: Phase);
    fn end_phase(&mut self, phase: Phase);

    // === Layer-level (per-layer from training-step-profiling-v1) ===
    fn begin_layer(&mut self, layer: usize);
    fn end_layer_fwd(&mut self, layer: usize);
    fn end_layer_bwd(&mut self, layer: usize);

    // === Operation-level (16 ops from per-operation-training-profiling-v1) ===
    fn begin_op(&mut self);
    fn end_op(&mut self, op: OpId);

    // === Dispatch tracking (PMAT-499 new) ===
    fn record_dispatch(&mut self, workgroups: [u32; 3], label: &str);

    // === Memory tracking ===
    fn record_alloc(&mut self, bytes: u64);
    fn record_dealloc(&mut self, bytes: u64);

    // === Output ===
    fn report_json(&self) -> serde_json::Value;
    fn report_otlp(&self) -> Vec<OtlpSpan>;     // renacer compatibility
    fn report_chrome_trace(&self) -> Vec<ChromeTraceEvent>;  // Perfetto/torch.profiler parity
}

#[derive(Clone, Copy)]
pub enum ComputeBackend {
    Wgpu,       // Vulkan/Metal/DX12 via wgpu
    CudaPtx,    // trueno PTX JIT kernels
    CuBlas,     // cuBLAS GEMM
    Cutlass,    // CUTLASS templates
    Simd,       // CPU SIMD (trueno BLIS path)
}

/// 11 phases from entrenar StepProfiler (backend-independent)
pub enum Phase {
    Embed, H2D, Forward, NormLm, Loss,
    GradH2D, LmBwd, NormBwd, BlkBwd, EmbedBwd, Opt,
}

/// 16 per-op IDs from per-operation-training-profiling-v1.yaml
pub enum OpId {
    // Forward (0-8)
    RmsnormAttn, QkvGemm, Attention, OProj,
    RmsnormFfn, GateUpGemm, Silu, DownGemm, Lora,
    // Backward (9-15)
    DownBwd, SwigluBwd, GateUpBwd, AttnBwd, QkvBwd, NormBwd, LoraBwd,
}
```

### 10.3 Backend Implementations

| Backend | GPU Timing | Dispatch Tracking | Memory Tracking | Existing Code to Reuse |
|---------|-----------|-------------------|-----------------|----------------------|
| **Wgpu** | `TIMESTAMP_QUERY_INSIDE_PASSES` via wgpu-profiler | `dispatch_workgroups()` counter | `create_buffer()` wrapper | entrenar `wgpu_pipeline.rs` Instant::now |
| **CudaPtx** | `cudaEventRecord()` around kernel launches | Launch counter per stream | `cuMemAlloc` wrapper | trueno BrickProfiler `SyncMode::Immediate` |
| **CuBlas** | cuBLAS with stream events | cuBLAS handle call counter | Workspace pre-alloc tracking | entrenar StepProfiler 16 ops |
| **Cutlass** | Same as CudaPtx (CUTLASS launches on CUDA stream) | Template instantiation counter | Same as CudaPtx | — (future) |
| **Simd** | `Instant::now()` (CPU-only) | Iteration counter | Process RSS tracking | trueno BlisProfiler levels |

### 10.4 Output Convergence

All backends emit the SAME output schema. Three formats, one data model:

```
ComputeProfiler
  ├── report_json()         → training-step-profiling-v1.yaml schema
  ├── report_otlp()         → renacer OtlpExporter spans (Jaeger/Tempo)
  └── report_chrome_trace() → Chrome Trace Format (Perfetto/torch.profiler parity)
```

**Provable contract binding:** `training-step-profiling-v1.yaml` equation `training_step_decomposition` maps directly to Phase enum. `per-operation-training-profiling-v1.yaml` equation `layer_forward_decomposition` maps to OpId enum. The trait IS the binding — implementing the trait satisfies the contract.

### 10.5 Migration Path

| Step | From | To | Effort |
|------|------|----|--------|
| 1 | entrenar StepProfiler (CUDA) | Implement `ComputeProfiler for CudaProfiler` | Wrap existing 11 phases + 16 ops + per-layer |
| 2 | entrenar wgpu_pipeline.rs Instant::now | Implement `ComputeProfiler for WgpuProfiler` | Port begin_layer/end_layer, add wgpu timestamp queries |
| 3 | trueno BrickProfiler (inference) | Implement `ComputeProfiler for BrickProfilerAdapter` | Map 23 BrickIds to OpId enum |
| 4 | renacer GpuTracerConfig | Add OTLP export to ComputeProfiler | Wire report_otlp() to existing OtlpExporter |
| 5 | cgp NcuProfiler/NsysProfiler | CLI wrapper calls ComputeProfiler | Post-process ncu/nsys into same JSON schema |

### 10.6 Falsification Conditions

> **F-UNI-001:** If WGPU and CUDA profilers produce structurally different JSON for the same model, the interface is NOT unified. Action: fix the implementation that diverges from the schema.
> **F-UNI-002:** If switching `ComputeBackend::Wgpu` to `ComputeBackend::CudaPtx` requires ANY change to the training loop code (beyond the profiler constructor), the abstraction leaks. Action: fix the trait to absorb the backend difference.
> **F-UNI-003:** If report_chrome_trace() output from APR cannot be loaded alongside torch.profiler output in Perfetto, the format is wrong. Action: fix Chrome Trace event schema.
> **F-UNI-004:** If adding a new backend (e.g., CUTLASS) requires modifying existing backend implementations, the trait is not properly abstracted. Action: redesign trait to be additive-only.

## 11. Success Criteria

### v1 (DONE — 2026-04-05)

1. C-PROF-001: `wall_coverage >= 0.95` — **ACHIEVED** (1.000 on gx10)
2. C-PROF-002: `bottleneck` correctly classified — **ACHIEVED** (gpu_lora_bwd 55.7%)
3. F-PROF-002: buffer allocation hypothesis — **FALSIFIED** (0% alloc, 100% GPU compute)
4. F-PROF-006: batch scaling confirms regime — **CONFIRMED** (0.30 < 0.50)

### v2 (10 new falsification conditions)

| ID | Hypothesis | Falsification | Priority |
|----|-----------|---------------|----------|
| F-PROF-008 | Dispatch overhead is significant | GPU kernel time matches CPU time (0.95-1.05x) → overhead negligible | P0 |
| F-PROF-009 | Dispatch overhead dominates | GPU kernel time <50% of CPU time → fusion is correct fix | P0 |
| F-PROF-010 | LoRA backward is memory-bound | AI > ridge point → compute-bound (contradicts) | P0 |
| F-PROF-011 | LoRA backward is deeply memory-bound | AI < 10% ridge → no shader optimization will help | P1 |
| F-PROF-012 | Per-layer timing adds information | Variance <5% → aggregate is sufficient | P1 |
| F-PROF-013 | Anomalous layer exists | Any layer >10% of total (vs 3.6% expected) | P1 |
| F-PROF-014 | Per-step variance matters | CV <5% → averages sufficient | P2 |
| F-PROF-015 | Outlier steps exist | p99/p50 >2.0 → thermal/GC/driver issue | P2 |
| F-PROF-016 | Inter-dispatch gaps matter | Gap time <5% → fusion won't help | P0 |
| F-PROF-017 | Fusion reduces wall time | 4x dispatch reduction → >20% improvement | P0 |

**The profiler v2 is complete when:** F-PROF-008/009 disambiguate dispatch vs kernel overhead, F-PROF-010/011 classify memory vs compute bound, and this classification leads to the correct optimization (fusion, shader, or memory) that closes the 11.2x gap.
