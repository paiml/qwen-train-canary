# Training Profiler Specification

**Parent:** [Training Canary Spec](../training-canary-spec.md) Section 8
**PMAT:** PMAT-496 (inter-step overhead), PMAT-480 (BrickProfiler), PMAT-483 (per-op)
**Status:** ACTIVE — profiler implemented, F-PROF-002 falsified (v6.5.0)

---

## 1. Problem Statement

APR WGPU training on gx10 GB10 shows **24,094 tok/s step-level GPU compute** (4.6x faster than unsloth 5,262 tok/s) but only **421 tok/s wall-clock**. GPU is idle 99% of the time. Batch scaling from 4→16 yields only 21% throughput gain (421→509 tok/s) — confirming the bottleneck is NOT compute-bound.

The existing `[PROFILE]` output in `wgpu_pipeline.rs` captures only the 85ms of GPU dispatch per step, not the 8,765ms of inter-step overhead. We cannot fix what we cannot measure.

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

**Measurement pipeline:** `apr --profile` (measurement) → `canary` (collection) → `score.py` (gating) → `spec` (analysis)

## 3. Profiling Phases

### 3.1 Hierarchical Decomposition

Following Chopper's multi-level methodology (arXiv:2512.08242), the training step decomposes into phases that sum to 100% of wall time. Every nanosecond between `step_start` and `step_end` is attributed to exactly one phase.

| Phase | What | Current (gx10) | Category |
|-------|------|-----------------|----------|
| `data_prep` | Tokenize + build input_ids + prompt/response split | unmeasured | CPU |
| `embed` | CPU embedding lookup (vocab × hidden matmul) | ~0ms | CPU |
| `buf_alloc` | `create_buffer` + `zeros()` calls | **unmeasured** | Memory |
| `buf_write` | `queue.write_buffer` (CPU → GPU staging) | unmeasured | Transfer |
| `fwd_encode` | Command encoder recording for 28-layer forward | unmeasured | CPU |
| `fwd_submit` | `queue.submit()` for forward pass | unmeasured | Dispatch |
| `gpu_fwd` | GPU compute: 28-layer forward + LoRA inline | 22ms | Compute |
| `gpu_lm` | GPU compute: RMSNorm + lm_head GEMM | 22ms | Compute |
| `gpu_ce` | GPU compute: cross-entropy fwd + bwd | ~0ms | Compute |
| `gpu_lm_bwd` | GPU compute: lm_head backward GEMM | 23ms | Compute |
| `gpu_lora_bwd` | GPU compute: LoRA backward + AdamW | 16ms | Compute |
| `sync` | `device.poll` / `map_async` — GPU→CPU loss readback | **unmeasured** | Sync |
| `overhead` | Residual (Rust runtime, loop bookkeeping) | residual | Other |

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
    prediction: >
      alloc_time >= 50% of wall-clock per step. Based on: 600+
      create_buffer calls per step, each hitting Vulkan driver.
      STAlloc (arXiv:2507.16274) shows cudaMalloc dominates in
      similar patterns. Burn-Compute found identical issue with
      early burn-wgpu.
    test: >
      Run 10-step training with allocation tracking.
      Compute alloc_ratio = alloc_time / wall_time.
      Assert alloc_ratio >= 0.50.
    if_fails: >
      Buffer allocation is NOT the dominant bottleneck. Investigate:
      (1) sync phase (device.poll blocking), (2) queue.submit overhead,
      (3) data_prep CPU work. The fix is NOT buffer pooling — profile
      deeper to find actual bottleneck.

  - id: F-PROF-004
    rule: GPU compute is NOT the bottleneck
    prediction: >
      gpu_compute_pct < 5% of wall time. Measured: 85ms GPU / 8850ms
      wall = 0.96%. Batch=4→16 scaling: 421→509 = 21% (not 400%).
    test: >
      Assert gpu_compute_pct < 20% from profiler output.
      Assert batch_scaling_efficiency < 0.50.
    if_fails: >
      GPU IS compute-bound. PMAT-496 finding was wrong. The profiler
      is measuring something other than actual GPU utilization.
      Validate with nvidia-smi or wgpu timestamp queries.

  - id: F-PROF-005
    rule: pre-allocation fixes the bottleneck
    prediction: >
      After implementing buffer pool (ensure_training_activations +
      reuse zeros), wall-clock throughput improves >= 2x.
      STAlloc saw 79% fragmentation reduction; Burn saw 40-50%
      memory reduction with chunk/slice pooling.
    test: >
      Measure before: wall_time with current zeros() pattern.
      Implement buffer pool. Measure after.
      Assert wall_time_after <= wall_time_before * 0.50.
    if_fails: >
      Pre-allocation is NOT sufficient. The overhead is in a
      different phase (sync, submit, encode). Profile again with
      the pool in place to find the NEXT bottleneck.

  - id: F-PROF-006
    rule: batch scaling confirms overhead-dominated regime
    prediction: >
      batch_scaling_efficiency < 0.50. Measured: (509/421)/4 = 0.30.
      Time-is-Not-Compute (arXiv:2603.28823) predicts short-budget
      training is overhead-dominated, not compute-dominated.
    test: >
      Already measured: batch=4 → 421 tok/s, batch=16 → 509 tok/s.
      scaling = (509/421)/4 = 0.30. Assert < 0.50.
    if_fails: >
      Training is closer to compute-bound than expected. The 99%
      idle claim needs revision — check if profiled GPU times are
      undercounting actual GPU work.

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
  "host": "gx10-a5b5",
  "backend": "wgpu",
  "model": "qwen2.5-coder-1.5b",
  "steps_profiled": 13,
  "wall_time_ms": 115000,
  "phases": {
    "data_prep":    { "total_ms": 200,   "pct": 0.2,  "avg_ms": 15.4 },
    "embed":        { "total_ms": 50,    "pct": 0.0,  "avg_ms": 3.8 },
    "buf_alloc":    { "total_ms": 95000, "pct": 82.6, "avg_ms": 7308 },
    "buf_write":    { "total_ms": 500,   "pct": 0.4,  "avg_ms": 38.5 },
    "fwd_encode":   { "total_ms": 300,   "pct": 0.3,  "avg_ms": 23.1 },
    "fwd_submit":   { "total_ms": 100,   "pct": 0.1,  "avg_ms": 7.7 },
    "gpu_fwd":      { "total_ms": 286,   "pct": 0.2,  "avg_ms": 22.0 },
    "gpu_lm":       { "total_ms": 286,   "pct": 0.2,  "avg_ms": 22.0 },
    "gpu_ce":       { "total_ms": 13,    "pct": 0.0,  "avg_ms": 1.0 },
    "gpu_lm_bwd":   { "total_ms": 299,   "pct": 0.3,  "avg_ms": 23.0 },
    "gpu_lora_bwd": { "total_ms": 208,   "pct": 0.2,  "avg_ms": 16.0 },
    "sync":         { "total_ms": 15000, "pct": 13.0, "avg_ms": 1154 },
    "overhead":     { "total_ms": 2958,  "pct": 2.6,  "avg_ms": 227 }
  },
  "wall_coverage": 0.974,
  "bottleneck": "buf_alloc",
  "bottleneck_pct": 82.6,
  "alloc_stats": {
    "count_per_step": 612,
    "total_alloc_ms": 95000,
    "avg_alloc_us": 11.9
  },
  "gpu_compute_ms": 1092,
  "gpu_compute_pct": 0.9,
  "theoretical_tok_s": 24094,
  "actual_tok_s": 421,
  "speedup_if_zero_overhead": 57.2
}
```

**Note:** Phase values are HYPOTHETICAL — illustrating expected finding. The profiler will measure the actual breakdown.

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

Create `contracts/entrenar/profiler-wall-coverage-v1.yaml` and `profiler-bottleneck-classification-v1.yaml` in provable-contracts repo. Add binding to `contracts/entrenar/binding.yaml`.

## 9. References

| ID | Paper | Relevance |
|----|-------|-----------|
| arXiv:2512.08242 | Chopper: Multi-Level GPU Characterization for LLM Training | Hierarchical decomposition methodology (kernel→phase→iteration) |
| arXiv:2506.08528 | EROICA/PerfTracker: Online Performance Troubleshooting | Priority taxonomy for root-cause localization |
| arXiv:2507.16274 | STAlloc: Reducing GPU Memory Fragmentation | Pre-allocated memory pool eliminates alloc overhead (79% reduction) |
| arXiv:2401.08156 | GMLake: GPU Memory Defragmentation (ASPLOS'24) | Quantified allocation overhead: 9.2 GB average reclaimed |
| arXiv:2407.12117 | Memo: Fine-grained Tensor Management | Fragmentation → reorganization → training stall causal chain |
| arXiv:2603.22774 | CPU-Induced Slowdowns in Multi-GPU Inference | CPU oversubscription creates GPU idle time via launch latency |
| arXiv:2603.28823 | Time is Not Compute: Scaling Laws for Wall-Clock Training | Dual U-shape: short budgets overhead-dominated, long compute-dominated |
| arXiv:2501.16909 | Measuring GPU Utilization One Level Deeper | Per-resource idle time decomposition beyond SM-occupancy |
| arXiv:2410.07192 | PipeFill: Using GPUs During Bubbles | Pipeline bubble baselines (15-30%) for idle-time detection |
| arXiv:2404.09151 | TapML: Emerging Platforms Meet Emerging LLMs | Vulkan-specific compilation failures and platform testing |

## 10. Success Criteria

The profiler is complete when:
1. C-PROF-001: `wall_coverage >= 0.95` on a 5-step gx10 run
2. C-PROF-002: `bottleneck` field correctly classifies the dominant phase
3. F-PROF-002 or its falsification identifies the actual bottleneck
4. F-PROF-006: batch scaling measurement confirms overhead-dominated regime (already measured: 0.30)
5. The identified bottleneck leads to a fix that improves wall-clock throughput >= 2x
