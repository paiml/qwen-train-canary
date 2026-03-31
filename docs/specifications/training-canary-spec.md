# Training Canary Performance Specification

**Document ID:** PAIML-TRAIN-CANARY-001
**Version:** 1.0.0
**Last Updated:** 2026-03-31
**Status:** ACTIVE
**Methodology:** Popperian Falsification + Deterministic Canary Benchmarks
**Target:** Detect training performance regressions across CUDA and WGPU backends
**Model:** Qwen2.5-Coder-1.5B-Instruct (1.78B params, 28 layers, hidden=1536, heads=12/2)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Scope](#2-architecture-scope)
3. [Canary Workloads](#3-canary-workloads)
4. [cuBLAS Parity Gate](#4-cublas-parity-gate)
5. [Hardware Targets](#5-hardware-targets)
6. [Metrics Contract](#6-metrics-contract)
7. [Baseline Thresholds](#7-baseline-thresholds)
8. [Scoring & Regression Detection](#8-scoring--regression-detection)
9. [Benchmarking Methodology](#9-benchmarking-methodology)
10. [Dataset Specification](#10-dataset-specification)
11. [Deployment Topology](#11-deployment-topology)
12. [Optimization Roadmap](#12-optimization-roadmap)
13. [Falsification Tests](#13-falsification-tests)
14. [PMAT Compliance](#14-pmat-compliance)
15. [External Contracts](#15-external-contracts)
16. [Academic References](#16-academic-references)
17. [Revision History](#17-revision-history)

---

## 1. Executive Summary

### What This Is

Performance specification for training canary benchmarks that detect regressions in fine-tuning throughput, memory usage, and numerical correctness across CUDA and WGPU backends. Four canary workloads exercise distinct training paths for Qwen2.5-Coder-1.5B-Instruct.

### Chain of Reasoning

**Step 1: Why canaries?** Training performance depends on a deep stack: PyTorch, CUDA runtime, cuBLAS/cuDNN, driver, GPU clocks, memory allocator, model weights, tokenizer, dataset pipeline. A regression in ANY layer silently degrades throughput. Canaries are short (100 steps, ~2 min), deterministic (seed=42, clock-locked), and produce machine-readable JSON. Run them before and after changes to catch regressions within 10% tolerance.

**Step 2: Why four workloads?** Each canary isolates a different bottleneck:

| Canary | Bottleneck Measured | Why It Matters |
|--------|-------------------|---------------|
| **unsloth** | QLoRA adapter + 4-bit quant overhead | Production fine-tuning path. 2-4x faster than naive LoRA. Regression means unsloth API changed or quantization broke. |
| **pytorch** | Raw training loop throughput | Baseline with no optimizations. Isolates PyTorch/CUDA regression from library-specific issues. Full model in bf16. |
| **cublas** | GEMM backend numerical parity | Runs the SAME loop twice (default vs cuBLAS-forced). Detects silent numerical divergence from backend changes. |
| **wgpu** | Non-NVIDIA training viability | Burn framework via Vulkan on AMD. Measures cross-platform training feasibility and WGPU driver regressions. |

**Step 3: What constitutes a regression?** Three gates must pass:

1. **Throughput**: tokens/sec within 10% of baseline (THROUGHPUT_TOLERANCE=0.10)
2. **Memory**: peak VRAM within 5% of baseline (VRAM_TOLERANCE=0.05)
3. **Convergence**: final loss below baseline threshold (loss < 2.0 for CUDA, < 2.5 for WGPU)

The cuBLAS canary adds two additional gates:
4. **Numerical parity**: loss divergence < 0.01 between default and cuBLAS backends
5. **Performance parity**: throughput ratio between 0.95x-1.05x

**Step 4: Where do we run?** Three hardware targets spanning the compute spectrum:

| Host | GPU | VRAM | Compute | Role |
|------|-----|------|---------|------|
| yoga | RTX 4060 Laptop | 8 GB | sm_89 | Primary canary (constrained VRAM) |
| gx10 | Grace Blackwell GB10 | 120 GB | sm_121 | Large-batch canary (unified memory) |
| intel | Radeon Pro W5700X | 8 GB | Navi 10 | WGPU/Vulkan canary |

### Key Constraints

- **8 GB VRAM ceiling** on yoga — full fine-tune of 1.5B bf16 requires ~6 GB weights + ~2 GB activations. Batch size 4 is the safe maximum at seq_len=512.
- **QLoRA NF4** reduces weight memory 4x (1.5 GB), allowing larger batch or sequence.
- **Deterministic seeds** (42) + **locked GPU clocks** (1900 MHz) ensure <2% run-to-run variance.
- **No gradient accumulation** in canary mode — measures raw per-step throughput.

---

## 2. Architecture Scope

### Training Stack

```
┌─────────────────────────────────────────────────────┐
│                 Canary Runner (Python)                │
│  train.py → argparse → dataset → training loop → JSON│
├──────────┬──────────┬──────────────┬─────────────────┤
│ unsloth  │ pytorch  │   cublas     │     wgpu        │
│ QLoRA    │ full FT  │  parity gate │   burn binary   │
│ SFTTrainer│ raw loop│  2x compare  │   subprocess    │
├──────────┴──────────┴──────────────┴─────────────────┤
│              transformers + datasets + trl            │
├──────────────────────────────────────────────────────┤
│           PyTorch (bf16/fp16) │ Burn (f32/f16)       │
├──────────────────────────────┼───────────────────────┤
│   CUDA + cuBLAS + cuDNN      │  WGPU + Vulkan       │
├──────────────────────────────┼───────────────────────┤
│   NVIDIA Driver 590.48.01    │  Mesa/AMDVLK         │
├──────────────────────────────┼───────────────────────┤
│   RTX 4060L / GB10           │  Radeon W5700X       │
└──────────────────────────────┴───────────────────────┘
```

### Model Architecture (Qwen2.5-Coder-1.5B-Instruct)

| Parameter | Value |
|-----------|-------|
| Parameters | 1.78B (1,543M non-embedding) |
| Layers | 28 |
| Hidden dim | 1536 |
| Attention heads | 12 (Q) / 2 (KV, GQA) |
| Intermediate | 8960 (SwiGLU gate+up+down) |
| Vocab size | 151,936 |
| Context length | 32,768 (canary uses 512) |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE (base 1,000,000) |

### Memory Budget (per canary step, batch=4, seq_len=512)

| Component | bf16 Full FT | QLoRA NF4 |
|-----------|-------------|-----------|
| Model weights | 3,560 MB | 890 MB (4-bit) |
| LoRA adapters | — | 26 MB (r=16) |
| Optimizer states | 7,120 MB (AdamW) | 52 MB (AdamW 8-bit) |
| Activations | ~1,200 MB | ~600 MB (gradient ckpt) |
| Gradients | 3,560 MB | 26 MB |
| KV cache (training) | ~384 MB | ~384 MB |
| **Total** | **~15.8 GB** | **~1.98 GB** |

Full fine-tune exceeds 8 GB — yoga canary runs with gradient checkpointing or bf16 mixed precision to fit. QLoRA fits comfortably.

### Scope Boundaries

**In scope:**
- Fine-tuning throughput measurement (tokens/sec, samples/sec)
- Memory regression detection (peak VRAM)
- Numerical parity between GEMM backends (cuBLAS canary)
- Cross-platform training feasibility (WGPU)
- Deterministic, reproducible benchmarks

**Out of scope:**
- Training to convergence (canary=100 steps, not full training)
- Evaluation metrics (BLEU, HumanEval, etc.)
- Multi-GPU training
- Distributed training (FSDP, DeepSpeed)
- Inference performance (see qwen-coder-deploy)

---

## 3. Canary Workloads

### 3.1 Unsloth QLoRA Canary

**File:** `canaries/unsloth/train.py`
**Backend:** CUDA (sm_89+)
**Duration:** ~2 min (100 steps, yoga)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | NF4 (4-bit) | Standard QLoRA, via bitsandbytes |
| LoRA rank | 16 | Balance: quality vs memory. r=8 too low for code, r=32 OOMs |
| LoRA alpha | 32 | alpha/rank = 2 (standard) |
| Target modules | q,k,v,o,gate,up,down | All linear projections |
| Dropout | 0 | Deterministic canary |
| Gradient checkpointing | unsloth | 60% memory savings via unsloth's optimized checkpointing |
| Optimizer | AdamW 8-bit | Via bitsandbytes. 75% optimizer memory reduction |
| LR schedule | Cosine, warmup=10 | Standard for QLoRA |
| bf16 | Auto-detect | True on sm_89+ |

**Key measurements:**
- `throughput_samples_sec`: Direct from trainer metrics
- `tokens_per_sec`: batch × seq_len × steps / wall_time
- `peak_vram_mb`: torch.cuda.max_memory_allocated()
- `final_loss`: trainer.train().training_loss

**Falsification condition:** If unsloth canary throughput < 5,000 tok/s on yoga, investigate:
1. unsloth API change (check version)
2. bitsandbytes quantization regression
3. PyTorch CUDA backend change
4. GPU clock drift (verify 1900 MHz)

### 3.2 PyTorch Baseline Canary

**File:** `canaries/pytorch/train.py`
**Backend:** CUDA (sm_89+) or CPU fallback
**Duration:** ~3 min (100 steps, yoga)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | bf16 (sm_89+) / fp16 | Native mixed precision |
| Optimizer | AdamW | Standard PyTorch, weight_decay=0.01 |
| LR schedule | CosineAnnealingLR | T_max=steps |
| Gradient clipping | 1.0 | Prevent divergence in short runs |
| Quantization | None | Full precision baseline |

**Training loop order (CRITICAL — verified by falsification):**
```python
optimizer.zero_grad()   # Clear BEFORE backward
loss.backward()         # Compute gradients
clip_grad_norm_(1.0)    # Prevent explosion
optimizer.step()        # Update weights
scheduler.step()        # Adjust LR
```

**Falsification condition:** If pytorch canary throughput < 3,000 tok/s on yoga, investigate:
1. PyTorch version change (check torch.__version__)
2. cuDNN autotuner regression
3. bf16 matmul precision change
4. Memory allocator fragmentation (check torch.cuda.memory_stats())

### 3.3 cuBLAS Parity Canary

**File:** `canaries/cublas/train.py`
**Backend:** CUDA (sm_89+)
**Duration:** ~4 min (50 steps × 2 runs, yoga)

This canary is unique: it runs the **same training loop twice** with different GEMM backends and compares:

| Run | Backend Config | Purpose |
|-----|---------------|---------|
| Run 1: default | PyTorch default (cuDNN autotuner) | Baseline |
| Run 2: cublas | `torch.backends.cuda.preferred_linalg_library("cusolver")` + TF32 enabled | Force cuBLAS path |

**Parity metrics:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `loss_divergence` | < 0.01 | Absolute difference in final loss after 50 steps |
| `max_step_divergence` | < 0.05 | Max per-step loss difference (catches early divergence) |
| `throughput_ratio` | 0.95-1.05 | cuBLAS / default throughput ratio |
| `vram_delta_mb` | < 200 | Absolute VRAM difference |

**Why this matters:** Silent GEMM backend changes can cause:
- Numerical drift that compounds over full training runs
- Throughput regressions masked by identical loss curves
- Memory layout changes that trigger OOM on constrained hardware

**Falsification conditions:**
1. If `loss_divergence > 0.01`: backends are numerically divergent — investigate GEMM precision
2. If `throughput_ratio < 0.95`: cuBLAS regression — check driver version, cuBLAS library
3. If `throughput_ratio > 1.05`: default backend regression — likely cuDNN autotuner change
4. If both runs produce identical results (ratio=1.0000): TF32 flag may not be taking effect

**Design decision: 50 steps not 100.** The parity canary runs the model twice, so 50 steps keeps total wall time comparable to other canaries while providing enough steps for loss divergence to manifest.

### 3.4 WGPU/Burn Canary

**File:** `canaries/wgpu/train.py`
**Backend:** WGPU/Vulkan via burn-canary Rust binary
**Duration:** ~5 min (100 steps, intel)

This canary wraps a Rust binary (`burn-canary`) via subprocess because burn's Python bindings are not yet mature. The binary:

1. Loads model weights from SafeTensors/GGUF
2. Constructs the training graph in burn
3. Runs forward + backward + optimizer step
4. Emits JSON metrics to stdout

**Subprocess protocol:**
```
canaries/wgpu/train.py
  └─→ subprocess.run(["burn-canary", "--backend", "wgpu", ...])
       └─→ stdout: JSON metrics
       └─→ stderr: progress logging
       └─→ exit 0: success, non-zero: failure
```

**Falsification conditions:**
1. If `burn-canary` binary not found: build not deployed (check forjar)
2. If throughput < 100 tok/s: expected for current WGPU training (Vulkan compute shaders)
3. If VRAM > 7,000 MB: memory leak in burn allocator
4. If exit code non-zero: Vulkan driver or burn framework regression

---

## 4. cuBLAS Parity Gate

### Background

The cuBLAS parity gate exists because the realizr inference engine (in qwen-coder-deploy) uses hand-written PTX kernels and cuBLAS for GEMM operations. Training canaries use PyTorch's GEMM backends. If these backends silently diverge in numerical behavior, models fine-tuned via one path may behave differently when served via another.

### Parity Chain

```
Training (PyTorch default) ──┐
                              ├── loss divergence < 0.01? ── PARITY
Training (cuBLAS forced)   ──┘

Training (cuBLAS forced)   ──┐
                              ├── weight divergence? ── measure via loss proxy
Inference (realizr PTX)    ──┘
```

The canary measures the first link. The second link (training weights → inference parity) is validated by qwen-coder-deploy's correctness tests after fine-tuning.

### TF32 Configuration

| Setting | Run 1 (default) | Run 2 (cuBLAS) |
|---------|-----------------|----------------|
| `matmul.allow_tf32` | PyTorch default | `True` |
| `cudnn.allow_tf32` | PyTorch default | `True` |
| `preferred_linalg_library` | default | `cusolver` |

TF32 uses 10-bit mantissa (vs FP32's 23-bit) for matmul. On sm_89+ this is the production path. The canary verifies that enabling TF32 explicitly doesn't cause meaningful divergence from the autotuned default.

### Historical Context

The realizr inference engine discovered a Q6K alignment bug (PMAT-078) where shared memory Q8 cache produced subtly different activations depending on warp scheduling order. This was caught by inference correctness tests, not training metrics. The cuBLAS parity canary prevents the training-side analog: a GEMM backend change that shifts weight distributions enough to affect inference quality.

---

## 5. Hardware Targets

### Yoga (Primary — RTX 4060 Laptop)

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX 4060 Laptop GPU |
| Compute | sm_89 (Ada Lovelace) |
| VRAM | 8 GB GDDR6 |
| Memory BW | 256 GB/s |
| TDP | 115W |
| Clock | Locked 1900 MHz (nvidia-smi -lgc) |
| CUDA | 12.6 Runtime / 13.1 Driver |
| Host | AMD, 32 GB RAM |
| Network | LAN 192.168.50.38 |

**Constraints:** 8 GB VRAM means full fine-tune of 1.5B at batch=4 is tight. QLoRA fits comfortably. cuBLAS parity canary loads the model twice (sequentially, with torch.cuda.empty_cache() between).

### gx10 (Grace Blackwell GB10)

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10 |
| Compute | sm_121 (Blackwell) |
| VRAM | 120 GB unified |
| CUDA | 13.0 |
| Host | ARM Grace, 120 GB unified |
| Network | localhost (runs locally) |

**Advantage:** 120 GB unified memory allows batch=16 and larger sequence lengths. Canary uses batch=16 to measure Blackwell-specific throughput.

### Intel (WGPU — Radeon Pro W5700X)

| Property | Value |
|----------|-------|
| GPU | AMD Radeon Pro W5700X |
| Architecture | Navi 10 (RDNA 1) |
| VRAM | 8 GB GDDR6 |
| Compute | 36 CUs, 2304 stream processors |
| API | Vulkan 1.3 (via Mesa RADV) |
| Host | Intel CPU, 64 GB RAM |
| Network | LAN 192.168.50.100 |

**Constraint:** Vulkan compute shaders are significantly slower than CUDA for training. Expected throughput: 50-200 tok/s (vs 3,000-6,000 CUDA). The canary measures feasibility, not competitiveness.

---

## 6. Metrics Contract

### Output Schema (JSON)

Every canary produces a JSON file conforming to this schema:

```json
{
  "canary": "string",           // "unsloth" | "pytorch" | "cublas" | "wgpu"
  "backend": "string",          // "cuda" | "wgpu" | "vulkan" | "cpu"
  "host": "string",             // hostname
  "gpu": {
    "device": "string",         // GPU name
    "vram_total_mb": "int",     // Total VRAM
    "cuda_version": "string",   // CUDA runtime version
    "compute_capability": "string"  // e.g. "8.9"
  },
  "timestamp": "string",        // ISO 8601 UTC
  "config": {
    "model": "string",          // HuggingFace model ID
    "batch_size": "int",
    "seq_len": "int",
    "steps": "int",
    "lr": "float",
    "seed": "int",
    "dtype": "string",          // "bf16" | "fp16" | "f32"
    "optimizer": "string",
    "quantization": "string"    // "nf4" | "none"
  },
  "metrics": {
    "throughput_samples_sec": "float",
    "tokens_per_sec": "float",
    "peak_vram_mb": "int",
    "final_loss": "float",
    "step_time_ms": {
      "mean": "float",
      "p50": "float",
      "p95": "float",
      "p99": "float"
    },
    "wall_time_sec": "float"
  }
}
```

### cuBLAS Parity Extension

The cuBLAS canary adds nested metrics:

```json
{
  "metrics": {
    "default": { /* standard metrics */ },
    "cublas": { /* standard metrics */ },
    "parity": {
      "loss_divergence": "float",      // |default_loss - cublas_loss|
      "max_step_divergence": "float",  // max per-step |d_loss - c_loss|
      "throughput_ratio": "float",     // cublas_tok_s / default_tok_s
      "vram_delta_mb": "int",          // cublas_vram - default_vram
      "numerically_equivalent": "bool", // loss_divergence < 0.01
      "perf_equivalent": "bool"        // 0.95 <= ratio <= 1.05
    }
  }
}
```

### Metric Definitions

| Metric | Formula | Unit |
|--------|---------|------|
| `throughput_samples_sec` | (batch_size × steps) / wall_time | samples/s |
| `tokens_per_sec` | (batch_size × seq_len × steps) / wall_time | tok/s |
| `peak_vram_mb` | torch.cuda.max_memory_allocated() / 1024² | MB |
| `final_loss` | mean(losses[-10:]) | dimensionless |
| `step_time_ms.mean` | mean(all step times) | ms |
| `step_time_ms.p50` | median(all step times) | ms |
| `step_time_ms.p95` | 95th percentile step time | ms |
| `step_time_ms.p99` | 99th percentile step time | ms |
| `wall_time_sec` | end - start (perf_counter) | sec |

---

## 7. Baseline Thresholds

### Initial Baselines (to be updated after first real runs)

| Canary | Metric | Baseline | Tolerance | Gate |
|--------|--------|----------|-----------|------|
| unsloth | tokens_per_sec | 5,000 | -10% | ≥ 4,500 |
| unsloth | peak_vram_mb | 7,000 | +5% | ≤ 7,350 |
| unsloth | final_loss | 2.0 | — | ≤ 2.0 |
| pytorch | tokens_per_sec | 3,000 | -10% | ≥ 2,700 |
| pytorch | peak_vram_mb | 8,000 | +5% | ≤ 8,400 |
| pytorch | final_loss | 2.0 | — | ≤ 2.0 |
| cublas | tokens_per_sec | 3,000 | -10% | ≥ 2,700 |
| cublas | loss_divergence | 0.01 | — | ≤ 0.01 |
| cublas | throughput_ratio | 0.95 | — | ≥ 0.95 |
| wgpu | tokens_per_sec | 100 | -10% | ≥ 90 |
| wgpu | peak_vram_mb | 7,000 | +5% | ≤ 7,350 |
| wgpu | final_loss | 2.5 | — | ≤ 2.5 |

**Baseline update policy:** After 5 consecutive nightly runs with <5% variance, update baselines to the median observed value. Always use a round number (floor to nearest 100 for tok/s, ceil to nearest 100 for VRAM).

### Expected Throughput Projections

| Canary | yoga (8GB) | gx10 (120GB) | intel (8GB) |
|--------|-----------|-------------|------------|
| unsloth | 5,000-7,000 tok/s | 8,000-12,000 tok/s | N/A |
| pytorch | 3,000-4,000 tok/s | 6,000-10,000 tok/s | N/A |
| cublas | 3,000-4,000 tok/s | 6,000-10,000 tok/s | N/A |
| wgpu | N/A | N/A | 50-200 tok/s |

---

## 8. Scoring & Regression Detection

### Score Computation

Each canary result is scored against its baseline with three checks:

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
| Throughput fails, others pass | WARN | Investigate, may be transient |
| VRAM fails | FAIL | Memory regression — block deploy |
| Loss fails | FAIL | Convergence broken — block deploy |
| Numerical parity fails | FAIL | GEMM backend divergence — critical |
| Multiple failures | FAIL | Systematic regression — urgent |

### CI Gate

```bash
make score-gate  # Exits non-zero if ANY canary fails
```

The nightly pipeline runs `make score` after all canaries complete. Non-zero exit blocks the pipeline and alerts.

---

## 9. Benchmarking Methodology

### Determinism Contract

| Factor | Control Method |
|--------|---------------|
| Random seed | `torch.manual_seed(42)` before model load |
| GPU clocks | `nvidia-smi -lgc 1900,1900` (yoga) |
| Dataset order | Fixed YAML, no shuffling in cuBLAS canary |
| Batch composition | DataLoader shuffle=True (pytorch/unsloth) / False (cublas) |
| Model weights | Same HuggingFace revision (pinned) |
| CUDA context | Fresh process per canary (no warm-start contamination) |
| Memory | `torch.cuda.reset_peak_memory_stats()` before training |
| Precision | Auto-detect bf16 on sm_89+, fp16 otherwise |

### Run-to-Run Variance

Expected variance with deterministic seeds and locked clocks:

| Metric | Expected Variance | Acceptable |
|--------|------------------|------------|
| tokens_per_sec | < 2% | < 5% |
| peak_vram_mb | 0% (deterministic) | < 1% |
| final_loss | < 0.1% (deterministic) | < 1% |
| step_time p50 | < 3% | < 5% |
| step_time p99 | < 10% | < 15% |

If variance exceeds these bounds, the benchmark environment is contaminated (thermal throttling, background processes, clock drift).

### Isolation Protocol

1. **Kill competing GPU processes** before canary (forjar: `clean-gpu-mem`)
2. **Lock clocks** to 1900 MHz (forjar: `lock-clocks`)
3. **Run canaries sequentially** (.NOTPARALLEL in Makefile)
4. **Empty cache between cuBLAS runs** (torch.cuda.empty_cache())
5. **Fresh Python process** per canary (SSH command, not subprocess)

---

## 10. Dataset Specification

### Canary Dataset

**File:** `prompts/canary-dataset.yaml`
**Size:** 50 seed samples (expandable to 500 via scripts/expand-dataset.py)
**Format:** YAML with instruction/response pairs

| Property | Value |
|----------|-------|
| Total samples | 50 (seed) |
| Languages | Python (35), Rust (8), SQL (2), Mixed (5) |
| Mean instruction length | ~45 tokens |
| Mean response length | ~120 tokens |
| Mean total length | ~165 tokens |
| Max total length | ~400 tokens |
| Tokenizer | Qwen2.5-Coder tokenizer (151,936 vocab) |

### Training Format

All canaries format samples as:

```
### Instruction:
{instruction}

### Response:
{response}
```

Tokenized with padding to `seq_len` (512) and truncation. The SFTTrainer (unsloth) handles this via `dataset_text_field="text"`. The PyTorch canary tokenizes in the Dataset `__init__`.

### Dataset Quality

The dataset is intentionally simple (basic algorithms, data structures, standard patterns) because the canary measures **training throughput**, not model quality. Complex datasets would introduce confounding variables:
- Variable tokenization length → step time variance
- Hard examples → loss spikes → false regression signals
- Domain-specific patterns → optimizer sensitivity

---

## 11. Deployment Topology

### Forjar Configurations

| Config | Host | Purpose |
|--------|------|---------|
| `forjar-yoga.yaml` | yoga (SSH) | Deploy scripts, create uv venvs, lock clocks |
| `forjar-yoga-teardown.yaml` | yoga (SSH) | Kill training, reset clocks |
| `forjar-intel-wgpu.yaml` | intel (SSH) | Deploy WGPU scripts, verify Vulkan |
| `forjar-gx10.yaml` | gx10 (local) | Setup venvs locally |

### Virtual Environment Strategy

All Python dependencies installed via **uv** (not pip):

```yaml
# forjar-yoga.yaml
command: |
  test -d ~/venvs/unsloth/bin || uv venv ~/venvs/unsloth
  uv pip install -q --python ~/venvs/unsloth/bin/python -r requirements.txt
```

Two venvs per CUDA host:
1. `~/venvs/unsloth` — unsloth + dependencies (bitsandbytes, peft, trl)
2. `~/venvs/pytorch-canary` — minimal PyTorch + transformers (shared by pytorch + cublas canaries)

### Nightly Pipeline

```
scripts/nightly.sh [cuda|wgpu|gx10|all]
  ├── run_cuda_canaries()     # yoga: deploy → unsloth → pytorch → cublas → teardown
  ├── run_wgpu_canaries()     # intel: deploy → wgpu
  ├── run_gx10_canaries()     # gx10: deploy → unsloth → pytorch → cublas
  └── make score              # Score all results against baselines
```

---

## 12. Optimization Roadmap

### Phase 0: Establish Baselines (Current)

| PMAT | Item | Status |
|------|------|--------|
| PMAT-420 | Initial repo scaffold + 4 canary workloads | ✅ DONE |
| PMAT-421 | First yoga canary run (unsloth + pytorch) | Planned |
| PMAT-422 | First cuBLAS parity measurement | Planned |
| PMAT-423 | First WGPU/burn training measurement | Planned |
| PMAT-424 | Establish baselines from 5 nightly runs | Planned |
| PMAT-425 | Batuta stack registration (47 crates) | ✅ DONE |

### Phase 1: Training Throughput Optimization

| PMAT | Item | Expected Impact |
|------|------|----------------|
| PMAT-426 | Torch.compile() canary (PyTorch 2.x graph mode) | +20-40% throughput |
| PMAT-427 | Flash Attention 2 for training (if not default) | +15-25% step time |
| PMAT-428 | Gradient accumulation canary (batch=1, accum=4) | Memory vs throughput tradeoff |
| PMAT-429 | DeepSpeed ZeRO Stage 2 canary | Enable full FT on 8 GB |
| PMAT-430 | FSDP canary (multi-GPU, future) | Distributed training baseline |

### Phase 2: WGPU Training Maturity

| PMAT | Item | Expected Impact |
|------|------|----------------|
| PMAT-431 | burn-canary Rust binary (MVP) | Enable WGPU canary |
| PMAT-432 | WGPU compute shader optimization | 2-5x throughput |
| PMAT-433 | WGPU vs CUDA numerical parity | Cross-backend convergence |
| PMAT-434 | Apple Metal backend canary (M-series) | macOS training feasibility |

### Phase 3: Advanced Canaries

| PMAT | Item | Expected Impact |
|------|------|----------------|
| PMAT-435 | Multi-model canary (3B, 7B) | Scale sensitivity |
| PMAT-436 | Long-context canary (seq_len=2048) | Attention scaling |
| PMAT-437 | Mixed-precision canary (fp8 training) | Next-gen precision |
| PMAT-438 | LoRA merge + inference parity gate | End-to-end validation |

---

## 13. Falsification Tests

### Methodology

Every claim in this spec has a falsification condition. If the condition triggers, the claim is revised or retracted.

### Active Falsification Conditions

| ID | Claim | Falsification Condition | Status |
|----|-------|------------------------|--------|
| F-001 | Canaries detect 10% throughput regressions | Inject artificial 15% slowdown → canary MUST fail | Planned |
| F-002 | cuBLAS parity holds (divergence < 0.01) | If divergence > 0.01 on any hardware → investigate | Planned |
| F-003 | Locked clocks ensure <2% variance | Run 10x same canary → if variance > 2%, clocks aren't locking | Planned |
| F-004 | seed=42 ensures deterministic loss | Run 2x same canary → if loss differs > 0.001, non-determinism | Planned |
| F-005 | WGPU training is feasible on W5700X | If burn-canary crashes or produces 0 tok/s → WGPU not ready | Planned |
| F-006 | Unsloth is faster than raw PyTorch for QLoRA | If unsloth_tok_s < pytorch_tok_s → unsloth overhead > savings | Planned |
| F-007 | Full FT fits in 8 GB at batch=4 seq=512 | If OOM → reduce batch or add gradient checkpointing | Planned |

### Falsified Claims (None Yet)

| ID | Claim | Falsification Date | What Happened |
|----|-------|--------------------|---------------|
| — | — | — | No falsified claims yet (v1.0.0) |

---

## 14. PMAT Compliance

### Work Item Summary

| Range | Area | Count |
|-------|------|-------|
| PMAT-420–425 | Phase 0: Scaffold + baselines | 6 |
| PMAT-426–430 | Phase 1: Throughput optimization | 5 |
| PMAT-431–434 | Phase 2: WGPU maturity | 4 |
| PMAT-435–438 | Phase 3: Advanced canaries | 4 |
| **Total** | | **19** |

### Quality Gates

| Gate | Tool | Threshold |
|------|------|-----------|
| Canary pass/fail | `make score` | All canaries PASS |
| CI gate | `make score-gate` | Exit 0 |
| Nightly regression | `scripts/nightly.sh` | All platforms pass |

---

## 15. External Contracts

### Dependencies

| Dependency | Version | Purpose | Risk |
|------------|---------|---------|------|
| unsloth | latest (git) | QLoRA optimization | API changes break canary |
| PyTorch | ≥2.4.0 | Training framework | GEMM backend changes |
| transformers | ≥4.44.0 | Model loading | Qwen2 support |
| bitsandbytes | ≥0.43.0 | NF4 quantization | CUDA version compat |
| trl | ≥0.9.0 | SFTTrainer | API changes |
| peft | ≥0.12.0 | LoRA implementation | Adapter format changes |
| burn | TBD | WGPU training | Immature, expect breakage |
| uv | latest | Package management | N/A (stable) |
| forjar | latest | Deployment | N/A (PAIML internal) |

### Integration Points

| System | Interface | Direction |
|--------|-----------|-----------|
| qwen-coder-deploy | Correctness tests after fine-tune | canary → deploy |
| batuta | Stack health, PAIML_CRATES registry | canary → batuta |
| forjar | Deployment YAML | canary → forjar |
| probador | Future: training-specific load tests | planned |

---

## 16. Academic References

1. **Hu et al. (2021)** — "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685. Foundation for QLoRA canary.
2. **Dettmers et al. (2023)** — "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314. NF4 quantization methodology.
3. **Han et al. (2024)** — "Unsloth: Efficient LLM Fine-tuning." Optimized gradient checkpointing and memory management.
4. **Dao et al. (2022)** — "FlashAttention: Fast and Memory-Efficient Exact Attention." arXiv:2205.14135. Attention optimization in training.
5. **Micikevicius et al. (2018)** — "Mixed Precision Training." arXiv:1710.03740. bf16/fp16 training methodology.
6. **Rajbhandari et al. (2020)** — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." arXiv:1910.02054. Future Phase 1 reference.
7. **NVIDIA cuBLAS Documentation** — GEMM precision modes, TF32, algorithm selection. cuBLAS parity gate reference.

---

## 17. Revision History

| Version | Date | Changes | PMAT |
|---------|------|---------|------|
| 1.0.0 | 2026-03-31 | Initial spec: 4 canaries, 3 hardware targets, 19 PMAT items | PMAT-420 |
