# Benchmarking Methodology

**Parent:** [Training Canary Spec](../training-canary-spec.md)

---

## Determinism Contract

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

## Run-to-Run Variance (Yoga Primary)

Expected variance with deterministic seeds and locked clocks on yoga:

| Metric | Expected Variance | Acceptable |
|--------|------------------|------------|
| tokens_per_sec | < 2% | < 5% |
| peak_vram_mb | 0% (deterministic) | < 1% |
| final_loss | < 0.1% (deterministic) | < 1% |
| step_time p50 | < 3% | < 5% |
| step_time p99 | < 10% | < 15% |

If variance exceeds these bounds, the benchmark environment is contaminated (thermal throttling, background processes, clock drift).

## Isolation Protocol

1. **Kill competing GPU processes** before canary (forjar: `clean-gpu-mem`)
2. **Lock clocks** to 1900 MHz (forjar: `lock-clocks`)
3. **Run canaries sequentially** (.NOTPARALLEL in Makefile)
4. **Empty cache between cuBLAS runs** (`torch.cuda.empty_cache()`)
5. **Fresh Python process** per canary (SSH command, not subprocess)

## Memory Budget (per canary step, batch=4, seq_len=512)

| Component | bf16 Full FT | QLoRA NF4 |
|-----------|-------------|-----------|
| Model weights | 3,560 MB | 890 MB (4-bit) |
| LoRA adapters | -- | 26 MB (r=16) |
| Optimizer states | 7,120 MB (AdamW) | 52 MB (AdamW 8-bit) |
| Activations | ~1,200 MB | ~600 MB (gradient ckpt) |
| Gradients | 3,560 MB | 26 MB |
| KV cache (training) | ~384 MB | ~384 MB |
| **Total** | **~15.8 GB** | **~1.98 GB** |

Full fine-tune exceeds 8 GB -- pytorch/cublas canaries auto-enable gradient checkpointing when VRAM <= 16 GB. On yoga (8 GB) this is always active. On gx10 (120 GB) it is disabled to measure raw throughput. QLoRA fits comfortably without checkpointing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                 Canary Runner (Python)                │
│  train.py -> argparse -> dataset -> training loop -> JSON│
├──────────┬──────────┬──────────────┬─────────────────┤
│ unsloth  │ pytorch  │   cublas     │     wgpu        │
│ QLoRA    │ full FT  │  parity gate │   burn binary   │
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

## Model Architecture (Qwen2.5-Coder-1.5B-Instruct)

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

## Diagnostic Discipline

Six approved diagnostic methods, in strict priority order. Use higher-numbered
methods ONLY when lower-numbered methods are insufficient. No ad-hoc printf
debugging. No `let _ =` on any Result. No guessing.

### A. Source Code Reading (pmat query)

**ALWAYS first.** Read the code before forming any hypothesis. Use `pmat query`
(never grep/glob) for semantic, quality-annotated, cross-project search.

```bash
pmat query "forward NaN attention" --include-source --include-project ../entrenar
pmat query --regex "fn\s+rms_norm" --include-project ../trueno --limit 5
pmat query "CudaNf4TransformerBlock forward" --include-source
```

If pmat query finds the function, READ IT. Understand the data flow before
running anything. Most bugs are visible in the code.

### B. Brick Profiling (`apr profile --granular`)

Per-operation GPU timing with roofline analysis. Identifies which brick
(RMSNorm, QkvProjection, AttentionScore, etc.) is the bottleneck and
whether it's compute-bound or memory-bound.

```bash
apr profile model.apr --granular --perf-grade --json --warmup 1 --measure 3 --tokens 16
```

Produces: per-brick timing, bottleneck classification, efficiency grade,
kernel launch overhead, roofline analysis. Grade D or below = needs optimization.

### C. Layer Tracing (`apr trace`, `apr finetune` stderr)

Per-layer data flow: weight statistics (nonzero counts), tensor shapes,
anomaly detection (NaN, zero tensors, shape mismatches).

```bash
apr trace model.apr --verbose --json            # inference layer trace
apr finetune model.apr ... 2>&1 | grep TRACE    # training weight upload trace
```

Training stderr traces show per-tensor dequant quality (`nonzero=N`), upload
verification, and GPU memory state. Use to verify weights survive upload.

### D. NVIDIA Tooling (nsys, ncu) — fallback only

Use ONLY when A-C are insufficient to identify a specific GPU kernel issue.

```bash
nsys profile -o /tmp/trace -t cuda,nvtx --duration 30 apr finetune ...
ncu --target-processes all apr finetune ...   # per-kernel roofline
```

### E. Sister Project Approaches (qwen-coder-deploy)

The inference project (qwen-coder-deploy) uses the same model on the same
hardware via realizr. When entrenar's training path has a bug, compare with
realizr's inference path to isolate whether the issue is in:
- Model loading (shared: aprender format reader)
- Weight layout (different: realizr uses PTX GEMM, entrenar uses cuBLAS)
- Attention (different: realizr has production-optimized kernels)

Key comparison: `apr profile` on Q4K shows 151 tok/s inference. Same model,
same GPU, same weights. If inference works but training doesn't, the bug is
in entrenar's training-specific code path.

### F. Provable Contracts — L5 Enforcement ONLY

ALL diagnostic findings MUST be recorded in provable-contracts YAML:
- Hypotheses with falsification conditions (test BEFORE investing effort)
- DIAG protocol results (operation-level postconditions)
- Measured evidence (not assumptions)

**L5 = compiler-enforced.** The Rust compiler's `#[must_use]` on `Result` is
the enforcement mechanism. NEVER circumvent it:
- `let _ =` on a `Result` is BANNED — it silences compiler enforcement
- `.unwrap()` or `?` REQUIRED on all GPU operations
- If a diagnostic produces uniform zeros from multiple independent buffers,
  the diagnostic is broken (silenced error), not the data

Contract: `apr-training-parity-v1.yaml` defines the hypothesis tree and
DIAG protocol. Every hypothesis must be FALSIFIED or CONFIRMED before
investing engineering effort.

## Measurement Stack

| Layer | Tool | Method | What It Measures |
|-------|------|--------|------------------|
| **Throughput** | canary scripts | A (read) + B (profile) | tok/s, VRAM, loss (regression gate) |
| **Brick profiling** | `apr profile --granular` | B | Per-operation timing, roofline, grade |
| **Layer trace** | `apr trace` / stderr | C | Weight quality, tensor shapes, anomalies |
| **Per-stage validation** | instrumented forward | A + F | Post-op non-zero check (`.unwrap()`) |
| **GPU kernels** | `nsys` / `ncu` | D (fallback) | Kernel timeline, occupancy, stalls |
| **Cross-runtime comparison** | qwen-coder-deploy | E | Inference vs training path isolation |

### Parity diagnosis protocol

When Runtime A is slower than Runtime B on same hardware:
1. **Read the code** (A) — understand both forward paths via `pmat query`
2. **Profile both** (B) — `apr profile --granular` on both runtimes
3. **Trace the slower** (C) — `apr trace` or stderr analysis
4. **Form hypothesis** (F) — record in provable-contract with falsification test
5. **Test hypothesis** (F) — run the falsification test, never guess
6. **Fix the code** (A) — in the upstream repo, not just this project
7. **Re-measure** (B) — confirm parity improvement with canary

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-HW-01 | >5% variance on yoga with locked clocks | Clocks not effective -- investigate |
| F-BM-01 | peak_vram_mb varies >1% across runs | Memory allocator non-determinism |
| F-BM-02 | step_time p99 > 2x p50 | Thermal throttling or GC interference |
