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

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-HW-01 | >5% variance on yoga with locked clocks | Clocks not effective -- investigate |
| F-BM-01 | peak_vram_mb varies >1% across runs | Memory allocator non-determinism |
| F-BM-02 | step_time p99 > 2x p50 | Thermal throttling or GC interference |
