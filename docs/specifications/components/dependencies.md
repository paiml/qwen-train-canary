# Dependencies & References

**Parent:** [Training Canary Spec](../training-canary-spec.md)

---

## External Dependencies

| Dependency | Version | Purpose | Risk |
|------------|---------|---------|------|
| unsloth | latest (git) | QLoRA optimization | API changes break canary |
| PyTorch | >=2.4.0 | Training framework | GEMM backend changes |
| transformers | >=4.44.0 | Model loading | Qwen2 support |
| bitsandbytes | >=0.43.0 | NF4 quantization | CUDA version compat |
| trl | >=0.9.0 | SFTTrainer | API changes |
| peft | >=0.12.0 | LoRA implementation | Adapter format changes |
| burn | latest (git) | WGPU training | Synthetic MLP working (6,730 tok/s, PMAT-431 DONE); real model loading deferred |
| apr-cli | **0.4.15** (crates.io) | Sovereign Stack training | cuBLAS: 2,101 tok/s on gx10. F-ECOSYSTEM-01 fixed: `cargo install apr-cli` works. |
| uv | latest | Package management | N/A (stable) |
| forjar | latest | Deployment | N/A (PAIML internal) |

## Integration Points

| System | Interface | Direction |
|--------|-----------|-----------|
| qwen-coder-deploy | Correctness tests after fine-tune | canary -> deploy |
| batuta | Stack health, PAIML_CRATES registry | canary -> batuta |
| forjar | Deployment YAML | canary -> forjar |
| probador | Future: training-specific load tests | planned |

## Academic References

### Foundations
1. **Hu et al. (2021)** -- "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
2. **Dettmers et al. (2023)** -- "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314
3. **Unsloth** (2024) -- unslothai/unsloth GitHub. No peer-reviewed paper. Key techniques: custom Triton kernels for QLoRA, manual backward (no autograd), RoPE+cross-entropy fusion. See [Competitive Analysis](#competitive-analysis).
4. **Micikevicius et al. (2018)** -- "Mixed Precision Training." arXiv:1710.03740
5. **Rajbhandari et al. (2020)** -- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." arXiv:1910.02054

### Attention & Kernel Fusion
6. **Dao et al. (2022)** -- "FlashAttention: Fast and Memory-Efficient Exact Attention." arXiv:2205.14135
7. **Dao (2023)** -- "FlashAttention-2: Faster Attention with Better Parallelism." arXiv:2307.08691
8. **Shah et al. (2024)** -- "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608
9. **Zadouri et al. (2026)** -- "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design." arXiv:2603.05451
10. **Li et al. (2022)** -- "Automatic Horizontal Fusion for GPU Kernels." CGO'22 (kernel launch overhead elimination)
11. **Markidis et al. (2018)** -- "NVIDIA Tensor Core Programmability, Performance & Precision." arXiv:1803.04014

### CUDA Graph & Megakernel
12. **Ding et al. (2025)** -- "PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch." arXiv:2503.19779 (>2x training speedup from graph capture)
13. **Fusco et al. (2025)** -- "Boosting Performance of Iterative Applications on GPUs: Kernel Batching with CUDA Graphs." arXiv:2501.09398 (>1.4x from optimal batch size)
14. **Jia et al. (2025)** -- "Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs." arXiv:2512.22219 (entire model as single megakernel)
15. **Hazy Research (2025)** -- "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B." (78% memory BW on H100, 1.5x over existing systems)

### Profiling & Benchmarking
16. **Hoefler & Belli (2015)** -- "Scientific Benchmarking of Parallel Computing Systems." SC'15
17. **NVIDIA CUPTI Documentation** -- Activity API for kernel-level GPU timing
18. **NVIDIA cuBLAS Documentation** -- GEMM precision modes, TF32, algorithm selection
19. **PyTorch torch.profiler Documentation** -- CUPTI-backed per-kernel training profiling

### LoRA & Quantized Training Optimization (added 2026-04-05, PMAT-500)
20. **Ajwebdevs (2026)** -- "Chronicals: A High-Performance Framework for LLM Fine-Tuning with 3.51x Speedup over Unsloth." arXiv:2601.02609. **Critical finding:** Unsloth's 46k tok/s benchmark had zero gradient norms (not training). Fused Triton kernels: RMSNorm 7x, SwiGLU 5x, RoPE 2.3x. Cut Cross-Entropy: logit memory 5GB→135MB.
21. **Zhu et al. (2025)** -- "LoRAFusion: Efficient LoRA Fine-Tuning for LLMs." arXiv:2510.00206 (EuroSys 2026). Fused LoRA kernel: 1.39x speedup from graph-splitting + memory-bound op fusion. Directly applicable to PMAT-484 (fused backward GEMM).
22. **Li et al. (2025)** -- "LoQT: Low-Rank Adapters for Quantized Training." arXiv:2405.16528. NF4 precision for training without storing FP32 weights. Validates our NF4 training approach.
23. **Yu et al. (2025)** -- "Convergence of Adaptive Optimizers under Floating-point Quantization." arXiv:2510.21314. Adam sensitive to β₂ quantization near 1.0. Relevant to APR convergence defect (PMAT-497).
24. **Li et al. (2025)** -- "DQT: Dynamic Quantization Training via Dequantization-Free Nested Integer Arithmetic." arXiv:2508.09176. Eliminates costly dequant-to-float cycle. Relevant to our NF4 dequant-during-forward bottleneck.
25. **Liger-Kernel** (2024-2025) -- Open-source fused Triton kernels for transformer blocks: cross-entropy, RMSNorm, SwiGLU, RoPE. GitHub: linkedin/Liger-Kernel.

### Quantized GEMM & Kernel Fusion (added 2026-04-05, PMAT-500 — from deep arXiv sweep)
26. **Li et al. (2025)** -- "LiquidGEMM: W4A8 GEMM with Implicit Pipeline." arXiv:2509.01229. **SUPPORTS** dequant-in-SHMEM design: fully overlaps weight loading, NF4 dequant, and MMA across warp groups.
27. **Ye et al. (2025)** -- "Accurate INT8 Training." arXiv:2503.08040. Dynamic block-level fallback mixed-precision GEMM for training, 425 TOPS on RTX 4090. **First quantized GEMM training on Qwen-2.5.** SwiGLU activation outliers specifically called out.
28. **Li et al. (2025)** -- "Blockbuster: Operator Fusion for Transformer Blocks." arXiv:2505.07829. Fuses RMSNorm+FFN-SwiGLU (three GEMMs + Hadamard) into single mega-kernel. **Directly validates** our RMSNorm+GEMV and Gate+Up fusion patterns.
29. **Chen et al. (2025)** -- "RedFuser: Cross-Reduction Fusion." arXiv:2603.10026. 2x speedup on FP8 Quant+GEMM workloads on H800. Cascaded reduction fusion.
30. **Zhao et al. (2025)** -- "LLMQ: Lower-Precision Pretraining." arXiv:2512.15306. Joint RMS-norm + residual + abs-max kernel for quantized pretraining on consumer GPUs.
31. **Meng et al. (2025)** -- "DeepCompile: Compiler-Driven Distributed Training." arXiv:2504.09983. Coordinates CUDA graph capture with memory scheduling. **SUPPORTS** graph-based training optimization.

### Convergence & Numerical Precision (added 2026-04-05, PMAT-500 — grounding for PMAT-497)
32. **Sun et al. (2025)** -- "FP4 All the Way." arXiv:2505.19115. **Key finding:** convergence threshold exists at ~sqrt(3) × quantization noise. Stochastic rounding on activations causes divergence. **Directly relevant** to APR loss 11.7 vs 0.45.
33. **Liu et al. (2026)** -- "Quartet II: NVFP4 Pre-Training." arXiv:2601.22813 (ICLR 2026). Unbiased backward-pass quantization (MS-EDEN) critical; biased gradient estimators accumulate error → divergence. **Supports** hypothesis that APR dequant bug produces biased values.
34. **Wei et al. (2025)** -- "any4: Learned 4-bit Codebooks." arXiv:2507.04610. Per-row lookup tables outperform fixed NF4 quantiles on Qwen families. **CONTRADICTS** NF4 as optimal — any4 is strictly better.
35. **Park et al. (2025)** -- "MemAscend: Memory Optimization for Fine-Tuning." arXiv:2505.23254. Pinned buffer allocation is a major bottleneck; alignment-free pinned alloc reduces peak memory 55.7%.
36. **Xu et al. (2025)** -- "WgPy: WebGPU Array Library." arXiv:2503.00279. WebGPU training at 95x CPU speed in-browser. **No direct CUDA comparison** — WGPU/Vulkan vs CUDA for training remains a literature gap.

## Competitive Analysis (added 2026-04-05, PMAT-500)

### Unsloth (unslothai/unsloth)

PyTorch monkey-patcher (not standalone stack). Custom `torch.autograd.Function` classes (`LoRA_MLP`, `LoRA_QKV`, `LoRA_W`) manually derive backward chain rule — eliminates autograd tape and intermediate tensor allocations. Triton kernels: RMSNorm, SwiGLU/GEGLU, fused Q+K RoPE, chunked cross-entropy (logsumexp, 65K vocab chunks). NF4 dequant: calls bitsandbytes `cdequantize_blockwise_fp32_nf4` (NOT fused with GEMM), but pre-allocates global `WEIGHT_BUFFERS`/`ABSMAX_BUFFERS` to eliminate per-call allocation. In-place `addmm_` for fused add+matmul in LoRA backward. 3D→2D flattening for optimal GEMM. Custom gradient checkpointing: async offloads activations to system RAM via non-blocking GPU→CPU. `TiledMLP` tiles along sequence dim. Qwen2 treated as Llama (identical arch). **No peer-reviewed paper.** Chronicals (arXiv:2601.02609) discovered 46k benchmark had zero gradient norms.

> **Root cause of 11x gap (from agent research):** (1) cuBLAS GEMM (years of autotuning) — our WMMA kernel must compete; (2) manual autograd with in-place buffer reuse — eliminates intermediate tensor allocations per-op; (3) async activation offloading keeps GPU saturated. Our 55.7% LoRA backward time likely reflects per-operation allocation instead of fused chain-rule across full MLP/attention blocks.

### Candle (huggingface/candle)

Tape-based autograd (`backprop.rs`) with first-order differentiation. SGD + AdamW optimizers. Training is second-class — no LR schedulers, no native QLoRA/PEFT (community `candle-lora` crate). NF4: **none** (only GGML Q4_0/Q4_1/Q5/Q8/Q2K-Q8K). **No WGPU backend** (has Metal + CPU). Delegates GEMM to **cuBLAS** — no custom GEMM kernels. MNIST: ~3.1s/epoch vs PyTorch ~6s (lower framework overhead). No LLM training benchmarks exist.

> **Implication for APR:** (1) Candle validates Rust ML training is viable but NF4 + WGPU requires custom work regardless. (2) cuBLAS delegation is the pragmatic path — our custom WMMA kernels competing with cuBLAS is ambitious. (3) No other Rust framework has WGPU training — our approach is differentiated territory.

### Axolotl (axolotl-ai-cloud/axolotl)

Adopted Unsloth-style optimizations: custom `torch.autograd.Function` for LoRA MLP + attention, Triton SwiGLU/GEGLU kernels. Open bounty for optimized Triton kernels for full fine-tune. Validates the manual-backward approach as industry standard for fast LoRA training.

### Chronicals (arXiv:2601.02609)

Claims 3.51x over Unsloth on Qwen2.5-0.5B via: (1) fused Triton RMSNorm/SwiGLU/RoPE, (2) Cut Cross-Entropy (logit memory 5GB→135MB), (3) LoRA+ differential learning rates, (4) best-fit sequence packing. 41,184 tok/s full FT on A100. **Discovered Unsloth's 46k benchmark was not training (zero gradients).**

> **Implication for APR:** Cut Cross-Entropy is highly relevant — APR's `gpu_ce` phase is 0.0% today but would dominate at longer sequences. Fused kernels are the proven path.

### PyTorch Internals (from agent research)

CUDA graph training: `make_graphed_callables()` captures forward+backward as separate graphs with **non-shared memory pools** (activation memory waste). `torch.compile` with CUDAGraph Trees solves via unified pool. Hard constraints: static shapes, no CPU-GPU sync, no dynamic control flow. bitsandbytes NF4: dequant to compute dtype inside CUDA kernel, gradients only for LoRA adapters (always bf16). `torch.compile` has known RMSNorm fusion bug (PR #174824). CUDACachingAllocator: `cuMemCreate`+`cuMemAddressReserve` expandable segments — one segment per stream, grows by appending pages (our `buf_alloc` equivalent). Liger-Kernel: 20% throughput + 60% memory reduction for fused ops. PyTorch profiler: 4 default phases (data/forward/backward/optimizer) — our 13-phase decomposition is far more granular.

> **Implication for APR:** PyTorch's CUDACachingAllocator pattern (pre-allocate large virtual ranges, grow on demand) is what our WGPU async pipeline already does (0 allocs/step measured). Their CUDA graph forward+backward memory pool separation is a known limitation we can avoid in our Rust implementation.

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-DEP-01 | If unsloth API breaks canary | Pin to last working version, open issue |
| F-DEP-02 | If bitsandbytes incompatible with CUDA 12.6 | Pin bitsandbytes version or upgrade CUDA |
| F-DEP-03 | If Chronicals' claim that Unsloth benchmarks don't train is confirmed | Re-validate unsloth canary: check gradient norms are non-zero at steps=100 |
| F-DEP-04 | If any4 (arXiv:2507.04610) codebooks are strictly better than NF4 | Evaluate any4 for APR quantization — NF4 may not be optimal 4-bit format |
