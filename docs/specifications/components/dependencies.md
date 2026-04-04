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
| burn | latest (git) | WGPU training | Synthetic MLP working (6,730 tok/s); real model loading TBD |
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
3. **Han et al. (2024)** -- "Unsloth: Efficient LLM Fine-tuning."
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

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-DEP-01 | If unsloth API breaks canary | Pin to last working version, open issue |
| F-DEP-02 | If bitsandbytes incompatible with CUDA 12.6 | Pin bitsandbytes version or upgrade CUDA |
