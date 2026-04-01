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

1. **Hu et al. (2021)** -- "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
2. **Dettmers et al. (2023)** -- "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314
3. **Han et al. (2024)** -- "Unsloth: Efficient LLM Fine-tuning."
4. **Dao et al. (2022)** -- "FlashAttention: Fast and Memory-Efficient Exact Attention." arXiv:2205.14135
5. **Micikevicius et al. (2018)** -- "Mixed Precision Training." arXiv:1710.03740
6. **Rajbhandari et al. (2020)** -- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." arXiv:1910.02054
7. **NVIDIA cuBLAS Documentation** -- GEMM precision modes, TF32, algorithm selection

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-DEP-01 | If unsloth API breaks canary | Pin to last working version, open issue |
| F-DEP-02 | If bitsandbytes incompatible with CUDA 12.6 | Pin bitsandbytes version or upgrade CUDA |
