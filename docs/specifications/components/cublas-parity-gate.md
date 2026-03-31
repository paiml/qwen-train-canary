# cuBLAS Parity Gate

**Parent:** [Training Canary Spec](../training-canary-spec.md) Section 3.3

---

## Background

The cuBLAS parity gate exists because the realizr inference engine (in qwen-coder-deploy) uses hand-written PTX kernels and cuBLAS for GEMM operations. Training canaries use PyTorch's GEMM backends. If these backends silently diverge in numerical behavior, models fine-tuned via one path may behave differently when served via another.

## Parity Chain

```
Training (PyTorch default) ──┐
                              ├── loss divergence < 0.01? ── PARITY
Training (cuBLAS forced)   ──┘

Training (cuBLAS forced)   ──┐
                              ├── weight divergence? ── measure via loss proxy
Inference (realizr PTX)    ──┘
```

The canary measures the first link. The second link (training weights -> inference parity) is validated by qwen-coder-deploy's correctness tests after fine-tuning.

## TF32 Configuration

| Setting | Run 1 (default) | Run 2 (cuBLAS) |
|---------|-----------------|----------------|
| `matmul.allow_tf32` | PyTorch default | `True` |
| `cudnn.allow_tf32` | PyTorch default | `True` |
| `preferred_linalg_library` | default | `cusolver` |

TF32 uses 10-bit mantissa (vs FP32's 23-bit) for matmul. On sm_89+ this is the production path. The canary verifies that enabling TF32 explicitly doesn't cause meaningful divergence from the autotuned default.

## Parity Metrics

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `loss_divergence` | < 0.01 | Absolute difference in final loss after 50 steps |
| `max_step_divergence` | < 0.05 | Max per-step loss difference (catches early divergence) |
| `throughput_ratio` | 0.95-1.05 | cuBLAS / default throughput ratio |
| `vram_delta_mb` | < 200 | Absolute VRAM difference |

## cuBLAS Parity JSON Extension

```json
{
  "metrics": {
    "default": { /* standard metrics */ },
    "cublas": { /* standard metrics */ },
    "parity": {
      "loss_divergence": "float",
      "max_step_divergence": "float",
      "throughput_ratio": "float",
      "vram_delta_mb": "int",
      "numerically_equivalent": "bool",
      "perf_equivalent": "bool"
    }
  }
}
```

## Historical Context

The realizr inference engine discovered a Q6K alignment bug (PMAT-078) where shared memory Q8 cache produced subtly different activations depending on warp scheduling order. This was caught by inference correctness tests, not training metrics. The cuBLAS parity canary prevents the training-side analog: a GEMM backend change that shifts weight distributions enough to affect inference quality.

## Design Decision: 50 Steps Not 100

The parity canary runs the model twice, so 50 steps keeps total wall time comparable to other canaries while providing enough steps for loss divergence to manifest.

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-WL-03 | loss_divergence > 0.01 | GEMM precision investigation |
| F-WL-04 | throughput_ratio = 1.0000 exactly | TF32 flag not taking effect -- test invalid |
| F-CB-01 | Both runs OOM on yoga | Model too large for sequential dual-run on 8 GB |
| F-CB-02 | max_step_divergence > 0.05 but final divergence < 0.01 | Early divergence self-corrects -- document but don't fail |
