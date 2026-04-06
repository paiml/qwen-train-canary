# Deployment Topology

**Parent:** [Training Canary Spec](../training-canary-spec.md)

---

## Forjar Configurations

| Config | Host | Purpose |
|--------|------|---------|
| `forjar-yoga.yaml` | yoga (SSH) | Deploy scripts, create uv venvs, lock clocks |
| `forjar-yoga-teardown.yaml` | yoga (SSH) | Kill training, reset clocks |
| `forjar-intel-wgpu.yaml` | intel (SSH) | Deploy WGPU scripts, verify Vulkan |
| `forjar-gx10.yaml` | gx10 (local) | Setup venvs locally |

**All three targets are active.** Yoga baselines established (PMAT-424 DONE, 0.34% variance). gx10 and WGPU canaries have measured results in `results/`.

## Python Environment Strategy

All Python dependencies managed via **uv** (the ONLY packaging tool — no pip, conda, poetry):

```bash
# All canaries run via uv:
uv run canaries/unsloth/train.py ...
uv run canaries/pytorch/train.py ...
uv run canaries/cublas/train.py ...
```

Dependencies defined in `pyproject.toml` with optional extras (`cuda`, `wgpu`). Remote hosts use `uv run` directly — no manual venv management.

## Nightly Pipeline

```
scripts/nightly.sh [cuda|wgpu|gx10|all]
  ├── run_cuda_canaries()     # yoga: deploy -> unsloth -> pytorch -> cublas -> teardown
  ├── run_wgpu_canaries()     # wgpu: deploy -> burn-canary binary (PMAT-431 DONE, active)
  ├── run_gx10_canaries()     # gx10: deploy -> unsloth -> pytorch -> cublas (PMAT-424 DONE, active)
  └── make score              # Score all results against baselines
```

**Default:** `scripts/nightly.sh all` — all three targets active as of 2026-04-01.
Note: WGPU results carry host="mac-server" (the Vulkan build host on Intel hardware); spec refers to this as "intel".

**APR canary (2026-04-06):** `canaries/apr/train.py` wraps `apr finetune`. Default `--gpu-backend wgpu`; use `--gpu-backend cuda` for cuBLAS path (2,101 tok/s on gx10). apr-cli 0.4.14 published to crates.io. Install via `cargo install apr-cli`.

## Scope Boundaries

**In scope:**
- Fine-tuning throughput measurement (tokens/sec, samples/sec)
- Memory regression detection (peak VRAM)
- Numerical parity between GEMM backends (cuBLAS canary)
- Cross-platform training feasibility (WGPU)
- Deterministic, reproducible benchmarks

**Out of scope:**
- Training to convergence (canary=100 steps, not full training)
- Evaluation metrics (BLEU, HumanEval, etc.)
- Multi-GPU / distributed training (FSDP, DeepSpeed)
- Inference performance (see qwen-coder-deploy)

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-DT-01 | If forjar deploy to yoga fails | SSH or venv setup broken -- fix before canary |
| F-DT-02 | If nightly.sh hangs >30 min on any single canary | Process leak or OOM hang -- add timeout |
