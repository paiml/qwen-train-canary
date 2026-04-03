#!/usr/bin/env bash
# nightly.sh — Automated training canary pipeline
# Usage: bash scripts/nightly.sh [yoga|gx10|wgpu|all]
#
# Default: all (run all three target hosts)
# Note: Full fine-tune (pytorch/cublas) OOMs on yoga 8GB (F-EXEC-02 falsified)
set -euo pipefail

MODE="${1:-all}"
DATE=$(date +%Y%m%d)
RESULTS_DIR="results"

echo "=== Training Canary Nightly — $DATE ==="
echo "Mode: $MODE"

mkdir -p "$RESULTS_DIR"

run_yoga_canaries() {
    echo ""
    echo "--- Yoga Canaries (RTX 4060L, QLoRA only) ---"

    echo "Deploying to yoga..."
    make deploy-yoga

    echo "Running unsloth QLoRA canary..."
    make canary-unsloth
    echo "  -> results/canary-unsloth-$DATE.json"

    echo "Running apr fine-tune canary..."
    make canary-apr
    echo "  -> results/canary-apr-$DATE.json"

    echo "Running pytorch gradient accumulation canary (PMAT-459)..."
    make canary-pytorch-gradacc
    echo "  -> results/canary-pytorch-gradacc-$DATE.json"

    make teardown-yoga
}

run_gx10_canaries() {
    echo ""
    echo "--- GB10 Canaries (full FT + parity + QLoRA) ---"

    echo "Deploying to gx10..."
    make deploy-gx10

    echo "Running unsloth QLoRA canary..."
    make canary-unsloth-gx10
    echo "  -> results/canary-unsloth-gx10-$DATE.json"

    echo "Running pytorch canary..."
    make canary-pytorch-gx10
    echo "  -> results/canary-pytorch-gx10-$DATE.json"

    echo "Running cuBLAS parity canary..."
    make canary-cublas-gx10
    echo "  -> results/canary-cublas-gx10-$DATE.json"

    echo "Running apr fine-tune canary on gx10 (PMAT-463)..."
    make canary-apr-gx10
    echo "  -> results/canary-apr-gx10-$DATE.json"
}

run_wgpu_canaries() {
    echo ""
    echo "--- WGPU Canaries (Vulkan, synthetic Qwen-sized MLP) ---"

    echo "Deploying to wgpu host..."
    make deploy-wgpu

    echo "Running wgpu canary..."
    make canary-wgpu
    echo "  -> results/canary-wgpu-$DATE.json"
}

case "$MODE" in
    yoga)
        run_yoga_canaries
        ;;
    gx10)
        run_gx10_canaries
        ;;
    wgpu)
        run_wgpu_canaries
        ;;
    all)
        run_yoga_canaries
        run_gx10_canaries
        run_wgpu_canaries
        ;;
    *)
        echo "Usage: $0 [yoga|gx10|wgpu|all]"
        exit 1
        ;;
esac

echo ""
echo "--- Scoring ---"
make score

echo ""
echo "=== Nightly complete ==="
