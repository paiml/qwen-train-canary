#!/usr/bin/env bash
# nightly.sh — Automated training canary pipeline
# Usage: bash scripts/nightly.sh [cuda|wgpu|gx10|all]
set -euo pipefail

MODE="${1:-all}"
DATE=$(date +%Y%m%d)
RESULTS_DIR="results"

echo "=== Training Canary Nightly — $DATE ==="
echo "Mode: $MODE"

mkdir -p "$RESULTS_DIR"

run_cuda_canaries() {
    echo ""
    echo "--- CUDA Canaries (yoga) ---"

    echo "Deploying to yoga..."
    make deploy-yoga

    echo "Running unsloth canary..."
    make canary-unsloth
    echo "  -> results/canary-unsloth-$DATE.json"

    echo "Running pytorch canary..."
    make canary-pytorch
    echo "  -> results/canary-pytorch-$DATE.json"

    echo "Running cuBLAS parity canary..."
    make canary-cublas
    echo "  -> results/canary-cublas-$DATE.json"

    make teardown-yoga
}

run_wgpu_canaries() {
    echo ""
    echo "--- WGPU Canaries (intel) ---"

    echo "Deploying to intel..."
    make deploy-wgpu

    echo "Running wgpu canary..."
    make canary-wgpu
    echo "  -> results/canary-wgpu-$DATE.json"
}

run_gx10_canaries() {
    echo ""
    echo "--- GB10 Canaries (gx10) ---"

    echo "Deploying to gx10..."
    make deploy-gx10

    echo "Running all gx10 canaries..."
    make canary-gx10
    echo "  -> results/canary-*-gx10-$DATE.json"
}

case "$MODE" in
    cuda)
        run_cuda_canaries
        ;;
    wgpu)
        run_wgpu_canaries
        ;;
    gx10)
        run_gx10_canaries
        ;;
    all)
        run_cuda_canaries
        run_wgpu_canaries
        run_gx10_canaries
        ;;
    *)
        echo "Usage: $0 [cuda|wgpu|gx10|all]"
        exit 1
        ;;
esac

echo ""
echo "--- Scoring ---"
make score

echo ""
echo "=== Nightly complete ==="
