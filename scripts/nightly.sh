#!/usr/bin/env bash
# nightly.sh — Automated training canary pipeline
# Usage: bash scripts/nightly.sh [yoga|gx10|wgpu|all]
#
# Default: yoga (QLoRA canary on RTX 4060L)
# F-EXEC-02: full fine-tune (pytorch/cublas) runs on gx10 only
set -euo pipefail

MODE="${1:-yoga}"
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

    make teardown-yoga
}

run_gx10_canaries() {
    echo ""
    echo "--- GB10 Canaries (full FT + parity) ---"

    echo "Deploying to gx10..."
    make deploy-gx10

    echo "Running pytorch canary..."
    make canary-pytorch-gx10
    echo "  -> results/canary-pytorch-gx10-$DATE.json"

    echo "Running cuBLAS parity canary..."
    make canary-cublas-gx10
    echo "  -> results/canary-cublas-gx10-$DATE.json"
}

run_wgpu_canaries() {
    echo ""
    echo "--- WGPU Canaries (intel, Vulkan) ---"

    echo "Deploying to intel..."
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
        # run_wgpu_canaries  # PMAT-423: blocked until burn-canary built
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
