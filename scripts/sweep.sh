#!/usr/bin/env bash
# sweep.sh — Systematic optimization flag sweep for APR training canary.
#
# Runs APR canary with each optimization flag combination to measure
# isolated and combined impact. Produces JSON results for parity-report.py.
#
# Usage: bash scripts/sweep.sh [yoga|gx10]
#
# PMAT-488: Measures CUDA graph, TC GEMM, fused backward, FP16 effects.
set -euo pipefail

HOST="${1:-yoga}"
DATE=$(date +%Y%m%d)
RESULTS_DIR="results/sweep-${DATE}"
mkdir -p "$RESULTS_DIR"

MODEL_ID="Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_PATH="~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr"
STEPS=20  # Short sweep for quick comparison
BATCH=4
SEQ=512
LR="2e-4"
SEED=42

echo "=== APR Optimization Sweep — $DATE ==="
echo "Host: $HOST, Steps: $STEPS, Results: $RESULTS_DIR"

run_variant() {
    local name="$1"
    local env_vars="$2"

    echo ""
    echo "--- $name ---"
    echo "  Flags: $env_vars"

    local output="/tmp/sweep-${name}-${DATE}.json"
    ssh "$HOST" "cd ~/qwen-train-canary && \
        sudo nvidia-smi -lgc 1900,1900 && \
        $env_vars python3 canaries/apr/train.py \
            --model $MODEL_ID \
            --model-path $MODEL_PATH \
            --steps $STEPS \
            --batch-size $BATCH \
            --seq-len $SEQ \
            --lr $LR \
            --seed $SEED \
            --method qlora \
            --profile-interval 1 \
            --output $output"
    scp "${HOST}:${output}" "${RESULTS_DIR}/${name}.json"

    # Extract throughput
    local tok_s
    tok_s=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${name}.json')); print(d['metrics']['tokens_per_sec'])" 2>/dev/null || echo "0")
    echo "  → ${tok_s} tok/s"
}

echo ""
echo "=== Phase 1: Isolated flags ==="

run_variant "baseline" ""
run_variant "fused-fwd" "NF4_FUSED_GEMM=1"
run_variant "fused-bwd" "NF4_FUSED_BWD_GEMM=1"
run_variant "tc-fwd" "NF4_TC_GEMM=1"
run_variant "tc-bwd" "NF4_TC_BWD_GEMM=1"
run_variant "fp16" "FP16_GEMM=1"
run_variant "graph" "CUDA_GRAPH=1"

echo ""
echo "=== Phase 2: Combined flags ==="

run_variant "fused-both" "NF4_FUSED_GEMM=1 NF4_FUSED_BWD_GEMM=1"
run_variant "tc-both" "NF4_TC_GEMM=1 NF4_TC_BWD_GEMM=1"
run_variant "graph-tc" "CUDA_GRAPH=1 NF4_TC_GEMM=1 NF4_TC_BWD_GEMM=1"
run_variant "graph-fused" "CUDA_GRAPH=1 NF4_FUSED_GEMM=1 NF4_FUSED_BWD_GEMM=1"

echo ""
echo "=== Phase 3: Maximum throughput ==="

run_variant "max" "NF4_FUSED_GEMM=1 NF4_FUSED_BWD_GEMM=1 NF4_TC_GEMM=1 NF4_TC_BWD_GEMM=1 FP16_GEMM=1 CUDA_GRAPH=1"

echo ""
echo "=== Sweep Complete ==="
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Generate comparison report:"
echo "  python3 scripts/parity-report.py ${RESULTS_DIR}/*.json"
