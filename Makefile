# ============================================================================
# qwen-train-canary — Training performance canary benchmarks
# ============================================================================
# Canaries:
#   unsloth:  QLoRA fine-tune via unsloth (CUDA)
#   pytorch:  Baseline full fine-tune via PyTorch (CUDA)
#   cublas:   cuBLAS parity gate — default vs cuBLAS GEMM (CUDA)
#   wgpu:     Training via burn framework (WGPU/Vulkan)
#
# Targets:
#   Yoga (RTX 4060L):  make canary-yoga (primary)
#   GB10 (Blackwell):  make canary-gx10
#   Intel (W5700X):    make canary-wgpu
# ============================================================================

.NOTPARALLEL:

DATE := $(shell date +%Y%m%d)

# --- Hosts ---
YOGA_HOST := 192.168.50.38
INTEL_HOST := 192.168.50.100
GX10_HOST := 127.0.0.1

# --- Model ---
MODEL_ID := Qwen/Qwen2.5-Coder-1.5B-Instruct

# --- Canary defaults ---
CANARY_STEPS := 100
CANARY_BATCH := 4
CANARY_SEQ_LEN := 512
CANARY_LR := 2e-4
CANARY_SEED := 42

# Full fine-tune batch (F-EXEC-02: batch=4 OOMs on 8GB with bf16 + 8-bit optimizer)
FT_BATCH := 2

# cuBLAS parity uses fewer steps (runs the model twice)
CUBLAS_STEPS := 50

# ============================================================================
# Yoga (CUDA, RTX 4060 Laptop)
# ============================================================================

.PHONY: deploy-yoga teardown-yoga canary-yoga canary-unsloth canary-pytorch canary-pytorch-gradacc canary-cublas

deploy-yoga:
	forjar apply -f forjar-yoga.yaml

teardown-yoga:
	forjar apply -f forjar-yoga-teardown.yaml

# F-EXEC-02: Full fine-tune (pytorch/cublas) needs >8GB — deferred to gx10
# Yoga runs QLoRA: apr (Sovereign Stack) + unsloth (Python baseline)
canary-yoga: canary-apr canary-unsloth

canary-unsloth:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		~/venvs/unsloth/bin/python canaries/unsloth/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--profile \
			--output /tmp/canary-unsloth-$(DATE).json'
	scp yoga:/tmp/canary-unsloth-$(DATE).json results/

# Gradient accumulation: batch=1, accum=4 — enables full FT on yoga 8GB (PMAT-459)
canary-pytorch-gradacc:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		~/venvs/pytorch-canary/bin/python canaries/pytorch/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size 1 \
			--gradient-accumulation-steps 4 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--profile \
			--output /tmp/canary-pytorch-gradacc-$(DATE).json'
	scp yoga:/tmp/canary-pytorch-gradacc-$(DATE).json results/

# WARNING: canary-pytorch and canary-cublas OOM on yoga 8GB (F-EXEC-02 falsified).
# Use canary-pytorch-gx10 or canary-pytorch-gradacc instead.
canary-pytorch:
	@echo "WARNING: F-EXEC-02 — full fine-tune OOMs on yoga 8GB. Use 'make canary-pytorch-gx10' instead." >&2
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		~/venvs/pytorch-canary/bin/python canaries/pytorch/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size $(FT_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output /tmp/canary-pytorch-$(DATE).json'
	scp yoga:/tmp/canary-pytorch-$(DATE).json results/

canary-cublas:
	@echo "WARNING: F-EXEC-02 — cuBLAS parity OOMs on yoga 8GB. Use 'make canary-cublas-gx10' instead." >&2
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		~/venvs/pytorch-canary/bin/python canaries/cublas/train.py \
			--model $(MODEL_ID) \
			--steps $(CUBLAS_STEPS) \
			--batch-size $(FT_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output /tmp/canary-cublas-$(DATE).json'
	scp yoga:/tmp/canary-cublas-$(DATE).json results/

# ============================================================================
# Intel (WGPU, Radeon Pro W5700X)
# ============================================================================

.PHONY: deploy-wgpu canary-wgpu

deploy-wgpu:
	forjar apply -f forjar-intel-wgpu.yaml

canary-wgpu:
	ssh intel 'cd ~/qwen-train-canary && \
		python canaries/wgpu/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--backend wgpu \
			--output /tmp/canary-wgpu-$(DATE).json'
	scp intel:/tmp/canary-wgpu-$(DATE).json results/

# ============================================================================
# GB10 (CUDA, Grace Blackwell)
# ============================================================================

.PHONY: deploy-gx10 canary-gx10

deploy-gx10:
	forjar apply -f forjar-gx10.yaml

canary-gx10: canary-unsloth-gx10 canary-pytorch-gx10 canary-cublas-gx10

canary-unsloth-gx10:
	ssh gx10 'cd ~/qwen-train-canary && \
		~/venvs/unsloth/bin/python canaries/unsloth/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output /tmp/canary-unsloth-gx10-$(DATE).json'
	scp gx10:/tmp/canary-unsloth-gx10-$(DATE).json results/

canary-pytorch-gx10:
	ssh gx10 'cd ~/qwen-train-canary && \
		~/venvs/pytorch-canary/bin/python canaries/pytorch/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output /tmp/canary-pytorch-gx10-$(DATE).json'
	scp gx10:/tmp/canary-pytorch-gx10-$(DATE).json results/

canary-cublas-gx10:
	ssh gx10 'cd ~/qwen-train-canary && \
		~/venvs/pytorch-canary/bin/python canaries/cublas/train.py \
			--model $(MODEL_ID) \
			--steps $(CUBLAS_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output /tmp/canary-cublas-gx10-$(DATE).json'
	scp gx10:/tmp/canary-cublas-gx10-$(DATE).json results/

# ============================================================================
# APR fine-tune (Sovereign Stack — aprender/entrenar)
# ============================================================================

.PHONY: canary-apr canary-apr-fused canary-apr-fused-bwd canary-apr-tc canary-apr-tc-bwd canary-apr-fp16 canary-apr-fp16-graph canary-apr-profile canary-apr-gx10

# PMAT-491: --gpu-backend wgpu forces fast Q4K direct load (650ms) vs CPU dequant (20 min).
# Requires libvulkan1 on host + upstream fix (aprender: respect --gpu-backend in routing).
canary-apr:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--gpu-backend wgpu \
			--profile-interval 1 \
			--output /tmp/canary-apr-$(DATE).json'
	scp yoga:/tmp/canary-apr-$(DATE).json results/

# PMAT-475: Fused NF4 Gate+Up canary — validates NF4_FUSED_GEMM=1 path.
# Contract: nf4-fused-gate-up-swiglu-v1.yaml (throughput >= 1.15x, loss parity)
canary-apr-fused:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		NF4_FUSED_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--output /tmp/canary-apr-fused-$(DATE).json'
	scp yoga:/tmp/canary-apr-fused-$(DATE).json results/

# PMAT-484: Fused backward Gate+Up canary — validates NF4_FUSED_BWD_GEMM=1 path.
# Contract: fused-backward-gemm-v1.yaml (gradient parity, wall time improvement)
canary-apr-fused-bwd:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		NF4_FUSED_GEMM=1 NF4_FUSED_BWD_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--output /tmp/canary-apr-fused-bwd-$(DATE).json'
	scp yoga:/tmp/canary-apr-fused-bwd-$(DATE).json results/

# PMAT-479: NF4 tensor core GEMM canary — validates NF4_TC_GEMM=1 path.
# Contract: nf4-tensor-core-gemm-v1.yaml (throughput >= 5x naive, loss parity)
canary-apr-tc:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		NF4_TC_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--output /tmp/canary-apr-tc-$(DATE).json'
	scp yoga:/tmp/canary-apr-tc-$(DATE).json results/

# PMAT-481: NF4 tensor core backward GEMM canary — validates NF4_TC_BWD_GEMM=1 path.
# Contract: nf4-backward-tensor-core-gemm-v1.yaml (parity, launch reduction, throughput)
canary-apr-tc-bwd:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		NF4_TC_GEMM=1 NF4_TC_BWD_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--profile-interval 1 \
			--output /tmp/canary-apr-tc-bwd-$(DATE).json'
	scp yoga:/tmp/canary-apr-tc-bwd-$(DATE).json results/

# PMAT-473: FP16 tensor core canary — validates FP16_GEMM=1 path.
# Contract: fp16-training-parity-v1.yaml (loss parity, throughput >= 1.3x, NaN regression)
canary-apr-fp16:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		FP16_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--output /tmp/canary-apr-fp16-$(DATE).json'
	scp yoga:/tmp/canary-apr-fp16-$(DATE).json results/

# PMAT-464: FP16 + CUDA graph canary — maximum throughput path
canary-apr-fp16-graph:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		FP16_GEMM=1 CUDA_GRAPH=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--output /tmp/canary-apr-fp16-graph-$(DATE).json'
	scp yoga:/tmp/canary-apr-fp16-graph-$(DATE).json results/

# PMAT-488: CUDA graph (forward + backward) canary — validates CUDA_GRAPH=1 path.
# Contract: cuda-graph-training-step-v1.yaml (6.5x from launch overhead elimination)
canary-apr-graph:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		CUDA_GRAPH=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--profile-interval 1 \
			--output /tmp/canary-apr-graph-$(DATE).json'
	scp yoga:/tmp/canary-apr-graph-$(DATE).json results/

# PMAT-475+464: Max throughput path — all optimizations enabled
# All optimizations: fused fwd+bwd + TC GEMM fwd+bwd + FP16 + CUDA graph fwd+bwd
# This is the "everything on" target — should show maximum APR throughput.
canary-apr-max:
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		NF4_FUSED_GEMM=1 NF4_FUSED_BWD_GEMM=1 NF4_TC_GEMM=1 NF4_TC_BWD_GEMM=1 FP16_GEMM=1 CUDA_GRAPH=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--profile-interval 1 \
			--output /tmp/canary-apr-max-$(DATE).json'
	scp yoga:/tmp/canary-apr-max-$(DATE).json results/

# PMAT-480: Scientific profiling — step profiler at max granularity.
# Contract: training-step-profiling-v1.yaml (wall coverage >= 90%, per-phase breakdown)
# Runs with profile-interval=1 to capture every step. Exercises all canary variants.
canary-apr-profile:
	@echo "=== APR Scientific Profiling (PMAT-480) ==="
	@echo "--- Baseline (cuBLAS NF4) ---"
	ssh yoga 'cd ~/qwen-train-canary && \
		sudo nvidia-smi -lgc 1900,1900 && \
		python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps 20 --batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) --seed $(CANARY_SEED) \
			--method qlora --profile-interval 1 \
			--output /tmp/canary-apr-profile-baseline-$(DATE).json 2>&1' | tee results/profile-apr-baseline-$(DATE).log
	@echo "--- FP16 GEMM ---"
	ssh yoga 'cd ~/qwen-train-canary && \
		FP16_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps 20 --batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) --seed $(CANARY_SEED) \
			--method qlora --profile-interval 1 \
			--output /tmp/canary-apr-profile-fp16-$(DATE).json 2>&1' | tee results/profile-apr-fp16-$(DATE).log
	@echo "--- Fused NF4 Gate+Up ---"
	ssh yoga 'cd ~/qwen-train-canary && \
		NF4_FUSED_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps 20 --batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) --seed $(CANARY_SEED) \
			--method qlora --profile-interval 1 \
			--output /tmp/canary-apr-profile-fused-$(DATE).json 2>&1' | tee results/profile-apr-fused-$(DATE).log
	@echo "--- NF4 Tensor Core GEMM ---"
	ssh yoga 'cd ~/qwen-train-canary && \
		NF4_TC_GEMM=1 python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps 20 --batch-size $(CANARY_BATCH) \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) --seed $(CANARY_SEED) \
			--method qlora --profile-interval 1 \
			--output /tmp/canary-apr-profile-tc-$(DATE).json 2>&1' | tee results/profile-apr-tc-$(DATE).log
	scp yoga:/tmp/canary-apr-profile-*-$(DATE).json results/
	@echo "=== Profiling complete — compare results/ logs ==="

canary-apr-gx10:
	ssh gx10 'cd ~/qwen-train-canary && \
		python3 canaries/apr/train.py \
			--model $(MODEL_ID) \
			--model-path ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
			--steps $(CANARY_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--method qlora \
			--gpu-backend wgpu \
			--profile-interval 1 \
			--output /tmp/canary-apr-gx10-$(DATE).json'
	scp gx10:/tmp/canary-apr-gx10-$(DATE).json results/

# ============================================================================
# Phase 1: torch.compile canary (PMAT-426)
# ============================================================================

.PHONY: canary-compile-gx10

canary-compile-gx10:
	ssh gx10 'cd ~/qwen-train-canary && \
		~/venvs/pytorch-canary/bin/python canaries/pytorch/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--compile \
			--output /tmp/canary-compile-gx10-$(DATE).json'
	scp gx10:/tmp/canary-compile-gx10-$(DATE).json results/

# ============================================================================
# Profiling & Tracing (mirrors qwen-coder-deploy measurement stack)
# ============================================================================

.PHONY: profile-yoga profile-gx10 trace-yoga bench-yoga nsys-yoga

# Roofline analysis + per-brick hotspots (apr profile)
profile-yoga:
	ssh yoga 'apr profile ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
		--granular --perf-grade --json \
		--warmup 3 --measure 10 --tokens 32' | tee results/profile-yoga-$(DATE).json

profile-gx10:
	ssh gx10 'apr profile ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr \
		--granular --perf-grade --json \
		--warmup 3 --measure 10 --tokens 32' | tee results/profile-gx10-$(DATE).json

# Layer-by-layer trace (apr trace)
trace-yoga:
	ssh yoga 'apr trace ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr --verbose --json' \
		| tee results/trace-yoga-$(DATE).json

# NVIDIA kernel profiling (nsys/ncu)
nsys-yoga:
	@echo "=== nsys: Profiling training step on yoga ==="
	ssh yoga 'cd ~/qwen-train-canary && sudo nvidia-smi -lgc 1900,1900 && \
		nsys profile -o /tmp/canary-nsys-$(DATE) -t cuda,nvtx --duration 30 \
		~/venvs/unsloth/bin/python canaries/unsloth/train.py \
			--model $(MODEL_ID) --steps 10 --batch-size 4 --seq-len 512 \
			--seed 42 --output /dev/null 2>&1'
	scp yoga:/tmp/canary-nsys-$(DATE).nsys-rep results/

# ============================================================================
# Reports & Scoring
# ============================================================================

.PHONY: test report parity score score-json score-gate validate-schema

test:
	python3 -m pytest tests/ -v

report:
	python scripts/report.py --results-dir results/ --output performance.md

parity:
	python3 scripts/parity-report.py --latest --results-dir results/ --baselines baselines.json

validate-schema:
	python3 scripts/validate_schema.py results/

score: validate-schema
	python scripts/score.py --results-dir results/ --baselines baselines.json

score-json:
	python scripts/score.py --results-dir results/ --baselines baselines.json --format json --output results/scorecard-$(DATE).json

# CI gate — exits non-zero if ANY canary fails (same as score, explicit target for spec)
score-gate: score

# ============================================================================
# Nightly
# ============================================================================

.PHONY: nightly

nightly:
	bash scripts/nightly.sh
