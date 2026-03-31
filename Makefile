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

.PHONY: deploy-yoga teardown-yoga canary-yoga canary-unsloth canary-pytorch canary-cublas

deploy-yoga:
	forjar apply -f forjar-yoga.yaml

teardown-yoga:
	forjar apply -f forjar-yoga-teardown.yaml

canary-yoga: canary-unsloth canary-pytorch canary-cublas

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
			--output /tmp/canary-unsloth-$(DATE).json'
	scp yoga:/tmp/canary-unsloth-$(DATE).json results/

canary-pytorch:
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

canary-gx10:
	cd ~/qwen-train-canary && \
		~/venvs/unsloth/bin/python canaries/unsloth/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output results/canary-unsloth-gx10-$(DATE).json && \
		~/venvs/pytorch-canary/bin/python canaries/pytorch/train.py \
			--model $(MODEL_ID) \
			--steps $(CANARY_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output results/canary-pytorch-gx10-$(DATE).json && \
		~/venvs/pytorch-canary/bin/python canaries/cublas/train.py \
			--model $(MODEL_ID) \
			--steps $(CUBLAS_STEPS) \
			--batch-size 16 \
			--seq-len $(CANARY_SEQ_LEN) \
			--lr $(CANARY_LR) \
			--seed $(CANARY_SEED) \
			--output results/canary-cublas-gx10-$(DATE).json

# ============================================================================
# Reports & Scoring
# ============================================================================

.PHONY: report score score-json score-gate

report:
	python scripts/report.py --results-dir results/ --output performance.md

score:
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
