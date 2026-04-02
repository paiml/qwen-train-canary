"""Falsification tests for canary scoring logic.

Implements canary-score-gate-v1.yaml from provable-contracts.
Validates F-SC-01 and F-EXEC-01 conditions.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from score import score_result, score_cublas_result

UNSLOTH_BASELINE = {"tokens_per_sec": 6600, "peak_vram_mb": 3600, "final_loss": 2.0}
CUBLAS_BASELINE = {
    "tokens_per_sec": 4000,
    "max_loss_divergence": 0.01,
    "min_throughput_ratio": 0.95,
    "max_throughput_ratio": 1.05,
}


# --- Throughput gate (C-SCORE-001) ---


def test_15pct_slowdown_fails():
    """FALSIFY-SCORE-001: 15% throughput regression must trigger FAIL."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 5610, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert not score["pass"], "FALSIFIED: 15% regression not detected"
    assert not score["checks"]["throughput"]["pass"]


def test_5pct_variance_passes():
    """FALSIFY-SCORE-002: 5% natural variance must trigger PASS."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6270, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert score["pass"], "FALSIFIED: 5% natural variance wrongly rejected"


def test_exact_boundary_passes():
    """10% boundary (0.90x) should pass."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 5940, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert score["checks"]["throughput"]["pass"]


def test_just_below_boundary_fails():
    """Just below 10% boundary (0.899x) should fail."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 5939, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert not score["checks"]["throughput"]["pass"]


# --- VRAM gate (C-SCORE-002) ---


def test_10pct_vram_increase_fails():
    """FALSIFY-SCORE-003: 10% VRAM regression must trigger FAIL."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3960, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert not score["pass"], "FALSIFIED: 10% VRAM regression not detected"
    assert not score["checks"]["vram"]["pass"]


def test_3pct_vram_increase_passes():
    """3% VRAM increase should pass (within 5% tolerance)."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3708, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert score["checks"]["vram"]["pass"]


# --- Loss gate (C-SCORE-003) ---


def test_loss_above_baseline_fails():
    """Loss exceeding baseline must fail."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3515, "final_loss": 2.1},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert not score["checks"]["loss"]["pass"]


def test_loss_at_baseline_passes():
    """Loss exactly at baseline should pass."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3515, "final_loss": 2.0},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert score["checks"]["loss"]["pass"]


# --- VRAM skip when baseline omits peak_vram_mb ---


def test_vram_skipped_when_baseline_omits():
    """VRAM check should be absent when baseline lacks peak_vram_mb (apr, wgpu)."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 50, "peak_vram_mb": 9999, "final_loss": 3.0},
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 4.0}  # no peak_vram_mb
    score = score_result(result, baseline)
    assert "vram" not in score["checks"], "VRAM should be skipped when baseline omits it"
    assert score["pass"]


def test_vram_checked_when_baseline_includes():
    """VRAM check should be present when baseline includes peak_vram_mb."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert "vram" in score["checks"]


# --- cuBLAS parity gate (C-SCORE-004) ---


def test_cublas_divergence_002_fails():
    """FALSIFY-SCORE-004: cuBLAS divergence 0.02 must trigger FAIL."""
    result = {
        "canary": "cublas",
        "metrics": {
            "parity": {"loss_divergence": 0.02, "throughput_ratio": 1.0},
            "default": {"tokens_per_sec": 4100},
        },
    }
    score = score_cublas_result(result, CUBLAS_BASELINE)
    assert not score["pass"], "FALSIFIED: 0.02 divergence not caught"


def test_cublas_ratio_094_fails():
    """FALSIFY-SCORE-005: cuBLAS ratio 0.94 must trigger FAIL."""
    result = {
        "canary": "cublas",
        "metrics": {
            "parity": {"loss_divergence": 0.0, "throughput_ratio": 0.94},
            "default": {"tokens_per_sec": 4100},
        },
    }
    score = score_cublas_result(result, CUBLAS_BASELINE)
    assert not score["pass"], "FALSIFIED: 0.94 ratio not caught"


def test_cublas_ratio_110_fails():
    """cuBLAS ratio 1.10 must trigger FAIL (above 1.05 upper bound)."""
    result = {
        "canary": "cublas",
        "metrics": {
            "parity": {"loss_divergence": 0.0, "throughput_ratio": 1.10},
            "default": {"tokens_per_sec": 4100},
        },
    }
    score = score_cublas_result(result, CUBLAS_BASELINE)
    assert not score["pass"], "FALSIFIED: ratio 1.10 not caught by upper bound"
    assert not score["checks"]["perf_parity"]["pass"]


def test_cublas_perfect_parity_passes():
    """Perfect parity (0.000 divergence, 1.0043 ratio) should pass."""
    result = {
        "canary": "cublas",
        "metrics": {
            "parity": {"loss_divergence": 0.0, "throughput_ratio": 1.0043},
            "default": {"tokens_per_sec": 4100},
        },
    }
    score = score_cublas_result(result, CUBLAS_BASELINE)
    assert score["pass"]


# --- Good result passes all gates ---


def test_good_result_passes():
    """FALSIFY-SCORE-005 (contract): all-good result must PASS."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert score["pass"]
    assert all(c["pass"] for c in score["checks"].values())
