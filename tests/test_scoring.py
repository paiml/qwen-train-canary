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
    """VRAM check should be absent when baseline lacks peak_vram_mb (wgpu)."""
    result = {
        "canary": "wgpu",
        "metrics": {"tokens_per_sec": 7000, "peak_vram_mb": 9999, "final_loss": 2.0},
    }
    baseline = {"tokens_per_sec": 6600, "final_loss": 2.5}  # no peak_vram_mb
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


# --- NaN loss handling (PMAT-467) ---


def test_nan_loss_treated_as_fail():
    """NaN final_loss must trigger loss FAIL (not pass silently)."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3515, "final_loss": float("nan")},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert not score["checks"]["loss"]["pass"], "NaN loss should FAIL"


# --- PROVISIONAL status (PMAT-462/467) ---


def test_apr_provisional_flagged():
    """APR with _baseline_status=PROVISIONAL must surface provisional=True."""
    result = {
        "canary": "apr",
        "metrics": {
            "tokens_per_sec": 194,
            "peak_vram_mb": 4000,
            "final_loss": 16.8,
            "_baseline_status": "PROVISIONAL",
        },
    }
    baseline = {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert score["provisional"] is True


def test_non_provisional_flagged_false():
    """Standard canary should have provisional=False."""
    result = {
        "canary": "unsloth",
        "metrics": {"tokens_per_sec": 6700, "peak_vram_mb": 3515, "final_loss": 0.15},
    }
    score = score_result(result, UNSLOTH_BASELINE)
    assert score["provisional"] is False


# --- FP16 parity scoring (PMAT-473) ---

APR_FP16_BASELINE = {"tokens_per_sec": 150, "peak_vram_mb": 3000, "final_loss": 20.0}


def test_apr_fp16_good_result_passes():
    """FP16 canary with better-than-baseline throughput should PASS."""
    result = {
        "canary": "apr-fp16",
        "metrics": {"tokens_per_sec": 300, "peak_vram_mb": 2800, "final_loss": 16.8},
    }
    score = score_result(result, APR_FP16_BASELINE)
    assert score["pass"]


def test_apr_fp16_regression_fails():
    """FP16 canary 20% below baseline should FAIL."""
    result = {
        "canary": "apr-fp16",
        "metrics": {"tokens_per_sec": 100, "peak_vram_mb": 2800, "final_loss": 16.8},
    }
    score = score_result(result, APR_FP16_BASELINE)
    assert not score["pass"], "20% regression should FAIL"


def test_apr_fp16_high_nan_rate_fails():
    """FP16 with >50% NaN steps must FAIL (PMAT-473: training broken)."""
    result = {
        "canary": "apr-fp16",
        "config": {"steps": 100},
        "metrics": {
            "tokens_per_sec": 300,
            "peak_vram_mb": 2800,
            "final_loss": 16.8,
            "nan_backward_skips": 55,
        },
    }
    score = score_result(result, APR_FP16_BASELINE)
    assert not score["checks"]["nan_rate"]["pass"], ">50% NaN should FAIL"


def test_apr_fp16_low_nan_rate_passes():
    """FP16 with <50% NaN should pass NaN gate (PMAT-473)."""
    result = {
        "canary": "apr-fp16",
        "config": {"steps": 100},
        "metrics": {
            "tokens_per_sec": 300,
            "peak_vram_mb": 2800,
            "final_loss": 16.8,
            "nan_backward_skips": 10,
        },
    }
    score = score_result(result, APR_FP16_BASELINE)
    assert score["checks"]["nan_rate"]["pass"]


def test_apr_fp16_nan_loss_fails():
    """FP16 with NaN loss must FAIL (F-FP16-003: NaN regression check)."""
    result = {
        "canary": "apr-fp16",
        "metrics": {"tokens_per_sec": 300, "peak_vram_mb": 2800, "final_loss": float("nan")},
    }
    score = score_result(result, APR_FP16_BASELINE)
    assert not score["checks"]["loss"]["pass"]


# --- Host-specific baseline (PMAT-465) ---


def test_host_specific_baseline_used():
    """pytorch@yoga baseline should be used when host matches."""
    from score import load_baselines
    import tempfile, json as j

    baselines = {
        "pytorch": {"tokens_per_sec": 4000, "final_loss": 2.0},
        "pytorch@yoga": {"tokens_per_sec": 1500, "final_loss": 2.0},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        j.dump(baselines, f)
        f.flush()
        loaded = load_baselines(f.name)
    # Host-specific lookup
    baseline = loaded.get("pytorch@yoga", loaded.get("pytorch", {}))
    assert baseline["tokens_per_sec"] == 1500, "Should use host-specific baseline"


# --- PMAT-475: Fused canary scoring ---


def test_apr_fused_passes_at_baseline():
    """apr-fused canary at baseline throughput should PASS."""
    result = {
        "canary": "apr-fused",
        "metrics": {"tokens_per_sec": 45, "peak_vram_mb": 4000, "final_loss": 16.0},
    }
    baseline = {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert score["pass"], f"apr-fused at baseline should pass: {score}"


def test_apr_fused_fp16_graph_passes():
    """apr-fused-fp16-graph (max throughput path) should PASS at target."""
    result = {
        "canary": "apr-fused-fp16-graph",
        "metrics": {"tokens_per_sec": 350, "peak_vram_mb": 2800, "final_loss": 18.0},
    }
    baseline = {"tokens_per_sec": 300, "peak_vram_mb": 3000, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert score["pass"], f"apr-fused-fp16-graph at target should pass: {score}"


def test_apr_fused_regression_detected():
    """30% throughput regression in fused path should FAIL."""
    result = {
        "canary": "apr-fused",
        "metrics": {"tokens_per_sec": 25, "peak_vram_mb": 4000, "final_loss": 16.0},
    }
    baseline = {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert not score["pass"], "30% regression should fail"
