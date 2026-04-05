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


def test_vram_skipped_when_baseline_is_zero_sentinel():
    """F-VRAM-01: peak_vram_mb=0 in baseline is a sentinel meaning 'no VRAM tracking'.

    The WGPU path has no torch.cuda.max_memory_allocated equivalent, so the
    apr baseline uses peak_vram_mb=0. Without this skip, 0 * 1.05 = 0 causes
    any nonzero VRAM to FAIL the check (false positive).
    """
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "peak_vram_mb": 1166, "final_loss": 1.5,
                    "_metrics_quality": "measured", "valid_backward_steps": 8},
    }
    baseline = {"tokens_per_sec": 470, "peak_vram_mb": 0, "final_loss": 12.0}
    score = score_result(result, baseline)
    assert "vram" not in score["checks"], "VRAM should be skipped when baseline is 0 sentinel"


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
            "final_loss": 1.5,
            "_baseline_status": "PROVISIONAL",
            "_metrics_quality": "measured",
            "valid_backward_steps": 8,
        },
    }
    baseline = {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 2.0}
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

APR_FP16_BASELINE = {"tokens_per_sec": 150, "peak_vram_mb": 3000, "final_loss": 2.0}


def test_apr_fp16_good_result_passes():
    """FP16 canary with better-than-baseline throughput should PASS."""
    result = {
        "canary": "apr-fp16",
        "metrics": {"tokens_per_sec": 300, "peak_vram_mb": 2800, "final_loss": 1.5,
                    "_metrics_quality": "measured", "valid_backward_steps": 10},
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
            "final_loss": 1.5,
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
            "final_loss": 1.5,
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
        "metrics": {"tokens_per_sec": 45, "peak_vram_mb": 4000, "final_loss": 1.5,
                    "_metrics_quality": "measured", "valid_backward_steps": 8},
    }
    baseline = {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 2.0}
    score = score_result(result, baseline)
    assert score["pass"], f"apr-fused at baseline should pass: {score}"


def test_apr_fused_fp16_graph_passes():
    """apr-fused-fp16-graph (max throughput path) should PASS at target."""
    result = {
        "canary": "apr-fused-fp16-graph",
        "metrics": {"tokens_per_sec": 350, "peak_vram_mb": 2800, "final_loss": 1.5,
                    "_metrics_quality": "measured", "valid_backward_steps": 8},
    }
    baseline = {"tokens_per_sec": 300, "peak_vram_mb": 3000, "final_loss": 2.0}
    score = score_result(result, baseline)
    assert score["pass"], f"apr-fused-fp16-graph at target should pass: {score}"


def test_apr_fused_regression_detected():
    """30% throughput regression in fused path should FAIL."""
    result = {
        "canary": "apr-fused",
        "metrics": {"tokens_per_sec": 25, "peak_vram_mb": 4000, "final_loss": 1.5},
    }
    baseline = {"tokens_per_sec": 40, "peak_vram_mb": 4200, "final_loss": 2.0}
    score = score_result(result, baseline)
    assert not score["pass"], "30% regression should fail"


# --- NF4 tensor core canary tests (PMAT-479) ---

def test_apr_tc_passes_at_baseline():
    """apr-tc canary at or above baseline should PASS."""
    result = {
        "canary": "apr-tc",
        "metrics": {
            "tokens_per_sec": 60,
            "peak_vram_mb": 4000,
            "final_loss": 1.5,
            "nan_backward_skips": 0,
            "valid_backward_steps": 100,
            "_metrics_quality": "measured",
        },
    }
    baseline = {"tokens_per_sec": 50, "peak_vram_mb": 4200, "final_loss": 2.0}
    score = score_result(result, baseline)
    assert score["pass"], f"apr-tc at baseline should pass: {score}"


def test_apr_tc_regression_detected():
    """NF4 tensor core canary regression should FAIL."""
    result = {
        "canary": "apr-tc",
        "metrics": {
            "tokens_per_sec": 30,
            "peak_vram_mb": 4000,
            "final_loss": 1.5,
            "nan_backward_skips": 0,
            "valid_backward_steps": 100,
        },
    }
    baseline = {"tokens_per_sec": 50, "peak_vram_mb": 4200, "final_loss": 2.0}
    score = score_result(result, baseline)
    assert not score["pass"], "40% throughput regression should fail"


def test_apr_tc_nan_rate_fails():
    """NF4 tensor core canary with >50% NaN should FAIL."""
    result = {
        "canary": "apr-tc",
        "config": {"steps": 100},
        "metrics": {
            "tokens_per_sec": 60,
            "peak_vram_mb": 4000,
            "final_loss": 1.5,
            "nan_backward_skips": 60,
            "valid_backward_steps": 40,
        },
    }
    baseline = {"tokens_per_sec": 50, "peak_vram_mb": 4200, "final_loss": 2.0}
    score = score_result(result, baseline)
    assert not score["pass"], ">50% NaN rate should fail"


# --- Profiling wall coverage (F-TSP-001, PMAT-483) ---


def test_profiler_wall_coverage_passes():
    """F-TSP-001: Wall coverage >= 90% should PASS profiling check."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 194, "final_loss": 4.0},
        "profiler": {
            "_profiler": "step_profiler_v1",
            "steps": 20,
            "avg_step_ms": 500.0,
            "wall_coverage": 0.95,
            "bottleneck": "memory_bw",
        },
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert score["checks"]["wall_coverage"]["pass"], "95% wall coverage should pass"


def test_profiler_wall_coverage_fails():
    """F-TSP-001: Wall coverage < 90% should FAIL profiling check."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 194, "final_loss": 4.0},
        "profiler": {
            "_profiler": "step_profiler_v1",
            "steps": 20,
            "avg_step_ms": 500.0,
            "wall_coverage": 0.72,
            "bottleneck": "launch",
        },
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert not score["checks"]["wall_coverage"]["pass"], "72% wall coverage should fail"


def test_profiler_absent_skips_check():
    """No profiler data should skip wall coverage check (not fail)."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 194, "final_loss": 4.0},
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert "wall_coverage" not in score["checks"], "No profiler = no wall_coverage check"


# --- Per-operation GEMM dominance (F-POP-002, PMAT-483) ---


def test_gemm_dominance_passes():
    """F-POP-002: GEMM >= 30% of op time should PASS."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 194, "final_loss": 4.0},
        "profiler": {"wall_coverage": 0.95, "gemm_pct": 65.0, "bottleneck": "memory_bw"},
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert score["checks"]["gemm_dominance"]["pass"], "65% GEMM should pass"


def test_gemm_dominance_fails():
    """F-POP-002: GEMM < 30% means launch overhead or transfers dominate."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 194, "final_loss": 4.0},
        "profiler": {"wall_coverage": 0.95, "gemm_pct": 15.0, "bottleneck": "launch"},
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert not score["checks"]["gemm_dominance"]["pass"], "15% GEMM should fail"


def test_gemm_dominance_absent_skips():
    """No gemm_pct should skip check."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 194, "final_loss": 4.0},
        "profiler": {"wall_coverage": 0.95},
    }
    baseline = {"tokens_per_sec": 40, "final_loss": 20.0}
    score = score_result(result, baseline)
    assert "gemm_dominance" not in score["checks"]


# --- Phase A Provable Contracts (PMAT-504) ---

APR_BASELINE = {"tokens_per_sec": 470, "final_loss": 12.0}


def test_contract_convergence_loss_above_threshold_fails():
    """F-CONV-01: APR loss > 2.0 must trigger convergence FAIL."""
    result = {
        "canary": "apr",
        "backend": "wgpu",
        "metrics": {"tokens_per_sec": 470, "final_loss": 11.74},
    }
    score = score_result(result, APR_BASELINE)
    assert not score["checks"]["convergence"]["pass"], "Loss 11.74 > 2.5 should FAIL"


def test_contract_convergence_good_loss_passes():
    """F-CONV-01: APR loss < 2.0 should pass convergence check."""
    result = {
        "canary": "apr",
        "backend": "cuda",
        "metrics": {"tokens_per_sec": 470, "final_loss": 1.8},
    }
    score = score_result(result, APR_BASELINE)
    assert score["checks"]["convergence"]["pass"], "Loss 1.8 < 2.0 should PASS"


def test_contract_better_than_random_fails():
    """F-CONV-02: APR loss > ln(vocab)=11.93 means worse than random."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 12.5},
    }
    score = score_result(result, APR_BASELINE)
    assert not score["checks"]["better_than_random"]["pass"], "Loss 12.5 > 11.93 = worse than random"


def test_contract_better_than_random_passes():
    """F-CONV-02: APR loss < 11.93 is better than random."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 8.0},
    }
    score = score_result(result, APR_BASELINE)
    assert score["checks"]["better_than_random"]["pass"]


def test_contract_backward_executed_zero_fails():
    """F-BWD-01: Zero valid backward steps = training didn't happen."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 11.74, "valid_backward_steps": 0},
    }
    score = score_result(result, APR_BASELINE)
    assert not score["checks"]["backward_executed"]["pass"]


def test_contract_backward_executed_nonzero_passes():
    """F-BWD-01: Nonzero backward steps should pass."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 1.5, "valid_backward_steps": 8},
    }
    score = score_result(result, APR_BASELINE)
    assert score["checks"]["backward_executed"]["pass"]


def test_contract_metrics_quality_estimated_fails():
    """F-MET-02: Estimated metrics (loss not parsed) must FAIL."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 0, "_metrics_quality": "estimated"},
    }
    score = score_result(result, APR_BASELINE)
    assert not score["checks"]["metrics_quality"]["pass"]


def test_contract_metrics_quality_measured_passes():
    """F-MET-02: Measured metrics should pass."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 1.5, "_metrics_quality": "measured"},
    }
    score = score_result(result, APR_BASELINE)
    assert score["checks"]["metrics_quality"]["pass"]


def test_contract_config_steps_below_100_fails():
    """F-CFG-01: Steps < 100 must FAIL (warm-up dominated, not comparable)."""
    result = {
        "canary": "apr",
        "config": {"steps": 20},
        "metrics": {"tokens_per_sec": 470, "final_loss": 1.5},
    }
    score = score_result(result, APR_BASELINE)
    assert not score["checks"]["config_steps"]["pass"]


def test_contract_config_steps_100_no_flag():
    """F-CFG-01: Steps >= 100 should NOT have config_steps check (no flag needed)."""
    result = {
        "canary": "apr",
        "config": {"steps": 100},
        "metrics": {"tokens_per_sec": 470, "final_loss": 1.5, "_metrics_quality": "measured",
                    "valid_backward_steps": 8},
    }
    score = score_result(result, APR_BASELINE)
    assert "config_steps" not in score["checks"]


def test_contract_step_time_sanity_passes():
    """F-PROF-STEP: Reasonable step time should pass."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 1.5, "_metrics_quality": "measured",
                    "valid_backward_steps": 8},
        "profiler": {
            "wall_coverage": 0.95,
            "phases": {
                "gpu_fwd": {"avg_ms": 401.5},
                "gpu_lora_bwd": {"avg_ms": 550.6},
            },
        },
    }
    score = score_result(result, APR_BASELINE)
    assert score["checks"]["step_time_sanity"]["pass"]


def test_contract_loss_improved_learning_passes():
    """F-CONV-03: Loss trajectory showing >=20% drop should pass."""
    result = {
        "canary": "apr",
        "metrics": {
            "tokens_per_sec": 470,
            "final_loss": 11.74,
            "loss_trajectory": [18.9, 9.15, 12.3, 16.3, 15.5, 12.0, 10.8, 11.74],
        },
    }
    score = score_result(result, APR_BASELINE)
    # min=9.15, first=18.9, ratio=0.484 < 0.8
    assert score["checks"]["loss_improved"]["pass"]
    assert score["checks"]["loss_improved"]["value"] < 0.8


def test_contract_loss_improved_flat_fails():
    """F-CONV-03: Flat trajectory (no improvement) should FAIL."""
    result = {
        "canary": "apr",
        "metrics": {
            "tokens_per_sec": 470,
            "final_loss": 15.0,
            "loss_trajectory": [15.0, 15.0, 15.0, 15.0],
        },
    }
    score = score_result(result, APR_BASELINE)
    # min=15.0, first=15.0, ratio=1.0 >= 0.8
    assert not score["checks"]["loss_improved"]["pass"]


def test_contract_loss_improved_diverging_fails():
    """F-CONV-03: Diverging trajectory (loss going UP) should FAIL."""
    result = {
        "canary": "apr",
        "metrics": {
            "tokens_per_sec": 470,
            "final_loss": 20.0,
            "loss_trajectory": [10.0, 15.0, 20.0],
        },
    }
    score = score_result(result, APR_BASELINE)
    # min=10.0, first=10.0, ratio=1.0 >= 0.8
    assert not score["checks"]["loss_improved"]["pass"]


def test_contract_loss_improved_single_epoch_skipped():
    """F-CONV-03: Single epoch trajectory should not fire (no baseline for comparison)."""
    result = {
        "canary": "apr",
        "metrics": {
            "tokens_per_sec": 470,
            "final_loss": 12.0,
            "loss_trajectory": [12.0],
        },
    }
    score = score_result(result, APR_BASELINE)
    assert "loss_improved" not in score["checks"]


def test_contract_loss_improved_marginal_pass():
    """F-CONV-03: 20% improvement exactly at threshold — boundary."""
    result = {
        "canary": "apr",
        "metrics": {
            "tokens_per_sec": 470,
            "final_loss": 15.0,
            "loss_trajectory": [20.0, 15.8, 15.0],  # ratio 15.0/20.0 = 0.75 < 0.8
        },
    }
    score = score_result(result, APR_BASELINE)
    assert score["checks"]["loss_improved"]["pass"]


def test_contract_loss_improved_no_trajectory_skipped():
    """F-CONV-03: No loss_trajectory field should not fire."""
    result = {
        "canary": "apr",
        "metrics": {"tokens_per_sec": 470, "final_loss": 11.74},
    }
    score = score_result(result, APR_BASELINE)
    assert "loss_improved" not in score["checks"]
