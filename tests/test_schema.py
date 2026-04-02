"""Falsification tests for canary metrics schema validation.

Implements canary-metrics-schema-v1.yaml from provable-contracts.
Validates F-MET-01 condition.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from validate_schema import validate_result
import json
import tempfile


def _write_and_validate(data: dict) -> list[str]:
    """Write data to temp file and validate."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="canary-test-", delete=False
    ) as f:
        json.dump(data, f)
        path = f.name
    try:
        return validate_result(path)
    finally:
        os.unlink(path)


GOOD_CUDA_RESULT = {
    "canary": "unsloth",
    "backend": "cuda",
    "host": "yoga",
    "timestamp": "2026-03-31T10:00:00+00:00",
    "gpu": {
        "device": "RTX 4060",
        "vram_total_mb": 8000,
        "cuda_version": "13.0",
        "compute_capability": "8.9",
    },
    "config": {
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "batch_size": 4,
        "seq_len": 512,
        "steps": 100,
        "lr": 0.0002,
        "seed": 42,
    },
    "metrics": {
        "throughput_samples_sec": 13.0,
        "tokens_per_sec": 6700.0,
        "final_loss": 0.15,
        "peak_vram_mb": 3515,
        "wall_time_sec": 30.0,
    },
}


# --- FALSIFY-SCHEMA-001: Missing canary field ---


def test_missing_canary_field():
    """Validator must reject JSON missing 'canary' field."""
    data = {**GOOD_CUDA_RESULT}
    del data["canary"]
    errors = _write_and_validate(data)
    assert any("canary" in e for e in errors), f"Expected canary error, got: {errors}"


def test_unknown_canary_value():
    """Validator must reject unknown canary type."""
    data = {**GOOD_CUDA_RESULT, "canary": "bogus"}
    errors = _write_and_validate(data)
    assert any("Unknown canary" in e for e in errors)


# --- FALSIFY-SCHEMA-002: CUDA missing gpu block ---


def test_cuda_missing_gpu():
    """Validator must reject CUDA result missing gpu block."""
    data = {**GOOD_CUDA_RESULT}
    del data["gpu"]
    errors = _write_and_validate(data)
    assert any("gpu" in e.lower() for e in errors), f"Expected gpu error, got: {errors}"


def test_wgpu_no_gpu_is_ok():
    """WGPU results may omit gpu block (EX-001)."""
    data = {
        "canary": "wgpu",
        "backend": "wgpu",
        "host": "mac-server",
        "timestamp": "2026-03-31T12:00:00+00:00",
        "config": {
            "batch_size": 4,
            "seq_len": 512,
            "steps": 100,
            "lr": 0.0002,
            "seed": 42,
        },
        "metrics": {
            "throughput_samples_sec": 20.0,
            "tokens_per_sec": 6730.0,
            "final_loss": 1.0,
        },
    }
    errors = _write_and_validate(data)
    assert len(errors) == 0, f"WGPU without gpu block should pass: {errors}"


# --- FALSIFY-SCHEMA-003: Zero throughput ---


def test_zero_throughput():
    """Validator must reject zero throughput."""
    data = {**GOOD_CUDA_RESULT}
    data["metrics"] = {**data["metrics"], "tokens_per_sec": 0.0}
    errors = _write_and_validate(data)
    assert any("tokens_per_sec" in e for e in errors)


# --- FALSIFY-SCHEMA-004: Loss out of range ---


def test_loss_above_500():
    """Validator must reject loss >= 500 (broken training)."""
    data = {**GOOD_CUDA_RESULT}
    data["metrics"] = {**data["metrics"], "final_loss": 600.0}
    errors = _write_and_validate(data)
    assert any("final_loss" in e for e in errors)


def test_negative_loss():
    """Validator must reject negative loss."""
    data = {**GOOD_CUDA_RESULT}
    data["metrics"] = {**data["metrics"], "final_loss": -1.0}
    errors = _write_and_validate(data)
    assert any("final_loss" in e for e in errors)


# --- VRAM exceeds hardware ---


def test_vram_exceeds_gpu():
    """Validator must flag VRAM exceeding GPU total."""
    data = {**GOOD_CUDA_RESULT}
    data["metrics"] = {**data["metrics"], "peak_vram_mb": 9000}
    data["gpu"] = {**data["gpu"], "vram_total_mb": 8000}
    errors = _write_and_validate(data)
    assert any("peak_vram_mb" in e for e in errors)


# --- Missing config fields ---


def test_missing_config_seed():
    """Validator must reject config missing seed."""
    data = {**GOOD_CUDA_RESULT}
    data["config"] = {k: v for k, v in data["config"].items() if k != "seed"}
    errors = _write_and_validate(data)
    assert any("seed" in e for e in errors)


# --- cuBLAS parity structure ---


def test_cublas_missing_parity():
    """Validator must reject cublas result missing parity block."""
    data = {
        "canary": "cublas",
        "backend": "cuda",
        "host": "gx10",
        "timestamp": "2026-03-31T10:00:00+00:00",
        "gpu": GOOD_CUDA_RESULT["gpu"],
        "config": GOOD_CUDA_RESULT["config"],
        "metrics": {"default": {}, "cublas": {}},
    }
    errors = _write_and_validate(data)
    assert any("parity" in e for e in errors)


# --- Good result passes all checks ---


def test_good_result_passes():
    """Complete valid result must have zero errors."""
    errors = _write_and_validate(GOOD_CUDA_RESULT)
    assert len(errors) == 0, f"Good result should pass: {errors}"
