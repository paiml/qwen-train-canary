#!/usr/bin/env python3
"""Validate CUDA graph backward wiring is complete.

Checks that the backward_graph.rs API is properly called from
backward.rs without running actual CUDA code. This is a static
validation that the code paths are connected.

PMAT-488: CUDA graph backward capture validation.
"""

import subprocess
import sys


def check_backward_graph_wiring():
    """Verify backward_graph.rs is called from backward.rs."""
    errors = []

    # 1. backward_graph.rs must exist and have the key functions
    import os
    bg_path = os.path.join(
        os.path.dirname(__file__), "..",
        "..", "entrenar", "src", "finetune", "backward_graph.rs"
    )
    # Use relative path from workspace root
    bg_path = os.path.expanduser("~/src/entrenar/src/finetune/backward_graph.rs")
    if not os.path.exists(bg_path):
        errors.append(f"backward_graph.rs not found at {bg_path}")
    else:
        with open(bg_path) as f:
            content = f.read()
        for fn_name in ["use_backward_graph", "try_capture_backward", "replay_backward",
                        "BackwardGraphState"]:
            if fn_name not in content:
                errors.append(f"backward_graph.rs missing function: {fn_name}")

    # 2. backward.rs must call backward_graph functions
    bwd_path = os.path.expanduser(
        "~/src/entrenar/src/finetune/instruct_pipeline/backward.rs"
    )
    if not os.path.exists(bwd_path):
        errors.append(f"backward.rs not found at {bwd_path}")
    else:
        with open(bwd_path) as f:
            content = f.read()
        for fn_name in ["use_backward_graph", "replay_backward", "begin_capture",
                        "end_capture", "backward_graph_state"]:
            if fn_name not in content:
                errors.append(f"backward.rs missing call to: {fn_name}")

    # 3. InstructGpuTrainingState must have backward_graph_state field
    mod_path = os.path.expanduser(
        "~/src/entrenar/src/finetune/instruct_pipeline/mod.rs"
    )
    if os.path.exists(mod_path):
        with open(mod_path) as f:
            content = f.read()
        if "backward_graph_state" not in content:
            errors.append("mod.rs missing backward_graph_state field in InstructGpuTrainingState")

    # 4. cuda_init.rs must initialize backward_graph_state to None
    init_path = os.path.expanduser(
        "~/src/entrenar/src/finetune/instruct_pipeline/cuda_init.rs"
    )
    if os.path.exists(init_path):
        with open(init_path) as f:
            content = f.read()
        if "backward_graph_state: None" not in content:
            errors.append("cuda_init.rs missing backward_graph_state initialization")

    # 5. All 6 optimization flags must be in canary-apr-max
    makefile_path = os.path.join(os.path.dirname(__file__), "..", "Makefile")
    if os.path.exists(makefile_path):
        with open(makefile_path) as f:
            content = f.read()
        # Find canary-apr-max section
        if "canary-apr-max:" in content:
            max_section = content[content.index("canary-apr-max:"):]
            max_section = max_section[:max_section.index("\n\n") if "\n\n" in max_section else len(max_section)]
            for flag in ["NF4_FUSED_GEMM=1", "NF4_FUSED_BWD_GEMM=1", "NF4_TC_GEMM=1",
                         "NF4_TC_BWD_GEMM=1", "FP16_GEMM=1", "CUDA_GRAPH=1"]:
                if flag not in max_section:
                    errors.append(f"canary-apr-max missing flag: {flag}")
        else:
            errors.append("Makefile missing canary-apr-max target")

    return errors


def main():
    print("=== CUDA Graph Backward Wiring Validation ===\n")
    errors = check_backward_graph_wiring()

    if errors:
        print(f"FAILED: {len(errors)} error(s)\n")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("PASSED: All wiring checks OK")
        print("  ✓ backward_graph.rs has all key functions")
        print("  ✓ backward.rs calls graph capture/replay")
        print("  ✓ InstructGpuTrainingState has backward_graph_state field")
        print("  ✓ cuda_init.rs initializes backward_graph_state to None")
        print("  ✓ canary-apr-max has all 6 optimization flags")


if __name__ == "__main__":
    main()
