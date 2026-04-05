#!/usr/bin/env python3
"""Instrument wgpu_pipeline.rs with WgpuStepProfiler (PMAT-496)."""
import re
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "src/finetune/wgpu_pipeline.rs"

with open(filepath) as f:
    content = f.read()

changes = 0

# 1. Add import
old_use = "use trueno::backends::gpu::{wgpu, WgslForwardPass};"
new_use = (
    "use trueno::backends::gpu::{wgpu, WgslForwardPass};\n"
    "use super::wgpu_step_profiler::{WgpuStepProfiler, "
    "DATA_PREP, EMBED, BUF_ALLOC, BUF_WRITE, FWD_ENCODE, FWD_SUBMIT, "
    "GPU_FWD, GPU_LM, GPU_CE, GPU_LM_BWD, GPU_LORA_BWD, SYNC, OVERHEAD};"
)
if old_use in content:
    content = content.replace(old_use, new_use)
    changes += 1

# 2. Add profiler field to struct (first occurrence only)
if "profiler: WgpuStepProfiler," not in content:
    content = content.replace("    fwd: WgslForwardPass,", "    fwd: WgslForwardPass,\n    profiler: WgpuStepProfiler,", 1)
    changes += 1

# 3. Initialize profiler in constructor — find "Self {" block with "fwd,"
m = re.search(r"(Self \{[^}]*?fwd),", content)
if m and "profiler:" not in content[m.start():m.end()+200]:
    pos = m.end()
    content = content[:pos] + "\n            profiler: WgpuStepProfiler::new(true)," + content[pos:]
    changes += 1

# 4. Replace t0 with profiler.begin_step()
content = content.replace(
    "        let t0 = std::time::Instant::now();",
    "        self.profiler.begin_step();"
)

# 5. Replace t1 with profiler.begin(BUF_WRITE)
content = content.replace(
    "        let t1 = std::time::Instant::now();",
    "        self.profiler.begin(BUF_WRITE);"
)

# 6. Before forward loop — begin GPU_FWD
content = content.replace(
    "        let mut _saved_activations = Vec::with_capacity(self.num_layers);",
    "        self.profiler.begin(GPU_FWD);\n        let mut _saved_activations = Vec::with_capacity(self.num_layers);"
)

# 7. After forward loop — begin GPU_LM
content = content.replace(
    "        let t2 = std::time::Instant::now();",
    "        self.profiler.begin(GPU_LM);"
)

# 8. Before CE forward
content = content.replace(
    "        let t3a = std::time::Instant::now();\n        self.cross_entropy.forward_async(",
    "        self.profiler.begin(GPU_CE);\n        self.cross_entropy.forward_async("
)

# 9. Remove t3b
content = content.replace(
    "        let t3b = std::time::Instant::now();\n        // Fused CE backward",
    "        // Fused CE backward"
)

# 10. After CE backward — begin GPU_LM_BWD
content = content.replace(
    "        let t3c = std::time::Instant::now();",
    "        self.profiler.begin(GPU_LM_BWD);"
)

# 11. After lm_head backward — begin GPU_LORA_BWD
content = content.replace(
    "        let t4 = std::time::Instant::now();",
    "        self.profiler.begin(GPU_LORA_BWD);"
)

# 12. Before read_loss — SYNC phase
content = content.replace(
    "        // Read loss from GPU AFTER all backward",
    "        self.profiler.begin(SYNC);\n        // Read loss from GPU AFTER all backward"
)

# 13. Replace old PROFILE eprintln
old_profile = (
    '        eprintln!(\n'
    '            "[PROFILE] step: {:.0}ms (embed={:.0} fwd={:.0} lm={:.0} '
    'ce={:.0}[fwd={:.0} bwd={:.0}] lm_bwd={:.0} lora_bwd={:.0})",\n'
)
if old_profile in content:
    # Find the full eprintln block ending with ");\n"
    start = content.index(old_profile)
    # Find the closing ");" of the eprintln
    end = content.index(");", start + len(old_profile))
    end += 2  # include ");"
    content = content[:start] + "        self.profiler.finish_step();" + content[end:]
    changes += 1
else:
    # Try to find it with different whitespace
    print("WARN: could not find old PROFILE eprintln block")

# 14. Remove unused t variables
for var in [
    "let t3 = std::time::Instant::now();",
    "let _t2a = std::time::Instant::now();",
    "let t2b = std::time::Instant::now();",
    "let _t2c = t2b;",
    "let t5 = std::time::Instant::now();",
]:
    content = content.replace("        " + var + "\n", "")

with open(filepath, "w") as f:
    f.write(content)
print(f"OK: {changes}+ changes applied to {filepath}")
