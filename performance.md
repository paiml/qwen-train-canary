# Training Canary Performance

Generated: 2026-03-31T12:33:04.620757

| Canary | Backend | Host | tok/s | VRAM (MB) | Loss | Wall (s) |
|--------|---------|------|-------|-----------|------|----------|
| cublas (default) | cuda | gx10-a5b5 | 4009.5 | 49777 | 0.0139 | 102.16 |
| cublas (forced) | cuda | gx10-a5b5 | 4026.8 | 49778 | 0.0139 | 101.72 |
| pytorch | cuda | gx10-a5b5 | 4055.4 | 50580 | 0.0087 | 202.0 |
| unsloth | cuda | yoga | 6715.7 | 3515 | 0.152 | 30.5 |
| unsloth | cuda | yoga | 6660.6 | 3515 | 0.152 | 30.75 |
| unsloth | cuda | yoga | 6715.6 | 3515 | 0.1519 | 30.5 |
| unsloth | cuda | yoga | 6695.5 | 3515 | 0.1521 | 30.59 |
| unsloth | cuda | yoga | 6695.5 | 3515 | 0.1521 | 30.59 |
