# Dataset Specification

**Parent:** [Training Canary Spec](../training-canary-spec.md)

---

## Canary Dataset

**File:** `prompts/canary-dataset.yaml`
**Size:** 50 seed samples (expandable to 500 via `scripts/expand-dataset.py`)
**Format:** YAML with instruction/response pairs

| Property | Value |
|----------|-------|
| Total samples | 50 (seed) |
| Languages | Python (35), Rust (8), SQL (2), Mixed (5) |
| Mean instruction length | ~45 tokens |
| Mean response length | ~120 tokens |
| Mean total length | ~165 tokens |
| Max total length | ~400 tokens |
| Tokenizer | Qwen2.5-Coder tokenizer (151,936 vocab) |

## Training Format

All canaries format samples as:

```
### Instruction:
{instruction}

### Response:
{response}
```

Tokenized with padding to `seq_len` (512) and truncation. The SFTTrainer (unsloth) handles this via `dataset_text_field="text"`. The PyTorch canary tokenizes in the Dataset `__init__`.

## Dataset Quality Rationale

The dataset is intentionally simple (basic algorithms, data structures, standard patterns) because the canary measures **training throughput**, not model quality. Complex datasets would introduce confounding variables:
- Variable tokenization length -> step time variance
- Hard examples -> loss spikes -> false regression signals
- Domain-specific patterns -> optimizer sensitivity

## Falsification Conditions

| ID | Condition | Action |
|----|-----------|--------|
| F-DS-01 | If any sample exceeds 512 tokens after tokenization | Truncation masks real throughput -- fix or remove sample |
| F-DS-02 | If dataset expansion (50->500) changes loss trajectory by >10% | Expansion introduces distribution shift -- use original 50 |
