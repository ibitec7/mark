# MaRK SSM Parameter Ablation — Validation Loss on WikiText

> **For Hermes:** Execute task-by-task. Implement only — don't run code on RTX 3050; commands below are for A100 80GB.

**Goal:** Implement Reviewer Axfu's requested ablation (Weakness #2, Q2): isolate contribution of modulating the recurrence parameter A by running `all_except_A` mode (freeze A, modulate B, C, D, Δ) vs `full` (modulate all 5 params) for all 3 MaRK variants on WikiText. Also include all other ablation modes for completeness.

**Architecture:** Add an `ablation_mode` selector to `Hydra.forward()` that selectively restores original (pre-MaRK) parameter values after the adapter produces its shifts/scales. Create a standalone `src/ablation.py` script that reuses `perplexity.py`'s `_load_model_and_trainer` + `_evaluate_single`, runs 3 kernels × all modes, and saves CSV + Markdown table.

**Tech Stack:** Python, PyTorch, NeMo/Lightning, polars (for CSV), existing `src/hydra.py`, `src/perplexity.py`, `src/nemo.py`

**Git:** All changes on a new `rebuttal` branch, pushed to `origin/rebuttal`.

---

## Ablation Modes

Each mode selectively modulates only a subset of the 5 SSM parameters:

| Mode | A (recurrence) | B (input gate) | C (output gate) | dt/Δ (discretization) | D (skip) | Reviewer relevance |
|------|:-:|:-:|:-:|:-:|:-:|---|
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | Baseline — MaRK modulates all |
| `all_except_A` | ✗ | ✓ | ✓ | ✓ | ✓ | **Reviewer Q2: isolates A vs Mamba-style selection** |
| `A_only` | ✓ | ✗ | ✗ | ✗ | ✗ | Isolates A alone |
| `dt_only` | ✗ | ✗ | ✗ | ✓ | ✗ | Isolates Δ alone |
| `BC_only` | ✗ | ✓ | ✓ | ✗ | ✗ | Mamba-style selection (B,C only) |
| `all_except_dt` | ✓ | ✓ | ✓ | ✗ | ✓ | Isolates Δ contribution |
| `D_only` | ✗ | ✗ | ✗ | ✗ | ✓ | Isolates skip connection |
| `none` | ✗ | ✗ | ✗ | ✗ | ✗ | Base Hydra (no MaRK) |

---

## Task 1: Add `ablation_mode` to config, Hydra, and HydraUnpadMixer

**Files:**
- Modify: `src/hydra_model.py` (HydraForMaskedLMConfig)
- Modify: `src/hydra.py` (Hydra.__init__ + forward)
- Modify: `src/hydra_modules.py` (HydraUnpadMixer)

### Changes in `src/hydra_model.py`

Add `ablation_mode: str = "full"` to `HydraForMaskedLMConfig.__init__` and assign `self.ablation_mode = ablation_mode`.

### Changes in `src/hydra_modules.py`

In `HydraUnpadMixer.__init__`, when constructing `Hydra(...)`, add `ablation_mode=config.ablation_mode`.

### Changes in `src/hydra.py`

In `Hydra.__init__`, accept `ablation_mode: str = "full"` and store `self.ablation_mode = ablation_mode`.

In `Hydra.forward()`, BOTH paths (non-ensemble at line 334 and ensemble at line 344):

**Before** the mark call / param application, capture originals:
```python
_orig_A_log = self.A_log
_orig_dt_bias = self.dt_bias
_orig_D = self.D
_orig_B = B.clone()
_orig_C = C.clone()
```

**After** applying mark shifts, selectively restore:
```python
mode = self.ablation_mode
if mode == "full":
    pass
elif mode == "all_except_A":
    A_log = _orig_A_log
elif mode == "A_only":
    B, C, dt_bias, D = _orig_B, _orig_C, _orig_dt_bias, _orig_D
elif mode == "dt_only":
    A_log, B, C, D = _orig_A_log, _orig_B, _orig_C, _orig_D
elif mode == "BC_only":
    A_log, dt_bias, D = _orig_A_log, _orig_dt_bias, _orig_D
elif mode == "all_except_dt":
    dt_bias = _orig_dt_bias
elif mode == "D_only":
    A_log, B, C, dt_bias = _orig_A_log, _orig_B, _orig_C, _orig_dt_bias
elif mode == "none":
    A_log, B, C, dt_bias, D = _orig_A_log, _orig_B, _orig_C, _orig_dt_bias, _orig_D
```

---

## Task 2: Create `src/ablation.py` — orchestration script

**Files:**
- Create: `src/ablation.py`

### Key design:
- Reuses `perplexity.py`'s `_load_model_and_trainer` and `_evaluate_single`
- After `_load_model_and_trainer`, injects `ablation_mode` into every Hydra mixer:
```python
for layer_module in model.inner.hydra.encoder.layer:
    layer_module.layer.mixer.ablation_mode = mode
```
- Auto-sets up `data/benchmarks/wikitext/` from `data/wikitext/packed_test-00000-of-00001.parquet`
- Runs 3 kernels × 8 modes = 24 evaluations
- Saves CSV + Markdown table to `data/ablation_results/`

### Default modes to run:
```python
ABLATION_MODES = [
    "full", "all_except_A", "A_only", "dt_only",
    "BC_only", "all_except_dt", "D_only", "none",
]
```

### CLI:
```bash
python -m src.ablation \
  --wikitext-dir data/wikitext \
  --output-dir data/ablation_results \
  --checkpoint-dir chebyshev=/path/to/checkpoints ...
```

---

## Task 3: Setup WikiText data + benchmark configs

Create `data/benchmarks/wikitext/` directory with symlink/copy of `data/wikitext/packed_test-00000-of-00001.parquet`. The ablation script does this automatically on first run.

---

## Task 4: Git — rebuttal branch

```bash
git checkout -b rebuttal
git add -A
git commit -m "feat: add SSM parameter ablation suite for reviewer rebuttal"
git push origin rebuttal
```

Note: the branch must NOT include generated results or data symlinks — only source changes.

---

## Commands for A100 80GB

### Setup (once):
```bash
cd /home/admin/Desktop/mark
git fetch origin
git checkout rebuttal
# Ensure checkpoints exist at expected paths, or override:
```

### Smoke test (verify code works):
```bash
python -m src.ablation --limit-val-batches 5
```

### Full run:
```bash
python -m src.ablation \
  --wikitext-dir data/wikitext \
  --output-dir data/ablation_results
```

### With custom checkpoint directories:
```bash
python -m src.ablation \
  --checkpoint-dir chebyshev=/models/run_a \
  --checkpoint-dir hypernet=/models/run_b \
  --checkpoint-dir dct=/models/run_c
```

### Results will be in:
- `data/ablation_results/ablation_results.csv` — raw data
- `data/ablation_results/ablation_summary.md` — formatted table
