# Reproducibility supplement

We prepared this note for reviewers who already have the PDF and want to rerun the experiments we actually rely on in the submission, without wading through the rest of the training codebase.

A short map of what matters:

- **Figures** (AQS certificate, synthetic LPV recovery, Markov norm vs. lag): small `analysis/` scripts that write PNGs under `plots/` and a few CSV/JSON audit files under `analysis/results/`.
- **Main diffusion-language-model ablation table** (CART-weighted validation): `src/perplexity.py`, which runs the same NeMo validation path we used during training and writes one JSON file per model × dataset under `data/benchmark_results/`, plus a `summary.json`.
- **Formal statement** of Proposition 4.1: Lean code under `proofs/` (optional if you only care about experiments).

Everything below assumes you cloned this repository and are sitting at its root.

> **_NOTE:_**  Please read the README.md for this project on how to get setup with the dependencies for this project. We recommend using docker as it is the least likely for any conflicts but if you want to run locally then you can use uv.

---

## What to download and where to put it

### Checkpoints for `src/perplexity.py`

The harness loads a **kernel-specific frozen Hydra base** from a `.pt` file, then overlays the **Lightning adapter checkpoint** for that kernel. We only release the three checkpoints that correspond to the ablation table.

Place these files:

```text
models/hydra_hypernet_mark.pt
models/hydra_chebyshev_mark.pt
models/hydra_dct_mark.pt

models/hydra_mark_hypernet_apr/best_hydra_mark.ckpt
models/hydra_mark_chebyshev_apr/best_hydra_mark.ckpt
models/hydra_mark_dct_apr/best_hydra_mark.ckpt
```

If your filenames differ, rename or symlink so the paths above exist exactly. The script also searches `./models/hydra_mark_<kernel>/` before `./models/hydra_mark_<kernel>_apr/`; we standardize on the `_apr` layout because that is what ships with our release bundles.

You can override where we look for checkpoints without editing code:

```bash
uv run python -m src.perplexity --checkpoint-dir chebyshev=/path/to/ckpts ...
```

Repeat `--checkpoint-dir` for multiple kernels or multiple candidate directories per kernel.

### Benchmark parquet packs

`src/perplexity.py` scans `data/benchmarks/` for **one subdirectory per dataset**, each containing at least one `*.parquet` file. Every parquet must expose an `input_ids` column (fixed-length token chunks). Extra columns are stripped automatically.

We used the names `ptb`, `wikitext`, `lambada`, `ag_news`, and `arxiv` to match the paper table. A minimal layout:

```text
data/benchmarks/ptb/*.parquet
data/benchmarks/wikitext/*.parquet
data/benchmarks/lambada/*.parquet
data/benchmarks/ag_news/*.parquet
data/benchmarks/arxiv/*.parquet
```

If you only have a subset, the script still runs; it simply evaluates what it finds.

**Tokenizer cache:** the model and dataloader path use `bert-base-uncased`. The first run will download tokenizer assets from Hugging Face unless they are already cached; allow network access or vendor the cache yourself.

### Checkpoints for the figure scripts

These paths are hard-coded in the analysis code:

```text
models/hydra_bert_23layers.pt                    # AQS certificate (reads A_log / dt_bias)
models/hydra_mark_hypernet_apr/best_hydra_mark.ckpt
models/hydra_mark_chebyshev_apr/best_hydra_mark.ckpt
models/hydra_mark_dct_apr/best_hydra_mark.ckpt   # dynamics figure
```

The synthetic LPV scripts generate their own data; they do not need extra downloads.

---

## Environment (local machine)

We target **Python 3.12 or 3.13** and install dependencies with `uv`:

```bash
uv sync
```

All of the commands below work as `uv run …` from the repo root.

You need an **NVIDIA GPU** with a recent driver. `src/perplexity.py` moves models to CUDA inside `_load_model_and_trainer`; there is no CPU fallback in that path.

---

## CART-weighted validation / ablation table — `src/perplexity.py`

This is the evaluator that matches the training stack: NeMo `Trainer.validate`, diffusion masking, and **CART weights** when `cart: true` in the `configs/benchmark_config_*` YAMLs.

**Default CLI** runs **Hypernet, Chebyshev, and DCT** (one checkpoint each) across every dataset folder under `data/benchmarks/`:

```bash
uv run python -m src.perplexity
```

Outputs land in `data/benchmark_results/`:

- One JSON per run: `data/benchmark_results/<kernel>_stage1_<dataset>.json`
- Aggregated copy: `data/benchmark_results/summary.json`

Each JSON contains `raw_*` and `weighted_*` metrics. For the CART-weighted objective discussed in the paper, read **`weighted_ppl`** (and the corresponding `weighted_nll` / `weighted_bpb` if you prefer NLL or bits-per-byte). `raw_ppl` is the unweighted counterpart.

For a quick sanity check without waiting for full passes, you can cap validation batches:

```bash
uv run python -m src.perplexity --limit-val-batches 10
```

If you need a subset of kernels (for example a single row of the table), import `main` and `ModelEntry` in a short Python snippet and pass `models=[...]`; the defaults are the three released checkpoints above.

---

## Figures and diagnostics — `analysis/`

Run from the repo root.

**AQS certificate**

```bash
uv run python -m analysis.aqs_certificate
```

Writes `plots/aqs_certificate.png` and `analysis/results/aqs_certificate/`.

**Synthetic LPV recovery (three panels)**

```bash
uv run python analysis/hypernet_synthetic_lpv.py
uv run python analysis/chebyshev_synthetic_lpv.py
uv run python analysis/dct_synthetic_lpv.py
```

Writes `plots/*_synthetic_recovery.png` and `analysis/results/synthetic_identifiability/`.

**Markov norm vs. lag (K = 4096)**

```bash
uv run python -m analysis.dynamics_analysis
```

Writes `plots/markov_norm_vs_lag_k4096.png` and `analysis/results/dynamics/markov_norm_vs_lag_k4096.csv`.

`analysis/README.md` is our internal checklist of which generators are “paper-facing”; you can treat it as a companion to this file.

---

## Lean formalization (optional)

If you want to check the mechanized statement of Proposition 4.1:

```bash
cd proofs
lake build
```

---

## Docker workflow (`Dockerfile.nemo`)

The Dockerfile is our **GPU training image**: NVIDIA’s NeMo base plus `causal-conv1d` and `mamba-ssm` built from source. It does **not** pip-install this repository automatically, so treat it as a CUDA-ready shell and install our deps on top.

Build (from the repo root):

```bash
docker build -f Dockerfile.nemo -t mark-nemo .
```

Run with the repo and your weights mounted at `/workspace` (adjust host paths):

```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  mark-nemo bash
```

Inside the container, install Python dependencies. The simplest path that matches our metadata is still `uv` (install it per the upstream instructions) and then:

```bash
uv sync
```

If you prefer plain pip, install the packages listed in `pyproject.toml` yourself, but keep versions close — NeMo and Lightning are picky about their stack.

Then run the same commands as on bare metal, for example:

```bash
uv run python -m src.perplexity --limit-val-batches 50
uv run python -m analysis.aqs_certificate
```

Mount a host directory that contains `models/` and `data/benchmarks/` so you are not baking multi-gigabyte weights into the image.

---

## Noise you can safely ignore for the submission

Extra training configs, ad-hoc plots, and helper scripts that are not listed above are leftovers from development. They are not required to regenerate the figures or the `perplexity.py` table.

If something is missing on your machine (a parquet pack, a tokenizer download, or a checkpoint path), the failure mode is usually an immediate `FileNotFoundError` with the exact path we expected — symlink or copy the artifact there and rerun.
