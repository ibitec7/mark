# MaRK: Markov-adapted Recurrent Kernels for Hydra SSM

**MaRK** turns a *non-causal* bidirectional state-space backbone into a usable diffusion
language model by attaching a lightweight, parameter-efficient adapter that imposes a
**causal, Markov-structured time kernel** on the SSM transition. Concretely, MaRK
re-parameterises the five selective-scan parameters of every 23-layer Hydra block
(`A`, `B`, `C`, `D`, `Δ`) from the token representation itself, and the adapter can be
instantiated with three interchangeable function families (a **HyperNetwork**, a **Chebyshev polynomial** expansion, or **DCT** fourier basis) all built on the kernel-agnostic low-rank factors shared
across layers. The model is trained with the CART-weighted diffusion-masking
(masked-LM) objective on a 14-corpus pretraining mixture, in three stages
(pretraining → intermediate fine-tuning → task-specific fine-tuning), and this branch
contains the complete rebuttal/revision package: the three-stage training pipeline and
its configs, the released adapter checkpoints, and the three ablations — a
**leave-one-out SSM-parameter ablation** on WikiText, a **CART-weighted validation
ablation** over six benchmark datasets with multi-seed confidence intervals, and an
**AdaLN-Zero vs. input-injection vs. MaRK profiling** study on A100.

---

## Contents of this branch

| What | Where |
|------|-------|
| Three-stage diffusion-masked training (Hydra + MaRK) | `src/main.py`, `src/nemo.py`, `src/hydra*.py`, `train*.sh` |
| Training / benchmark configs (3 kernels × 3 stages) | `configs/training_config_*.yaml`, `configs/benchmark_config_*.yaml` |
| Weight transfer from the released Hydra BERT | `src/transfer.py` |
| Single-pass CART validation harness | `src/perplexity.py` |
| **All three ablations** (one driver) | `src/ablation.py`, `./run_ablation.sh` |
| Paper figures / diagnostics | `analysis/` |
| Lean statement of Proposition 4.1 | `proofs/` |
| Reviewer-facing supplement | `REPRODUCIBILITY.md` |

```text
mark/
├── configs/                 # training_config_*, benchmark_config_*, hydra.yaml, guider_config.yaml
├── data/
│   ├── benchmarks/          # one subdirectory per eval dataset (*.parquet with an `input_ids` column)
│   ├── train_shards1|2|3/   # per-stage packed training shards
│   ├── val_shards/          # packed validation shards
│   └── ablation_results/    # committed ablation outputs (CSV + Markdown + per-seed JSON)
├── models/                  # checkpoints (not tracked — see §2.2)
├── src/
│   ├── main.py              # training entry point (`python -m src.main`)
│   ├── ablation.py          # `loo` | `cart` | `profiling` | `all` suites
│   ├── perplexity.py        # single-pass CART validation
│   ├── transfer.py          # Hydra-BERT → HydraForMaskedLM weight transfer
│   ├── utils.py             # shard downloader/packer, dataloaders
│   ├── data.py, prepare_data.py  # WikiText and raw-corpus packing
│   └── performance.py       # kernel-level profiling utilities
├── train.sh                 # all 3 kernels × 3 stages, end to end
├── train_{dct,hypernet}_all_stages.sh
├── run_ablation.sh          # Docker wrapper around src.ablation
└── REPRODUCIBILITY.md       # reviewer-facing walkthrough
```

---

## 1. Setup

### Docker (recommended)

`Dockerfile.nemo` is the GPU training image: NVIDIA's NeMo base plus `causal-conv1d`
and `mamba-ssm` built from source.

```bash
docker build -f Dockerfile.nemo -t mark:latest .
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/mark -w /workspace/mark -it mark:latest bash
```

`./run_ablation.sh` wraps the same `docker run` invocation for every ablation suite
(see §2.4), so you normally do not need to type the mounts by hand.

### Local, with `uv`

We target Python 3.12–3.13. An NVIDIA GPU with a recent driver is required — the
evaluation and training paths move models to CUDA and have no CPU fallback.

```bash
uv sync          # or: pip install -r requirements.txt
```

All commands below are written for bare metal as `python -m …`; prefix them with
`uv run` if you are using the `uv` environment.

---

## 2. Reproducibility

### 2.1 Get the data

Everything the paper reports is driven by three kinds of data: **benchmark packs** for
the ablations, the **pretraining shard mixture**, and **WikiText**.

#### (a) Benchmark packs — `data/benchmarks/`

Every ablation dataset is a directory of parquet files whose only required column is
`input_ids` (packed, fixed-length token chunks; extra columns are stripped on load):

```text
data/benchmarks/ag_news/*.parquet
data/benchmarks/arxiv/*.parquet
data/benchmarks/lambada/*.parquet
data/benchmarks/ptb/*.parquet
data/benchmarks/pubmed/*.parquet
data/benchmarks/wikitext/*.parquet
```

Download the released validation-dataset bundle from the anonymous Hugging Face
repository listed in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and unpack it so the
layout above exists. Datasets are discovered by directory scan, so a subset works
too — the harness simply evaluates what it finds. All packs were tokenized with
`bert-base-uncased`; the first run downloads that tokenizer unless it is cached.

To build a pack yourself from raw text, put the raw parquet files (with a `text`
column) under the dataset directory and run the packer in `src/prepare_data.py`
(edit `directories` at the bottom of the file to point at your corpora); it
tokenizes, chunks to `seq_length=4096`, stamps CLS/SEP, and writes `packed_*.parquet`.

#### (b) Pretraining mixture — `data/train_shards*`

`src/utils.py::download_dataset` streams the 14 corpora below straight from the Hugging
Face Hub, shuffles each with seed 42, tokenizes with `bert-base-uncased`, packs to
`max_length=4096`, and writes 10 000-row shards:

```
common-pile/stackexchange_filtered      common-pile/project_gutenberg_filtered
common-pile/libretexts_filtered         common-pile/arxiv_papers_filtered
common-pile/youtube_filtered            common-pile/news_filtered
common-pile/pubmed_filtered             common-pile/doab_filtered
common-pile/cccc_filtered               common-pile/pressbooks_filtered
iohadrubin/wikitext-103-raw-v1          common-pile/data_provenance_initiative_filtered
common-pile/wikimedia_filtered          common-pile/arxiv_abstracts_filtered
```

The sampling ratio of each corpus is fixed in the `ratios` list next to the dataset
list. The function has no CLI — run it from the repo root:

```bash
python -c "from src.utils import download_dataset; download_dataset(max_length=4096)"
```

It is resumable: existing `shard_*.parquet` files in `./data/train_shards` are counted
and new shards continue from there. A final pass randomly splits the pool 70/15/15
into `./data/train_shards`, `./data/val_shards` and `./data/test_shards`.

The shipped configs read a **per-stage** directory, so point the trainer at the
mixture you want for each stage (either edit `train_data_dir` in the stage config or
symlink/copy the shard pool):

```text
configs/training_config_*_stage1.yaml   train_data_dir: ./data/train_shards1
configs/training_config_*_stage2.yaml   train_data_dir: ./data/train_shards2
configs/training_config_*_stage3.yaml   train_data_dir: ./data/train_shards3
all stages                              val_data_dir:   ./data/val_shards
```

#### (c) WikiText-2 and WikiText-103

`src/data.py` pulls both WikiText variants from `Salesforce/wikitext` over the `hf://`
parquet protocol, writes `./data/wikitext-2-v1` and `./data/wikitext-103-v1`, and
filters rows without word characters. It authenticates with `token.json` in the repo
root:

```bash
echo '{"token": "hf_your_read_token"}' > token.json
python -m src.data
```

> Note: the download writes to `./data/...` but the cleaning pass at the bottom of
> `src/data.py` reads `../data/...`. Run it from a directory whose parent contains
> `data/`, or change that one path.

#### (d) Helper for awkward shards

`src/utils.py` exposes `packing` and `tokenize_fast`, which is what
`src/prepare_data.py` uses. `src/repair_data.py` strips incompatible embedded Hugging
Face parquet metadata from third-party shards if a `load_dataset` call chokes on it.

### 2.2 Get the checkpoints

Place these under `models/`:

| Path | Meaning |
|------|---------|
| `models/hydra_bert_23layers.pt` | released Hydra BERT parent model (from [`goombalab/hydra`](https://huggingface.co/goombalab/hydra)) |
| `models/hydra_bert_23layers_mark_base.pt` | Hydra BERT remapped into `HydraForMaskedLM` — used by every `benchmark_config_*` |
| `models/hydra_{hypernet,chebyshev,dct}_mark.pt` | the same remapped base, one per kernel — used as the stage-1 `weights_path` for training |
| `models/hydra_mark_{hypernet,chebyshev,dct}/best_hydra_mark.ckpt` | the three released MaRK adapters evaluated in the paper |

The remap is **kernel-agnostic** (`src/transfer.py` only moves embeddings, the 23
encoder layers and the prediction head), so the per-kernel `.pt` files are copies of
the same derived artifact:

```bash
python -m src.transfer --source models/hydra_bert_23layers.pt \
  --config configs/benchmark_config_chebyshev_stage1.yaml \
  --output models/hydra_bert_23layers_mark_base.pt

for k in hypernet chebyshev dct; do
  cp models/hydra_bert_23layers_mark_base.pt models/hydra_${k}_mark.pt
done
```

`src/ablation.py` derives `hydra_bert_23layers_mark_base.pt` automatically if it is
missing, so the ablations run with only the parent `.pt` present. Checkpoint locations
can also be overridden without touching code:

```bash
python -m src.ablation --checkpoint-dir chebyshev=/path/to/ckpts
```

### 2.3 Run the training

Each stage copies its config over `configs/training_config.yaml` and runs
`python -m src.main`, which loads `weights_path`, builds the trainer at
`precision: bf16-true` with DDP, and writes `best_hydra_mark.ckpt` (monitored on
`val_loss`) into `checkpoint_dir`. Stage 1 resumes from that directory if a checkpoint
exists; stages 2 and 3 set `ckpt_weights_only: True`, so they start from the previous
stage's best weights.

**Everything — 3 kernels × 3 stages:**

```bash
./train.sh
```

**One kernel at a time:**

```bash
./train_hypernet_all_stages.sh
./train_dct_all_stages.sh
# Chebyshev: run the three stage configs directly, as below
```

**A single stage, manually:**

```bash
cp configs/training_config_hypernet_stage1.yaml configs/training_config.yaml   # ← your stage config
python -m src.main
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage1
```

Notes:

- The **stage-1 `weights_path` must exist first** — create
  `models/hydra_<kernel>_mark.pt` as in §2.2, or `python -m src.main` fails on
  `torch.load`.
- **W&B** is initialised at the top of `src/main.py` under the `wandb` key of the
  config. It is optional: if `wandb.init` fails, the run logs a warning and continues
  without tracking. Use `WANDB_MODE=offline` to keep it silent.
- `batch_size: 32`, `pad_length: 4096`, `matmul_precision: high`, `epochs: 3/5/3` and
  `total_steps: 40000` in the stage configs are the released A100 settings; multi-GPU
  is `trainer.devices: -1` with `strategy: ddp`.
- CART weighting is on for every stage (`cart: true`, `cart_p: 0.45`,
  `cart_scale: 1.0`) — this is the objective the evaluations in §2.4 use.
- Training logs to `logs/` and `data/training/hydra_train_metrics.parquet`.

### 2.4 Run the ablations

`src/ablation.py` is the single driver for all three ablations. `--suite` picks one,
`--suite all` runs them in sequence; `--help` lists every flag.

```bash
python -m src.ablation --help
```

**Run everything:**

```bash
python -m src.ablation --suite all --seeds 10
```

#### 1. Leave-one-out SSM-parameter ablation (`loo`) — WikiText

Freezes one subset of the five modulated SSM parameters (`A`, `B`, `C`, `Δ`, `D`) and
measures the CART-weighted loss on WikiText over 10 seeds.

```bash
python -m src.ablation --suite loo --seeds 10
python -m src.ablation --suite loo --modes full all_except_A A_only \
  --kernels chebyshev dct hypernet --seeds 10
```

| Mode | Modulated |
|------|-----------|
| `full` | A, B, C, Δ, D |
| `all_except_A` *(default)* | B, C, Δ, D |
| `A_only` | A |
| `dt_only` | Δ |
| `BC_only` | B, C (Mamba-style selection) |
| `all_except_dt` | A, B, C, D |
| `D_only` | D |
| `none` | none (base Hydra) |

Data comes from `--wikitext-dir` (default `data/wikitext`), i.e. packed parquet with an
`input_ids` column.

#### 2. CART-weighted validation benchmarks (`cart`) — all datasets

Runs the training-matching evaluator (NeMo `Trainer.validate` with diffusion masking
and CART weights) for every kernel × mode × seed on every dataset under
`data/benchmarks/`:

```bash
python -m src.ablation --suite cart --seeds 10 --limit-val-batches 500
python -m src.ablation --suite cart --datasets wikitext ptb lambada --seeds 10
```

- **The reported metric is `weighted_nll` — the CART loss.** `weighted_ppl` is
  `exp(weighted_nll)` and `weighted_bpb` is the bits-per-byte view; `raw_*` is the
  unweighted counterpart. (Earlier revisions of the summary tables showed perplexity
  by mistake; the Markdown now headlines the CART loss.)
- `--limit-val-batches N` caps each dataset at **N validation batches** (an absolute
  count, not a fraction). It is required in practice: at the released
  `batch_size: 1` / `pad_length: 4096`, arXiv (78 463 packed sequences) and PubMed
  (143 414) would each take hours per evaluation. Datasets with fewer than N batches
  still run in full. The committed results use 500.
- `--cart-rng-protocol {loo,reseed}` controls RNG positioning, and it matters.
  Validation draws from the global RNG every batch (`sample_timestep` →
  `torch.randint`, `masking_process` → `torch.rand`) and so does building the model
  (Hydra initialises `dt_bias` with `torch.rand`). The default `loo` captures the RNG
  state right after the model is built and restores it before each dataset, rebuilding
  the dataloader where the LOO harness does — draw-for-draw identical to
  `_evaluate_single`, so WikiText here matches the `loo` table. `reseed` re-seeds per
  dataset instead (order-invariant, but **not** comparable with the LOO numbers).
- `--no-dataloader-cache` disables loader reuse (only meaningful under `reseed`).
- `--cart-report-only` re-aggregates the CSV/Markdown from an existing `cart_raw.csv`
  without touching the GPU.

#### 3. Profiling ablation (`profiling`) — AdaLN-Zero vs. input injection vs. MaRK

Delegates to the companion `experiment` checkout, which owns the canonical
implementation (`profiling.profile_multiseed`), then copies the reportable artifacts
back into this repo:

```bash
git clone https://github.com/ibitec7/experiment.git ~/Desktop/experiment
python -m src.ablation --suite profiling \
  --experiment-dir ~/Desktop/experiment \
  --profiling-seeds 15 --profiling-mode both \
  --profiling-batch-sizes 1 4 8 --profiling-seq-len 512
```

`--profiling-mode adapter` measures the adapter path only, `e2e` the full model
forward/backward, `both` (default) runs both. `--profiling-latency-n` (100),
`--profiling-warmup` (20) and `--profiling-tag` control the measurement loop. The
published A100 numbers were produced with the defaults above.
`--profiling-summarize-only` re-summarizes existing artifacts.

#### Outputs

| Suite | Artifacts in `--output-dir` (default `data/ablation_results/`) |
|-------|-----------|
| `loo` | `ablation_results.csv`, `ablation_summary.md`, `<kernel>_<mode>_seed<seed>.json` |
| `cart` | `cart_raw.csv` (per-seed metrics), `cart_benchmark_results.csv` (mean / std / 95% CI), `cart_benchmark_summary.md`, `cart_<kernel>_<dataset>_<mode>_seed<seed>.json` |
| `profiling` | `profiling_summary.md`, `profiling/<tag>/…` |

**Protocol shared by `loo` and `cart`:** seed *i* is `42 + i × 100`; aggregation uses
the sample standard deviation (`ddof=1`) and a 95% confidence interval of
`mean ± 1.96 × SEM`, the same estimator as everywhere else in the paper.

#### Docker wrapper

`./run_ablation.sh` mounts the repo at `/workspace/mark`, exposes the GPU, forwards
every extra argument to `python -m src.ablation`, and (for the `profiling`/`all`
suites) also mounts the companion checkout:

```bash
./run_ablation.sh --suite loo  --seeds 10
./run_ablation.sh --suite cart --seeds 10 --limit-val-batches 500
./run_ablation.sh --suite profiling --profiling-seeds 15
./run_ablation.sh --suite all  --seeds 10
```

Environment overrides: `IMAGE` (default `mark:latest`), `EXPERIMENT_DIR`
(default `~/Desktop/experiment`), `REPO` (default: the script's directory), and
`NO_BUILD=1` to fail instead of building the image when it is missing.

### 2.5 Figures and diagnostics

```bash
python -m analysis.aqs_certificate                      # AQS certificate
python analysis/hypernet_synthetic_lpv.py               # synthetic LPV recovery
python analysis/chebyshev_synthetic_lpv.py
python analysis/dct_synthetic_lpv.py
python -m analysis.dynamics_analysis                    # Markov norm vs. lag (K = 4096)
```

They write `plots/*.png` and audit files under `analysis/results/`. The AQS
certificate and the dynamics figure read `A_log` / `dt_bias` out of
`models/hydra_bert_23layers.pt` and the adapter checkpoints listed in §2.2.

The Lean statement of Proposition 4.1 is in `proofs/` (`cd proofs && lake build`).

---

## 3. Results shipped in this branch

`data/ablation_results/` contains the outputs of the runs described above.

**CART loss (`weighted_nll`) across the six benchmark datasets, 10 seeds**
(mean ± 95% CI, `--limit-val-batches 500`, `--cart-rng-protocol loo`):

| Kernel | ag_news | arxiv | lambada | ptb | pubmed | wikitext | mean |
|--------|---------|-------|---------|-----|--------|----------|------|
| chebyshev | 2.849 ± 0.139 | 2.553 ± 0.058 | 2.395 ± 0.098 | 2.419 ± 0.257 | 2.554 ± 0.057 | 2.547 ± 0.155 | **2.553** |
| dct | 2.860 ± 0.142 | 2.614 ± 0.065 | 2.471 ± 0.109 | 2.477 ± 0.271 | 2.564 ± 0.063 | 2.552 ± 0.138 | **2.590** |
| hypernet | 3.909 ± 0.198 | 4.124 ± 0.102 | 3.704 ± 0.171 | 3.390 ± 0.306 | 3.937 ± 0.097 | 3.733 ± 0.228 | **3.800** |

The WikiText column is directly comparable with the `full` rows of the leave-one-out
ablation (`chebyshev 2.5465`, `dct 2.5520`, `hypernet 3.7333`); the residual ≤1e-3 is
GPU bf16/tf32 run-to-run jitter, the same magnitude as re-running the LOO path itself.

Also committed: the full leave-one-out sweep
(`data/ablation_results/ablation_results.csv`, 8 modes × 3 kernels × 10 seeds), the
profiling study (`profiling_summary.md`, `profiling/a100_neurips_512/`) and every
per-seed JSON behind the tables.

---

## 4. Tests

```bash
pytest tests/                    # or: pytest tests/test_cart.py tests/test_transfer.py
```

`tests/` covers the kernels, weight transfer, packing/padding, the NeMo module, the
training loop, and a `src/perplexity.py` smoke test that runs off `data/benchmarks/ptb`.

---

## 5. Reference

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — the reviewer-facing walkthrough, with the
  download links for the released datasets and checkpoints, and the exact commands
  behind each table and figure.
- [`PROFILING.md`](PROFILING.md) — the PyTorch profiler utilities in `src/performance.py`.
- [`analysis/README.md`](analysis/README.md) — which analysis generators are paper-facing.
