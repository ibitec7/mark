# ================================================
# MaRK ablation driver — one entry point, three suites.
#
#   --suite loo        SSM parameter leave-one-out ablation (WikiText)
#                      → data/ablation_results/ablation_{results.csv,summary.md}
#   --suite cart       CART-weighted validation benchmarks over every dataset
#                      in data/benchmarks/ (multi-seed, 95% CI)
#                      → data/ablation_results/cart_{raw.csv,benchmark_results.csv,benchmark_summary.md}
#   --suite profiling  AdaLN-Zero vs Input-Injection vs MaRK kernel profiling
#                      (delegates to the companion `experiment` repo's
#                      profiling.profile_multiseed multi-seed suite)
#                      → data/ablation_results/profiling/…
#   --suite all        run all three suites in sequence
#
# Examples:
#   python -m src.ablation --suite loo  --seeds 10
#   python -m src.ablation --suite cart --seeds 10 --limit-val-batches 500
#   python -m src.ablation --suite profiling --profiling-seeds 15
#   python -m src.ablation --suite all  --seeds 10
# ================================================

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np

from .perplexity import (
    BENCHMARKS_DIR,
    MODEL_REGISTRY,
    _load_model_and_trainer,
    apply_checkpoint_dir_overrides,
    discover_datasets,
    resolve_runtime_path,
    SUPPORTED_KERNELS,
)
from .utils import arrow_dataloader, load_config, log_setup

LOG_FILE = os.path.join("logs", "ablation.log")
LOG_LEVEL = logging.INFO
Path("logs").mkdir(exist_ok=True)
logger = log_setup("AblationLogger", LOG_FILE, LOG_LEVEL)

# ---- Ablation modes ---------------------------------------------------------
# The reviewer (Axfu Q2) only needs all_except_A to isolate A's contribution
# vs Mamba-style selection. Other modes are available for completeness via --modes.
ABLATION_MODES = [
    "all_except_A",   # ★ Reviewer Q2: freeze A, modulate B,C,D,Δ
]

# All supported modes (for --modes override):
ALL_MODES = [
    "full",           # All 5 params modulated (baseline — already in Table 1)
    "all_except_A",   # ★ Reviewer Q2
    "A_only",         # Only A modulated
    "dt_only",        # Only Δ modulated
    "BC_only",        # Only B,C modulated (Mamba-style selection)
    "all_except_dt",  # Freeze Δ, modulate A,B,C,D
    "D_only",         # Only D (skip) modulated
    "none",           # No modulation (base Hydra)
]

# Per-mode: which params are modulated (True = modulated, False = frozen)
MODE_PARAMS = {
    "full":          {"A": True,  "B": True,  "C": True,  "dt": True,  "D": True},
    "all_except_A":  {"A": False, "B": True,  "C": True,  "dt": True,  "D": True},
    "A_only":        {"A": True,  "B": False, "C": False, "dt": False, "D": False},
    "dt_only":       {"A": False, "B": False, "C": False, "dt": True,  "D": False},
    "BC_only":       {"A": False, "B": True,  "C": True,  "dt": False, "D": False},
    "all_except_dt": {"A": True,  "B": True,  "C": True,  "dt": False, "D": True},
    "D_only":        {"A": False, "B": False, "C": False, "dt": False, "D": True},
    "none":          {"A": False, "B": False, "C": False, "dt": False, "D": False},
}

# Default mode(s) for the CART benchmark suite: evaluate the released model
# exactly as trained (all params modulated). Override with --modes to also
# sweep the leave-one-out modes across every dataset.
CART_MODES = ["full"]

# Suites selectable from the CLI.
SUITES = ("loo", "cart", "profiling", "all")

# Metrics we collect per evaluation
METRIC_KEYS = ["raw_nll", "raw_ppl", "weighted_nll", "weighted_ppl"]

# z-score for 95% CI (two-tailed)
Z_95 = 1.96

# Multi-seed distribution shared by every suite: seed i = 42 + i*100.
BASE_SEED = 42
SEED_STRIDE = 100

# Default location of the companion repository that implements the
# AdaLN-Zero / Input-Injection profiling benchmarks.
DEFAULT_EXPERIMENT_DIR = "~/Desktop/experiment"


def seed_for_index(seed_idx: int) -> int:
    """Return the seed used for the ``seed_idx``-th replicate (0-based)."""
    return BASE_SEED + seed_idx * SEED_STRIDE


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs (validation masking is seed-dependent)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inject_ablation_mode(model, mode: str) -> None:
    """Set ablation_mode on every Hydra mixer layer in the encoder."""
    encoder = model.inner.hydra.encoder
    count = 0
    for i, layer_module in enumerate(encoder.layer):
        if hasattr(layer_module, "layer") and hasattr(layer_module.layer, "mixer"):
            mixer = layer_module.layer.mixer
            if hasattr(mixer, "ablation_mode"):
                mixer.ablation_mode = mode
                count += 1
    if count == 0:
        logger.warning(
            "No Hydra mixer layers found — ablation_mode not injected. "
            "Check that the model uses non-ensemble MaRK adapters."
        )
    else:
        logger.info(f"Injected ablation_mode='{mode}' into {count} Hydra mixer layers")


def ensure_base_weights(models) -> None:
    """Materialize ``weights_path`` base checkpoints that have not been derived yet.

    The benchmark configs point at ``models/hydra_bert_23layers_mark_base.pt``,
    which is *derived* from the released ``models/hydra_bert_23layers.pt`` by
    ``src.transfer``. A freshly provisioned machine normally only ships the
    source checkpoint, so generate the derived file on demand instead of
    failing with a confusing "no such file" error.
    """
    from .transfer import transfer_weights

    source = resolve_runtime_path("models/hydra_bert_23layers.pt")

    for entry in models:
        train_config = load_config(entry.config_path, dict_config=True)
        weights_path = resolve_runtime_path(str(train_config.get("weights_path", "")))
        if not weights_path:
            continue
        if os.path.exists(weights_path):
            continue

        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Base weights '{weights_path}' referenced by {entry.config_path} are missing, "
                f"and the source checkpoint '{source}' is unavailable, so they cannot be derived."
            )

        logger.info(
            f"[{entry.name}] Derived base weights missing ({weights_path}); "
            f"generating from {source}"
        )
        transfer_weights(
            source_path=source,
            config_path=entry.config_path,
            output_path=weights_path,
        )


def _ensure_wikitext_data(wikitext_dir: str) -> str:
    """Ensure WikiText packed parquet is available as a benchmark dataset.

    Creates data/benchmarks/wikitext/ if needed by symlinking/copying
    the packed parquet from the wikitext source directory.
    """
    source = Path(wikitext_dir)
    target = Path("data/benchmarks/wikitext")

    # If target already has parquet files, use it
    if target.exists() and list(target.rglob("*.parquet")):
        return str(target.absolute())

    # Find parquet files in source
    parquet_files = list(source.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {wikitext_dir}. "
            "Run prepare_data.py first or point to data/wikitext/"
        )

    target.mkdir(parents=True, exist_ok=True)

    for pf in parquet_files:
        dst = target / pf.name
        if not dst.exists():
            try:
                dst.symlink_to(pf.absolute())
                logger.info(f"Symlinked {pf.name} → {target}/")
            except OSError:
                import shutil as _shutil
                _shutil.copy2(pf, dst)
                logger.info(f"Copied {pf.name} → {target}/")

    return str(target.absolute())


def _run_ablation_evaluation(
    entry,
    mode: str,
    wikitext_benchmark_dir: str,
    results_dir: str,
    limit_val_batches: int | float | None,
    seed: int,
) -> dict | None:
    """Run a single leave-one-out ablation evaluation on WikiText."""
    from .perplexity import _evaluate_single

    _seed_everything(seed)

    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"  [{entry.name}] mode={mode}  seed={seed}")
    logger.info(f"{'='*70}")

    # Load model fresh (ensures clean state)
    model, trainer, train_config, ckpt_path = _load_model_and_trainer(entry)

    if limit_val_batches is not None:
        trainer.limit_val_batches = limit_val_batches

    # Inject ablation mode
    _inject_ablation_mode(model, mode)

    # Determine output path (per-seed)
    output_path = os.path.join(results_dir, f"{entry.name}_{mode}_seed{seed}.json")

    try:
        metrics = _evaluate_single(
            model=model,
            trainer=trainer,
            train_config=train_config,
            ckpt_path=ckpt_path,
            dataset_name="wikitext",
            dataset_dir=wikitext_benchmark_dir,
            output_path=output_path,
        )
        if metrics:
            # Only keep relevant keys to keep per-seed files clean
            slim = {k: metrics.get(k) for k in METRIC_KEYS if k in metrics}
            logger.info(
                f"  ✓ raw_ppl={slim.get('raw_ppl', 'N/A')}, "
                f"weighted_ppl={slim.get('weighted_ppl', 'N/A')}"
            )
            return slim
        else:
            logger.warning("  ✗ No metrics produced")
            return {"error": "no_metrics"}
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return {"error": str(e)}
    finally:
        del model, trainer, train_config
        torch.cuda.empty_cache()
        gc.collect()


def _compute_statistics(seed_metrics: list[dict]) -> dict:
    """Compute mean, std, and 95% CI across seeds for each metric key.

    Uses the same estimator as the WikiText leave-one-out ablation:
    sample standard deviation with ``ddof=1`` and a normal (z=1.96)
    95% confidence interval on the standard error of the mean.

    Args:
        seed_metrics: List of metrics dicts, one per seed.

    Returns:
        dict with keys like 'weighted_ppl', 'weighted_ppl_mean', 'weighted_ppl_std',
        'weighted_ppl_ci95_low', 'weighted_ppl_ci95_high', and 'n_seeds'.
    """
    if not seed_metrics:
        return {"n_seeds": 0, "error": "no_valid_seeds"}

    # Filter out errored seeds
    valid = [m for m in seed_metrics if "error" not in m]
    n_valid = len(valid)

    if n_valid == 0:
        return {"n_seeds": len(seed_metrics), "n_valid": 0, "error": "all_seeds_failed"}

    result: dict = {"n_seeds": len(seed_metrics), "n_valid": n_valid}

    for key in METRIC_KEYS:
        values = np.array([m[key] for m in valid if key in m and m[key] is not None], dtype=np.float64)
        if len(values) == 0:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if len(values) > 1 else 0.0

        result[key] = mean           # backward-compatible: the key itself = mean
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = std
        result[f"{key}_ci95_low"] = mean - Z_95 * sem
        result[f"{key}_ci95_high"] = mean + Z_95 * sem

    return result


# =============================================================================
# Suite 1 — leave-one-out SSM parameter ablation (WikiText)
# =============================================================================

def run_ablation_suite(
    models=None,
    modes=None,
    seeds: int = 1,
    wikitext_dir: str = "data/wikitext",
    results_dir: str = "data/ablation_results",
    limit_val_batches: int | float | None = None,
    checkpoint_dir_overrides: dict[str, list[str]] | None = None,
) -> dict:
    """Run the leave-one-out ablation suite: all kernels × all modes × N seeds.

    Returns:
        dict: {kernel: {mode: aggregated_metrics_dict}}
    """
    if models is None:
        models = MODEL_REGISTRY
    if modes is None:
        modes = ABLATION_MODES

    models = apply_checkpoint_dir_overrides(models, checkpoint_dir_overrides)

    os.makedirs(results_dir, exist_ok=True)

    ensure_base_weights(models)

    # Setup WikiText data
    wikitext_benchmark_dir = _ensure_wikitext_data(wikitext_dir)
    logger.info(f"WikiText benchmark dir: {wikitext_benchmark_dir}")

    total = len(models) * len(modes) * seeds
    logger.info(
        f"Running {total} evaluations "
        f"({len(models)} kernels × {len(modes)} modes × {seeds} seeds)"
    )

    all_results: dict[str, dict[str, dict]] = {}

    for entry in models:
        kernel = entry.kernel
        logger.info(f"\n{'#'*70}")
        logger.info(f"# KERNEL: {kernel}")
        logger.info(f"{'#'*70}")

        kernel_results: dict[str, dict] = {}

        for mode in modes:
            seed_metrics: list[dict] = []

            for seed_idx in range(seeds):
                result = _run_ablation_evaluation(
                    entry=entry,
                    mode=mode,
                    wikitext_benchmark_dir=wikitext_benchmark_dir,
                    results_dir=results_dir,
                    limit_val_batches=limit_val_batches,
                    seed=seed_for_index(seed_idx),
                )
                if result:
                    seed_metrics.append(result)

            # Aggregate across seeds
            aggregated = _compute_statistics(seed_metrics)
            kernel_results[mode] = aggregated

            # Log summary
            if aggregated.get("n_valid", 0) > 1:
                wppl_mean = aggregated.get("weighted_ppl_mean", float("nan"))
                wppl_ci = aggregated.get("weighted_ppl_ci95_low", float("nan"))
                logger.info(
                    f"  [{kernel}] {mode}: weighted_ppl = {wppl_mean:.4f} "
                    f"± {wppl_mean - wppl_ci:.4f} "
                    f"(95% CI, n={aggregated['n_valid']}/{aggregated['n_seeds']})"
                )
            elif aggregated.get("n_valid", 0) == 1:
                logger.info(
                    f"  [{kernel}] {mode}: weighted_ppl = "
                    f"{aggregated.get('weighted_ppl', 'N/A')} "
                    f"(single seed)"
                )

        all_results[kernel] = kernel_results

    return all_results


# =============================================================================
# Suite 2 — CART-weighted validation benchmarks over data/benchmarks
# =============================================================================

def _evaluate_with_loader(
    model,
    trainer,
    dataset_name: str,
    dataset_dir: str,
    output_path: str,
    val_dl,
) -> dict | None:
    """Validate ``model`` on a pre-built dataloader and return the written metrics.

    Mirrors ``perplexity._evaluate_single`` but accepts an already-built
    dataloader so large benchmark sets are not re-read for every seed.
    """
    model.reset_benchmark_state()
    model.val_dir = dataset_dir
    model.benchmark_path = output_path
    model._val_dl = val_dl  # force dataloader reuse

    with torch.inference_mode():
        trainer.validate(model, val_dl, ckpt_path=None)

    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return None


def _build_val_dataloader(train_config, dataset_dir: str):
    """Build the validation dataloader used by the CART benchmark harness."""
    return arrow_dataloader(
        data_dir=dataset_dir,
        split="validation",
        batch_size=train_config.get("batch_size", 1),
        num_workers=train_config.get("num_workers", 4),
        keep_in_memory=True,
    )


def _count_dataset_batches(val_dl) -> int | None:
    """Best-effort number of batches in a dataloader (None when unknown)."""
    try:
        return len(val_dl)
    except (TypeError, AttributeError):
        return None


def run_cart_benchmark_suite(
    models=None,
    modes=None,
    seeds: int = 1,
    benchmarks_dir: str = BENCHMARKS_DIR,
    datasets: list[str] | None = None,
    results_dir: str = "data/ablation_results",
    limit_val_batches: int | float | None = None,
    checkpoint_dir_overrides: dict[str, list[str]] | None = None,
    cache_dataloaders: bool = True,
) -> dict:
    """Run the CART-weighted validation benchmark ablation.

    For every kernel × mode × seed, the released checkpoint is validated on
    every dataset under ``benchmarks_dir`` using the training-matching CART
    evaluator (NeMo ``Trainer.validate`` + diffusion masking + CART weights).
    Metrics are aggregated across seeds with the same estimator as the
    leave-one-out ablation.

    Returns:
        dict with keys:
            ``aggregated``: {kernel: {mode: {dataset: stats}}}
            ``raw``:        list of per-(kernel, mode, dataset, seed) rows
            ``datasets``:   the evaluated dataset names
            ``modes``:      the evaluated ablation modes
    """
    if models is None:
        models = MODEL_REGISTRY
    if modes is None:
        modes = CART_MODES

    models = apply_checkpoint_dir_overrides(models, checkpoint_dir_overrides)

    ensure_base_weights(models)

    if datasets is None:
        datasets = discover_datasets(benchmarks_dir)

    os.makedirs(results_dir, exist_ok=True)

    total_evals = len(models) * len(modes) * len(datasets) * seeds
    logger.info(
        f"Running {total_evals} CART benchmark evaluations "
        f"({len(models)} kernels × {len(modes)} modes × {len(datasets)} datasets × {seeds} seeds)"
    )
    logger.info(f"Datasets: {datasets}")
    if limit_val_batches is not None:
        logger.info(f"Per-dataset validation cap: {limit_val_batches} batches")

    aggregated: dict[str, dict[str, dict[str, dict]]] = {}
    raw_rows: list[dict] = []

    for entry in models:
        kernel = entry.kernel
        logger.info(f"\n{'#'*70}")
        logger.info(f"# KERNEL: {kernel}")
        logger.info(f"{'#'*70}")

        aggregated[kernel] = {}

        for mode in modes:
            logger.info(f"\n{'='*70}")
            logger.info(f"# KERNEL {kernel} — MODE {mode}")
            logger.info(f"{'='*70}")

            dataset_seed_metrics: dict[str, list[dict]] = {ds: [] for ds in datasets}

            # Pre-build dataloaders once per (mode) so large parquet sets
            # (arxiv/pubmed) are read a single time instead of once per seed.
            dl_cache: dict[str, object] = {}

            for seed_idx in range(seeds):
                seed = seed_for_index(seed_idx)
                _seed_everything(seed)

                model, trainer, train_config, ckpt_path = _load_model_and_trainer(entry)
                if limit_val_batches is not None:
                    trainer.limit_val_batches = limit_val_batches
                _inject_ablation_mode(model, mode)

                try:
                    for ds_name in datasets:
                        ds_dir = os.path.join(benchmarks_dir, ds_name)
                        output_path = os.path.join(
                            results_dir, f"cart_{entry.name}_{ds_name}_{mode}_seed{seed}.json"
                        )

                        if cache_dataloaders:
                            if ds_name not in dl_cache:
                                dl_cache[ds_name] = _build_val_dataloader(train_config, ds_dir)
                                logger.info(
                                    f"  [{entry.name}|{ds_name}] dataloader built "
                                    f"({_count_dataset_batches(dl_cache[ds_name])} batches)"
                                )
                            val_dl = dl_cache[ds_name]
                        else:
                            val_dl = _build_val_dataloader(train_config, ds_dir)

                        # Re-seed immediately before each dataset validation so the
                        # diffusion masking / timestep sampling is reproducible and
                        # identical across datasets for a given seed.
                        _seed_everything(seed)

                        try:
                            metrics = _evaluate_with_loader(
                                model=model,
                                trainer=trainer,
                                dataset_name=ds_name,
                                dataset_dir=ds_dir,
                                output_path=output_path,
                                val_dl=val_dl,
                            )
                        except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                            logger.error(f"  [{entry.name}|{ds_name}|{mode}|seed={seed}] failed: {exc}")
                            metrics = {"error": str(exc)}

                        if not metrics:
                            metrics = {"error": "no_metrics"}

                        slim = {k: metrics.get(k) for k in METRIC_KEYS if k in metrics}
                        dataset_seed_metrics[ds_name].append(slim or {"error": "no_metrics"})

                        raw_rows.append({
                            "kernel": kernel,
                            "mode": mode,
                            "dataset": ds_name,
                            "seed": seed,
                            **{k: (slim or {}).get(k) for k in METRIC_KEYS},
                            "error": metrics.get("error", ""),
                        })

                        logger.info(
                            f"  [{entry.name}|{ds_name}|{mode}|seed={seed}] "
                            f"weighted_ppl={slim.get('weighted_ppl') if slim else 'N/A'} "
                            f"raw_ppl={slim.get('raw_ppl') if slim else 'N/A'}"
                        )
                finally:
                    del model, trainer, train_config
                    torch.cuda.empty_cache()
                    gc.collect()

            aggregated[kernel][mode] = {
                ds: _compute_statistics(dataset_seed_metrics[ds]) for ds in datasets
            }

            for ds in datasets:
                agg = aggregated[kernel][mode][ds]
                if agg.get("n_valid", 0) > 1:
                    mean = agg.get("weighted_ppl_mean", float("nan"))
                    half = mean - agg.get("weighted_ppl_ci95_low", float("nan"))
                    logger.info(
                        f"  [{kernel}|{ds}|{mode}] weighted_ppl = {mean:.4f} ± {half:.4f} "
                        f"(95% CI, n={agg['n_valid']}/{agg['n_seeds']})"
                    )

    return {
        "aggregated": aggregated,
        "raw": raw_rows,
        "datasets": datasets,
        "modes": modes,
        "limit_val_batches": limit_val_batches,
    }


def _cart_rows_from_csv(csv_path: str) -> list[dict]:
    """Read a ``cart_raw.csv`` back into per-seed metric rows."""
    import csv

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for record in csv.DictReader(f):
            row: dict = {
                "kernel": record["kernel"],
                "mode": record["mode"],
                "dataset": record["dataset"],
                "seed": record.get("seed", ""),
            }
            for key in METRIC_KEYS:
                value = record.get(key, "")
                row[key] = float(value) if value not in ("", None) else None
            error = record.get("error", "")
            if error:
                row["error"] = error
            rows.append(row)
    return rows


def rebuild_cart_result_from_raw(
    raw_rows: list[dict],
    limit_val_batches: int | None = None,
) -> dict:
    """Re-aggregate a CART run from its raw per-seed rows (no GPU work).

    Lets ``cart_benchmark_results.csv`` and ``cart_benchmark_summary.md`` be
    regenerated from an existing ``cart_raw.csv`` without re-running the sweep.
    """
    if not raw_rows:
        raise ValueError("No raw CART rows to aggregate")

    datasets = sorted({row["dataset"] for row in raw_rows})
    modes = list(dict.fromkeys(row["mode"] for row in raw_rows))
    kernels = list(dict.fromkeys(row["kernel"] for row in raw_rows))

    aggregated: dict[str, dict[str, dict[str, dict]]] = {}
    for kernel in kernels:
        aggregated[kernel] = {}
        for mode in modes:
            aggregated[kernel][mode] = {}
            for ds_name in datasets:
                seed_metrics: list[dict] = []
                for row in raw_rows:
                    if (
                        row["kernel"] != kernel
                        or row["mode"] != mode
                        or row["dataset"] != ds_name
                    ):
                        continue
                    metrics = {
                        key: row.get(key) for key in METRIC_KEYS if row.get(key) is not None
                    }
                    if row.get("error"):
                        metrics["error"] = row["error"]
                    seed_metrics.append(metrics)
                aggregated[kernel][mode][ds_name] = _compute_statistics(seed_metrics)

    return {
        "aggregated": aggregated,
        "raw": raw_rows,
        "datasets": datasets,
        "modes": modes,
        "limit_val_batches": limit_val_batches,
    }


def save_cart_results(
    result: dict,
    seeds: int = 1,
    results_dir: str = "data/ablation_results",
    benchmarks_dir: str = BENCHMARKS_DIR,
) -> str:
    """Save the CART benchmark ablation as raw CSV, aggregated CSV and Markdown.

    Returns:
        str: Path to the Markdown summary file.
    """
    import csv
    from datetime import datetime

    os.makedirs(results_dir, exist_ok=True)

    aggregated = result["aggregated"]
    raw_rows = result["raw"]
    datasets = result["datasets"]
    modes = result["modes"]
    cap = result.get("limit_val_batches")

    # ---- Raw per-seed CSV (one row per kernel × mode × dataset × seed) ----
    raw_csv_path = os.path.join(results_dir, "cart_raw.csv")
    raw_fields = ["kernel", "mode", "dataset", "seed"] + METRIC_KEYS + ["error"]
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)
    logger.info(f"Raw per-seed CSV saved to {raw_csv_path}")

    # ---- Aggregated CSV (mean ± CI per kernel × mode × dataset) ----
    agg_rows: list[dict] = []
    for kernel, mode_dict in aggregated.items():
        for mode, dataset_dict in mode_dict.items():
            for ds_name, agg in dataset_dict.items():
                row = {
                    "kernel": kernel,
                    "mode": mode,
                    "dataset": ds_name,
                    "n_seeds": agg.get("n_valid", agg.get("n_seeds", 0)),
                    "error": agg.get("error", ""),
                }
                for key in METRIC_KEYS:
                    row[key] = agg.get(key)
                    row[f"{key}_std"] = agg.get(f"{key}_std")
                    row[f"{key}_ci95_low"] = agg.get(f"{key}_ci95_low")
                    row[f"{key}_ci95_high"] = agg.get(f"{key}_ci95_high")
                agg_rows.append(row)

    agg_csv_path = os.path.join(results_dir, "cart_benchmark_results.csv")
    agg_fields = ["kernel", "mode", "dataset", "n_seeds"]
    for key in METRIC_KEYS:
        agg_fields += [key, f"{key}_std", f"{key}_ci95_low", f"{key}_ci95_high"]
    agg_fields.append("error")
    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)
    logger.info(f"Aggregated CSV saved to {agg_csv_path}")

    # ---- Markdown summary ----
    md_path = os.path.join(results_dir, "cart_benchmark_summary.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    multi_mode = len(modes) > 1
    multi_seed = seeds > 1

    def _agg(ds_name, mode, kernel, key):
        agg = aggregated.get(kernel, {}).get(mode, {}).get(ds_name, {})
        if not multi_seed:
            return agg.get(key)
        return agg.get(f"{key}_mean", agg.get(key))

    def _half_ci(ds_name, mode, kernel, key):
        agg = aggregated.get(kernel, {}).get(mode, {}).get(ds_name, {})
        mean = agg.get(f"{key}_mean")
        low = agg.get(f"{key}_ci95_low")
        if mean is None or low is None:
            return None
        return mean - low

    with open(md_path, "w") as f:
        f.write("# MaRK CART-Weighted Validation Ablation — All Benchmark Datasets\n\n")
        f.write(f"> Generated: {timestamp}\n")
        f.write(f"> Seeds: {seeds} (seed i = {BASE_SEED} + i×{SEED_STRIDE})\n")
        f.write(
            "> Estimator: sample std with `ddof=1`; 95% CI = mean ± 1.96 × SEM "
            "(same estimator as the WikiText leave-one-out SSM ablation)\n"
        )
        f.write(f"> Benchmark root: `{benchmarks_dir}`\n")
        f.write(f"> Datasets ({len(datasets)}): {', '.join(datasets)}\n")
        f.write(f"> Modes: {', '.join(modes)}\n")
        f.write("> Evaluator: NeMo `Trainer.validate` with diffusion masking and CART weights\n")
        f.write(
            "> **Primary metric:** `weighted_nll` = the **CART loss**. "
            "`weighted_ppl = exp(weighted_nll)` is the same number exponentiated.\n"
        )
        if cap is not None:
            f.write(
                f"> **Validation cap:** first {int(cap)} validation batches per dataset "
                "(datasets with fewer batches are evaluated in full)\n"
            )
        else:
            f.write("> **Validation cap:** none — every dataset is evaluated in full\n")
        f.write("\n")

        f.write("## Results: CART loss (weighted NLL) per dataset\n\n")
        f.write(
            "`weighted_nll` is the CART loss — the reportable number. "
            "`weighted_ppl = exp(weighted_nll)` is shown only as a convenience.\n\n"
        )
        header = "| Kernel | Dataset |"
        if multi_mode:
            header += " Mode |"
        header += " n | CART loss (W-NLL) ↓ | W-PPL (=e^NLL) | Raw PPL ↓ |\n"
        f.write(header)
        sep = "|--------|---------|"
        if multi_mode:
            sep += "------|"
        sep += "---|---|--------------------|---------------|-----------|\n"
        f.write(sep)

        for kernel in sorted(aggregated.keys()):
            for mode in modes:
                for ds_name in datasets:
                    n = aggregated[kernel][mode][ds_name].get("n_valid", 0)
                    wnl = _fmt(_agg(ds_name, mode, kernel, "weighted_nll"))
                    nhalf = _half_ci(ds_name, mode, kernel, "weighted_nll")
                    wppl = _fmt(_agg(ds_name, mode, kernel, "weighted_ppl"))
                    half = _half_ci(ds_name, mode, kernel, "weighted_ppl")
                    rppl = _fmt(_agg(ds_name, mode, kernel, "raw_ppl"))
                    rhalf = _half_ci(ds_name, mode, kernel, "raw_ppl")
                    if multi_seed:
                        wnl_s = f"{wnl} ± {_fmt(nhalf, '.3f')}" if nhalf else wnl
                        wppl_s = f"{wppl} ± {_fmt(half, '.3f')}" if half else wppl
                        rppl_s = f"{rppl} ± {_fmt(rhalf, '.3f')}" if rhalf else rppl
                    else:
                        wnl_s, wppl_s, rppl_s = wnl, wppl, rppl

                    row = f"| {kernel:>9s} | {ds_name:<9s} |"
                    if multi_mode:
                        row += f" {mode:<8s} |"
                    row += f" {n} | {wnl_s:>18s} | {wppl_s:>13s} | {rppl_s:>9s} |\n"
                    f.write(row)
        f.write("\n")

        # ---- Per-dataset ranking (best kernel by CART loss) ----
        f.write("## Best kernel per dataset (lowest CART loss)\n\n")
        f.write("| Dataset | Best kernel | CART loss | 2nd | 3rd |\n")
        f.write("|---------|-------------|-----------|-----|-----|\n")
        primary_mode = modes[0]
        for ds_name in datasets:
            ranked = sorted(
                (
                    (kernel, _agg(ds_name, primary_mode, kernel, "weighted_nll"))
                    for kernel in aggregated.keys()
                ),
                key=lambda kv: (kv[1] is None, kv[1]),
            )
            def _cell(idx):
                if idx >= len(ranked) or ranked[idx][1] is None:
                    return "—"
                return f"{ranked[idx][0]} ({_fmt(ranked[idx][1])})"
            f.write(f"| {ds_name:<9s} | {_cell(0).split(' (')[0]:<11s} | "
                    f"{_fmt(ranked[0][1]) if ranked else '—':>9s} | {_cell(1)} | {_cell(2)} |\n")
        f.write("\n")

        # ---- Kernel average across datasets ----
        f.write("## Kernel average across datasets (CART loss)\n\n")
        f.write("| Kernel | Mean CART loss | Min | Max | Datasets |\n")
        f.write("|--------|---------------|-----|-----|----------|\n")
        for kernel in sorted(aggregated.keys()):
            vals = [
                _agg(ds_name, primary_mode, kernel, "weighted_nll")
                for ds_name in datasets
            ]
            vals = [v for v in vals if v is not None]
            if not vals:
                f.write(f"| {kernel:>9s} | — | — | — | {len(datasets)} |\n")
                continue
            f.write(
                f"| {kernel:>9s} | {_fmt(float(np.mean(vals))):>14s} | "
                f"{_fmt(min(vals)):>5s} | {_fmt(max(vals)):>5s} | {len(vals)} |\n"
            )
        f.write("\n")

        if multi_mode:
            f.write("## Mode averages across datasets (CART loss)\n\n")
            f.write("| Mode | Mean CART loss | Datasets |\n")
            f.write("|------|---------------|----------|\n")
            for mode in modes:
                vals = [
                    _agg(ds_name, mode, kernel, "weighted_nll")
                    for kernel in aggregated.keys()
                    for ds_name in datasets
                ]
                vals = [v for v in vals if v is not None]
                if vals:
                    f.write(f"| {mode:<12s} | {_fmt(float(np.mean(vals))):>14s} | {len(vals)} |\n")
            f.write("\n")

        f.write("## Artifacts\n\n")
        f.write("- `cart_raw.csv` — one row per kernel × dataset × mode × seed; `weighted_nll` is the CART loss\n")
        f.write("- `cart_benchmark_results.csv` — mean, std and 95% CI per kernel × dataset × mode\n")
        f.write("- `cart_<kernel>_stage1_<dataset>_<mode>_seed<seed>.json` — per-run evaluator output\n")

    logger.info(f"Markdown summary saved to {md_path}")
    return md_path


# =============================================================================
# Suite 3 — AdaLN-Zero / Input-Injection profiling (companion repo delegation)
# =============================================================================

def run_profiling_suite(
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    results_dir: str = "data/ablation_results",
    seeds: int = 15,
    mode: str = "both",
    batch_sizes=(1, 4, 8),
    seq_len: int = 512,
    latency_n: int = 100,
    warmup: int = 20,
    tag: str | None = None,
    python_exe: str | None = None,
    extra_args=(),
    summarize_only: bool = False,
) -> dict:
    """Run the AdaLN-Zero / Input-Injection / MaRK profiling ablation.

    The profiling implementation lives in the companion ``experiment``
    repository (``profiling.profile_multiseed``), which is the canonical
    source for these measurements. This wrapper runs that suite with a fixed
    multi-seed protocol and copies the reportable artifacts (raw CSV +
    Markdown tables) into this repository's results directory.

    Returns:
        dict with the run's output directory, copied artifacts and summary path.
    """
    experiment_path = Path(os.path.expanduser(experiment_dir)).resolve()
    runner = experiment_path / "profiling" / "profile_multiseed.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Profiling suite not found at {runner}. "
            f"Set --experiment-dir to the checkout of the companion 'experiment' repository."
        )

    run_tag = tag or f"seq{seq_len}_{seeds}seeds"
    run_out = experiment_path / "profiling" / "results" / run_tag
    dest_dir = Path(results_dir) / "profiling" / run_tag
    dest_dir.mkdir(parents=True, exist_ok=True)

    exe = python_exe or sys.executable
    cmd = [
        exe, "-m", "profiling.profile_multiseed",
        "--mode", mode,
        "--seeds", str(seeds),
        "--batch-sizes", *[str(b) for b in batch_sizes],
        "--seq-len", str(seq_len),
        "--latency-n", str(latency_n),
        "--warmup", str(warmup),
        "--frozen",
        "--precompile",
        "--output", str(run_out),
        *extra_args,
    ]

    if summarize_only and run_out.exists():
        logger.info(f"Skipping profiling run; reusing existing artifacts in {run_out}")
    else:
        logger.info(f"Running profiling suite in {experiment_path}:")
        logger.info("  " + " ".join(cmd))
        completed = subprocess.run(cmd, cwd=str(experiment_path), check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Profiling suite exited with code {completed.returncode}. "
                f"Check that the companion repo's environment is active "
                f"(or pass --profiling-python)."
            )

    copied: list[str] = []
    for pattern in ("*.md", "*.csv", "*.txt"):
        for src in sorted(run_out.glob(pattern)):
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    summary_path = Path(results_dir) / "profiling_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("# MaRK Profiling Ablation — AdaLN-Zero vs Input-Injection vs MaRK Kernels\n\n")
        f.write(f"> Source: `{experiment_path}` (`profiling.profile_multiseed`)\n")
        f.write(f"> Seeds: {seeds} | Latency iterations/seed: {latency_n} | Warmup: {warmup}\n")
        f.write(f"> Mode: {mode} | Batch sizes: {list(batch_sizes)} | Seq len: {seq_len}\n")
        f.write(f"> Artifacts: `{dest_dir}`\n\n")
        for name in ("adapter_table.md", "e2e_table_bs1.md", "e2e_table_bs4.md", "e2e_table_bs8.md"):
            candidate = dest_dir / name
            if candidate.exists():
                f.write(f"## {name}\n\n")
                f.write(candidate.read_text())
                f.write("\n")
    logger.info(f"Profiling summary saved to {summary_path}")

    return {
        "experiment_dir": str(experiment_path),
        "output_dir": str(run_out),
        "artifacts_dir": str(dest_dir),
        "copied": copied,
        "summary_path": str(summary_path),
    }


# =============================================================================
# Reporting helpers
# =============================================================================

def _fmt(val, fmt_str: str = ".4f") -> str:
    """Format a float or return '—' if None."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    try:
        return f"{float(val):{fmt_str}}"
    except (TypeError, ValueError):
        return str(val)


def _checkmark(b: bool) -> str:
    return "✓" if b else "✗"


def save_results(
    all_results: dict,
    seeds: int = 1,
    results_dir: str = "data/ablation_results",
) -> str:
    """Save leave-one-out ablation results as CSV and Markdown table.

    When seeds > 1, reports mean ± std with 95% CI.

    Returns:
        str: Path to the Markdown summary file.
    """
    import csv
    from datetime import datetime

    os.makedirs(results_dir, exist_ok=True)

    # ---- Flatten into rows ----
    rows: list[dict] = []
    for kernel, mode_dict in all_results.items():
        for mode, agg in mode_dict.items():
            pm = MODE_PARAMS.get(mode, {})
            row = {
                "kernel": kernel,
                "mode": mode,
                "A": _checkmark(pm.get("A", True)),
                "B": _checkmark(pm.get("B", True)),
                "C": _checkmark(pm.get("C", True)),
                "dt": _checkmark(pm.get("dt", True)),
                "D": _checkmark(pm.get("D", True)),
                "n_seeds": agg.get("n_valid", agg.get("n_seeds", 1)),
                "error": agg.get("error", ""),
            }
            for key in METRIC_KEYS:
                row[key] = agg.get(key)  # mean
                row[f"{key}_std"] = agg.get(f"{key}_std")
                row[f"{key}_ci95_low"] = agg.get(f"{key}_ci95_low")
                row[f"{key}_ci95_high"] = agg.get(f"{key}_ci95_high")
            rows.append(row)

    # ---- CSV ----
    csv_path = os.path.join(results_dir, "ablation_results.csv")
    fieldnames = [
        "kernel", "mode", "A", "B", "C", "dt", "D", "n_seeds",
    ]
    for key in METRIC_KEYS:
        fieldnames += [key, f"{key}_std", f"{key}_ci95_low", f"{key}_ci95_high"]
    fieldnames.append("error")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"CSV saved to {csv_path}")

    # ---- Markdown Table ----
    md_path = os.path.join(results_dir, "ablation_summary.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w") as f:
        f.write("# MaRK SSM Parameter Ablation — Validation Loss on WikiText\n\n")
        f.write(f"> Generated: {timestamp}\n")
        f.write(f"> Seeds: {seeds}  |  Seed offset: {BASE_SEED} + i×{SEED_STRIDE}\n\n")

        # ---- Statistical note if multi-seed ----
        if seeds > 1:
            f.write(
                "**Note:** Results are reported as mean ± 95% CI across "
                f"{seeds} seeds. Seed 1 uses `torch.manual_seed({BASE_SEED})`, "
                f"subsequent seeds use {BASE_SEED} + i×{SEED_STRIDE}.\n\n"
            )

        # ---- Main results table ----
        f.write("## Results: All Kernels × All Ablation Modes\n\n")

        if seeds > 1:
            header = (
                "| Kernel | Mode | A | B | C | dt | D | n | "
                "Weighted PPL ↓ | Raw PPL ↓ | Weighted NLL ↓ |\n"
            )
            sep = (
                "|--------|------|---|---|---|----|---|---|"
                "---------------|-----------|----------------|\n"
            )
        else:
            header = (
                "| Kernel | Mode | A | B | C | dt | D | "
                "Weighted PPL ↓ | Raw PPL ↓ | Weighted NLL ↓ |\n"
            )
            sep = (
                "|--------|------|---|---|---|----|---|"
                "---------------|-----------|----------------|\n"
            )
        f.write(header)
        f.write(sep)

        for row in rows:
            if seeds > 1:
                wppl_mean = row.get("weighted_ppl")
                wppl_std = row.get("weighted_ppl_std")
                rppl_mean = row.get("raw_ppl")
                rppl_std = row.get("raw_ppl_std")
                wnl_mean = row.get("weighted_nll")
                wnl_std = row.get("weighted_nll_std")

                wppl_str = f"{_fmt(wppl_mean)} ± {_fmt(wppl_std, '.3f')}" if wppl_std else _fmt(wppl_mean)
                rppl_str = f"{_fmt(rppl_mean)} ± {_fmt(rppl_std, '.3f')}" if rppl_std else _fmt(rppl_mean)
                wnl_str = f"{_fmt(wnl_mean)} ± {_fmt(wnl_std, '.3f')}" if wnl_std else _fmt(wnl_mean)

                f.write(
                    f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                    f"| {row['A']} | {row['B']} | {row['C']} | {row['dt']} | {row['D']} "
                    f"| {row['n_seeds']} "
                    f"| {wppl_str:>13s} | {rppl_str:>9s} | {wnl_str:>14s} |\n"
                )
            else:
                f.write(
                    f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                    f"| {row['A']} | {row['B']} | {row['C']} | {row['dt']} | {row['D']} "
                    f"| {_fmt(row['weighted_ppl']):>13s} | {_fmt(row['raw_ppl']):>9s} "
                    f"| {_fmt(row['weighted_nll']):>14s} |\n"
                )
        f.write("\n")

        # ---- Detailed stats table (only if multi-seed) ----
        if seeds > 1:
            f.write("## Detailed Statistics (per kernel/mode)\n\n")
            f.write(
                "| Kernel | Mode | Metric | Mean | ±95% CI | Std |\n"
            )
            f.write(
                "|--------|------|--------|------|---------|-----|\n"
            )
            for row in sorted(rows, key=lambda r: (r["kernel"], r["mode"])):
                for key in METRIC_KEYS:
                    mean = row.get(key)
                    ci_low = row.get(f"{key}_ci95_low")
                    ci_high = row.get(f"{key}_ci95_high")
                    std = row.get(f"{key}_std")
                    if mean is not None and ci_low is not None:
                        half_ci = mean - ci_low
                        f.write(
                            f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                            f"| {key:<15s} | {_fmt(mean):>6s} "
                            f"| ±{half_ci:.4f} | {_fmt(std, '.4f'):>5s} |\n"
                        )
            f.write("\n")

        # ---- Analysis: Reviewer Q2 ----
        f.write("## Analysis: A-Modulation Contribution (Reviewer Axfu Q2)\n\n")
        f.write(
            "Comparing `full` (all 5 params modulated) vs `all_except_A` "
            "(A frozen, B/C/D/Δ modulated). The gap isolates how much the "
            "recurrence parameter A contributes beyond Mamba-style selection "
            "(which already modulates Δ, B, and C).\n\n"
        )
        f.write("| Kernel | Full W-PPL | All-except-A W-PPL | Δ W-PPL | A Contribution |\n")
        f.write("|--------|-----------|-------------------|---------|---------------|\n")
        for kernel in sorted(all_results.keys()):
            full_m = all_results[kernel].get("full", {})
            noA_m = all_results[kernel].get("all_except_A", {})
            if seeds > 1:
                full_val = full_m.get("weighted_ppl_mean")
                noA_val = noA_m.get("weighted_ppl_mean")
                full_std = full_m.get("weighted_ppl_std")
                noA_std = noA_m.get("weighted_ppl_std")
            else:
                full_val = full_m.get("weighted_ppl")
                noA_val = noA_m.get("weighted_ppl")
                full_std = None
                noA_std = None

            if full_val is not None and noA_val is not None:
                delta = float(noA_val) - float(full_val)
                full_str = f"{_fmt(full_val)} ± {_fmt(full_std, '.3f')}" if full_std else _fmt(full_val)
                noA_str = f"{_fmt(noA_val)} ± {_fmt(noA_std, '.3f')}" if noA_std else _fmt(noA_val)
                f.write(
                    f"| {kernel:>9s} | {full_str:>9s} | {noA_str:>17s} "
                    f"| {_fmt(delta):>7s} | {delta:>+.4f} |\n"
                )
            else:
                f.write(f"| {kernel:>9s} | — | — | — | — |\n")
        f.write("\n")

        # ---- Analysis: BC_only vs full ----
        f.write("## Analysis: Mamba-Style Selection\n\n")
        f.write(
            "If `BC_only` performs close to `full`, Mamba-style selection through "
            "B and C already captures much of MaRK's benefit. If `A_only` is close "
            "to `full`, then A-modulation is the key contribution.\n\n"
        )
        f.write("| Kernel | Full W-PPL | BC_only W-PPL | A_only W-PPL | dt_only W-PPL |\n")
        f.write("|--------|-----------|--------------|-------------|--------------|\n")
        for kernel in sorted(all_results.keys()):
            full_m = all_results[kernel].get("full", {})
            bc_m = all_results[kernel].get("BC_only", {})
            a_m = all_results[kernel].get("A_only", {})
            dt_m = all_results[kernel].get("dt_only", {})

            def _get(m, key):
                if seeds > 1:
                    return m.get(f"{key}_mean") if m else None
                return m.get(key) if m else None

            f.write(
                f"| {kernel:>9s} | {_fmt(_get(full_m, 'weighted_ppl')):>9s} "
                f"| {_fmt(_get(bc_m, 'weighted_ppl')):>12s} "
                f"| {_fmt(_get(a_m, 'weighted_ppl')):>11s} "
                f"| {_fmt(_get(dt_m, 'weighted_ppl')):>12s} |\n"
            )
        f.write("\n")

        # ---- Full table with all metrics ----
        f.write("## Full Results (All Metrics)\n\n")
        full_header = (
            "| Kernel | Mode | Raw NLL | Raw PPL | "
            "Weighted NLL | Weighted PPL |\n"
        )
        full_sep = (
            "|--------|------|---------|---------|"
            "-------------|-------------|\n"
        )
        f.write(full_header)
        f.write(full_sep)
        for row in rows:
            if seeds > 1:
                rn_str = f"{_fmt(row['raw_nll'])} ± {_fmt(row.get('raw_nll_std'), '.3f')}" if row.get('raw_nll_std') else _fmt(row['raw_nll'])
                rp_str = f"{_fmt(row['raw_ppl'])} ± {_fmt(row.get('raw_ppl_std'), '.3f')}" if row.get('raw_ppl_std') else _fmt(row['raw_ppl'])
                wn_str = f"{_fmt(row['weighted_nll'])} ± {_fmt(row.get('weighted_nll_std'), '.3f')}" if row.get('weighted_nll_std') else _fmt(row['weighted_nll'])
                wp_str = f"{_fmt(row['weighted_ppl'])} ± {_fmt(row.get('weighted_ppl_std'), '.3f')}" if row.get('weighted_ppl_std') else _fmt(row['weighted_ppl'])
                f.write(
                    f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                    f"| {rn_str:>7s} | {rp_str:>7s} "
                    f"| {wn_str:>11s} | {wp_str:>11s} |\n"
                )
            else:
                f.write(
                    f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                    f"| {_fmt(row['raw_nll']):>7s} | {_fmt(row['raw_ppl']):>7s} "
                    f"| {_fmt(row['weighted_nll']):>11s} | {_fmt(row['weighted_ppl']):>11s} |\n"
                )
        f.write("\n")

    logger.info(f"Markdown summary saved to {md_path}")
    return md_path


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MaRK ablation driver: leave-one-out SSM ablation (loo), "
                    "CART benchmark ablation (cart), AdaLN/Input-Injection profiling "
                    "(profiling), or all suites."
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default="loo",
        help="Which ablation suite to run (default: loo).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        metavar="KERNEL=DIR",
        help="Checkpoint directory override per kernel. Repeat for multiple dirs/kernels.",
    )
    parser.add_argument(
        "--wikitext-dir",
        default="data/wikitext",
        help="Path to WikiText packed parquet data (default: data/wikitext)",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default=BENCHMARKS_DIR,
        help=f"Benchmark dataset root for the cart suite (default: {BENCHMARKS_DIR})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset subdirectories to evaluate in the cart suite "
             "(default: auto-discover every dataset under --benchmarks-dir)",
    )
    parser.add_argument(
        "--cart-report-only",
        action="store_true",
        help="Skip the cart GPU sweep and rebuild cart_benchmark_results.csv and "
             "cart_benchmark_summary.md from the existing cart_raw.csv in --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ablation_results",
        help="Directory for outputs (default: data/ablation_results)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of validation batches per dataset at N (an absolute batch "
             "count, not a fraction). Used for smoke tests and as the evaluation cap for "
             "the large cart datasets such as arxiv/pubmed; datasets with fewer than N "
             "batches are still evaluated in full.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help=f"Ablation modes to run. loo default: 'all_except_A' only. "
             f"cart default: 'full' (the model as trained). "
             f"All supported: {{{', '.join(ALL_MODES)}}}",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=None,
        help="Kernels to evaluate (default: all 3). Use 'hypernet', 'chebyshev', 'dct'",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        metavar="N",
        help="Number of seeds for the loo/cart suites (default: 1). "
             f"Seeds are {BASE_SEED}, {BASE_SEED + SEED_STRIDE}, {BASE_SEED + 2 * SEED_STRIDE}, ...",
    )
    parser.add_argument(
        "--no-dataloader-cache",
        dest="cache_dataloaders",
        action="store_false",
        default=True,
        help="Rebuild the validation dataloader for every seed instead of caching it "
             "per dataset (cart suite).",
    )
    # ---- profiling suite ----
    parser.add_argument(
        "--experiment-dir",
        default=DEFAULT_EXPERIMENT_DIR,
        help=f"Checkout of the companion 'experiment' repo holding the AdaLN/Input-Injection "
             f"profiling suite (default: {DEFAULT_EXPERIMENT_DIR})",
    )
    parser.add_argument(
        "--profiling-seeds",
        type=int,
        default=15,
        help="Seeds for the profiling suite (default: 15, matching the published A100 runs)",
    )
    parser.add_argument(
        "--profiling-mode",
        choices=["adapter", "e2e", "both"],
        default="both",
        help="Profiling scope (default: both)",
    )
    parser.add_argument(
        "--profiling-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes for the end-to-end profiling sweep (default: 1 4 8)",
    )
    parser.add_argument(
        "--profiling-seq-len",
        type=int,
        default=512,
        help="Sequence length for the end-to-end profiling sweep (default: 512)",
    )
    parser.add_argument(
        "--profiling-latency-n",
        type=int,
        default=100,
        help="Latency iterations per seed (default: 100)",
    )
    parser.add_argument(
        "--profiling-warmup",
        type=int,
        default=20,
        help="Warmup iterations (default: 20)",
    )
    parser.add_argument(
        "--profiling-tag",
        default=None,
        help="Output tag for the profiling run (default: seq<len>_<seeds>seeds)",
    )
    parser.add_argument(
        "--profiling-python",
        default=None,
        help="Python executable used to launch the companion repo's profiling suite "
             "(default: the interpreter running this script)",
    )
    parser.add_argument(
        "--profiling-summarize-only",
        action="store_true",
        default=False,
        help="Do not run the profiling suite; only re-copy/summarize existing artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from .perplexity import parse_checkpoint_dir_overrides

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    if args.limit_val_batches is not None and args.limit_val_batches < 1:
        parser.error("--limit-val-batches must be >= 1 (absolute batch count)")

    try:
        checkpoint_dir_overrides = parse_checkpoint_dir_overrides(args.checkpoint_dir)
    except ValueError as exc:
        parser.error(str(exc))

    kernels = args.kernels if args.kernels else None

    # Filter MODEL_REGISTRY by requested kernels
    models = MODEL_REGISTRY
    if kernels:
        models = [m for m in models if m.kernel in kernels]
        if not models:
            parser.error(f"No models match kernels: {kernels}")

    logger.info(f"Suite: {args.suite}")
    logger.info(f"Kernels: {[m.kernel for m in models]}")
    logger.info(f"Seeds: {args.seeds}")

    outputs: dict[str, str] = {}

    if args.suite in ("loo", "all"):
        loo_modes = args.modes if args.modes else ABLATION_MODES
        logger.info(f"[loo] Modes: {loo_modes}")
        logger.info(f"[loo] Total evaluations: {len(models) * len(loo_modes) * args.seeds}")
        loo_results = run_ablation_suite(
            models=models,
            modes=loo_modes,
            seeds=args.seeds,
            wikitext_dir=args.wikitext_dir,
            results_dir=args.output_dir,
            limit_val_batches=args.limit_val_batches,
            checkpoint_dir_overrides=checkpoint_dir_overrides,
        )
        outputs["loo_markdown"] = save_results(loo_results, seeds=args.seeds, results_dir=args.output_dir)

    if args.suite in ("cart", "all"):
        cart_modes = args.modes if args.modes else CART_MODES
        logger.info(f"[cart] Modes: {cart_modes}")
        cart_seeds = args.seeds
        if args.cart_report_only:
            raw_csv = os.path.join(args.output_dir, "cart_raw.csv")
            if not os.path.exists(raw_csv):
                parser.error(f"--cart-report-only: no raw results found at {raw_csv}")
            logger.info(f"[cart] Report-only: re-aggregating {raw_csv}")
            raw_rows = _cart_rows_from_csv(raw_csv)
            cart_seeds = len({row.get("seed") for row in raw_rows})
            logger.info(f"[cart] Loaded {len(raw_rows)} per-seed rows ({cart_seeds} seeds)")
            cart_result = rebuild_cart_result_from_raw(raw_rows, args.limit_val_batches)
        else:
            cart_result = run_cart_benchmark_suite(
                models=models,
                modes=cart_modes,
                seeds=args.seeds,
                benchmarks_dir=args.benchmarks_dir,
                datasets=args.datasets,
                results_dir=args.output_dir,
                limit_val_batches=args.limit_val_batches,
                checkpoint_dir_overrides=checkpoint_dir_overrides,
                cache_dataloaders=args.cache_dataloaders,
            )
        outputs["cart_markdown"] = save_cart_results(
            cart_result,
            seeds=cart_seeds,
            results_dir=args.output_dir,
            benchmarks_dir=args.benchmarks_dir,
        )

    if args.suite in ("profiling", "all"):
        logger.info(f"[profiling] Seeds: {args.profiling_seeds}")
        profiling_result = run_profiling_suite(
            experiment_dir=args.experiment_dir,
            results_dir=args.output_dir,
            seeds=args.profiling_seeds,
            mode=args.profiling_mode,
            batch_sizes=args.profiling_batch_sizes,
            seq_len=args.profiling_seq_len,
            latency_n=args.profiling_latency_n,
            warmup=args.profiling_warmup,
            tag=args.profiling_tag,
            python_exe=args.profiling_python,
            summarize_only=args.profiling_summarize_only,
        )
        outputs["profiling_summary"] = profiling_result["summary_path"]

    # Print summary to console
    print("\n" + "=" * 80)
    print(f"ABLATION COMPLETE — suite={args.suite}")
    print("=" * 80)
    for label, path in outputs.items():
        print(f"{label:<20s}: {path}")
    if "loo_markdown" in outputs:
        print(f"{'loo CSV':<20s}: {os.path.join(args.output_dir, 'ablation_results.csv')}")
    if "cart_markdown" in outputs:
        print(f"{'cart raw CSV':<20s}: {os.path.join(args.output_dir, 'cart_raw.csv')}")
        print(f"{'cart agg CSV':<20s}: {os.path.join(args.output_dir, 'cart_benchmark_results.csv')}")


if __name__ == "__main__":
    main()
