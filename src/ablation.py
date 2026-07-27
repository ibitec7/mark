# ================================================
# MaRK SSM Parameter Ablation — Validation Loss
#
# Isolates which SSM parameters (A, B, C, dt/Δ, D)
# contribute to validation loss by selectively
# freezing subsets after MaRK modulation.
#
# Run: cd /home/admin/Desktop/mark && python -m src.ablation
# ================================================

import argparse
import gc
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import numpy as np

from .perplexity import (
    MODEL_REGISTRY,
    _load_model_and_trainer,
    apply_checkpoint_dir_overrides,
    discover_datasets,
    resolve_runtime_path,
    SUPPORTED_KERNELS,
)
from .utils import log_setup

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

# Metrics we collect per evaluation
METRIC_KEYS = ["raw_nll", "raw_ppl", "weighted_nll", "weighted_ppl"]

# z-score for 95% CI (two-tailed)
Z_95 = 1.96


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
                import shutil
                shutil.copy2(pf, dst)
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
    """Run a single ablation evaluation: load model, inject mode, validate."""
    from .perplexity import _evaluate_single

    # Seed everything for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def run_ablation_suite(
    models=None,
    modes=None,
    seeds: int = 1,
    wikitext_dir: str = "data/wikitext",
    results_dir: str = "data/ablation_results",
    limit_val_batches: int | float | None = None,
    checkpoint_dir_overrides: dict[str, list[str]] | None = None,
) -> dict:
    """Run full ablation suite: all kernels × all modes × N seeds on WikiText.

    Returns:
        dict: {kernel: {mode: aggregated_metrics_dict}}
    """
    if models is None:
        models = MODEL_REGISTRY
    if modes is None:
        modes = ABLATION_MODES

    models = apply_checkpoint_dir_overrides(models, checkpoint_dir_overrides)

    os.makedirs(results_dir, exist_ok=True)

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
                # Deterministic seed offset: base seed 42 + kernel index + seed index
                # to keep seeds reproducible across runs
                actual_seed = 42 + seed_idx * 100
                result = _run_ablation_evaluation(
                    entry=entry,
                    mode=mode,
                    wikitext_benchmark_dir=wikitext_benchmark_dir,
                    results_dir=results_dir,
                    limit_val_batches=limit_val_batches,
                    seed=actual_seed,
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
    """Save ablation results as CSV and Markdown table.

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
        f.write(f"> Seeds: {seeds}  |  Seed offset: 42 + i×100\n\n")

        # ---- Statistical note if multi-seed ----
        if seeds > 1:
            f.write(
                "**Note:** Results are reported as mean ± 95% CI across "
                f"{seeds} seeds. Seed 1 uses `torch.manual_seed(42)`, "
                "subsequent seeds use 42 + i×100.\n\n"
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
        if seeds > 1:
            f.write("| Kernel | Full W-PPL | All-except-A W-PPL | Δ W-PPL | A Contribution |\n")
            f.write("|--------|-----------|-------------------|---------|---------------|\n")
        else:
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
        if seeds > 1:
            full_header = (
                "| Kernel | Mode | Raw NLL | Raw PPL | "
                "Weighted NLL | Weighted PPL |\n"
            )
            full_sep = (
                "|--------|------|---------|---------|"
                "-------------|-------------|\n"
            )
        else:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MaRK SSM parameter ablation: validation loss on WikiText"
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
        "--output-dir",
        default="data/ablation_results",
        help="Directory for outputs (default: data/ablation_results)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Limit validation batches for smoke tests",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help=f"Ablation modes to run (default: 'all_except_A' only — the reviewer-relevant one). "
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
        help="Number of seeds to run per (kernel, mode) pair. Reports mean ± 95%% CI. "
             "Seeds are 42, 142, 242, ... (default: 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from .perplexity import parse_checkpoint_dir_overrides

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    try:
        checkpoint_dir_overrides = parse_checkpoint_dir_overrides(args.checkpoint_dir)
    except ValueError as exc:
        parser.error(str(exc))

    modes = args.modes if args.modes else ABLATION_MODES
    kernels = args.kernels if args.kernels else None

    # Filter MODEL_REGISTRY by requested kernels
    models = MODEL_REGISTRY
    if kernels:
        models = [m for m in models if m.kernel in kernels]
        if not models:
            parser.error(f"No models match kernels: {kernels}")

    logger.info(f"Kernels: {[m.kernel for m in models]}")
    logger.info(f"Modes: {modes}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Total evaluations: {len(models) * len(modes) * args.seeds}")

    all_results = run_ablation_suite(
        models=models,
        modes=modes,
        seeds=args.seeds,
        wikitext_dir=args.wikitext_dir,
        results_dir=args.output_dir,
        limit_val_batches=args.limit_val_batches,
        checkpoint_dir_overrides=checkpoint_dir_overrides,
    )

    md_path = save_results(all_results, seeds=args.seeds, results_dir=args.output_dir)

    # Print summary to console
    print("\n" + "=" * 80)
    print("ABLATION COMPLETE")
    print("=" * 80)
    print(f"Markdown summary: {md_path}")
    print(f"CSV results:      {os.path.join(args.output_dir, 'ablation_results.csv')}")

    # Print mini-table
    kernels_sorted = sorted(all_results.keys())
    if kernels_sorted:
        n_seeds = args.seeds
        print(f"\n{'Kernel':>10s}", end="")
        for mode in modes:
            if n_seeds > 1:
                print(f" | {mode + ' (±95% CI)':<26s}", end="")
            else:
                print(f" | {mode:<14s}", end="")
        print()
        print("-" * (10 + 28 * len(modes)))
        for kernel in kernels_sorted:
            print(f"{kernel:>10s}", end="")
            for mode in modes:
                m = all_results[kernel].get(mode, {})
                if n_seeds > 1:
                    mean = m.get("weighted_ppl_mean")
                    ci_lo = m.get("weighted_ppl_ci95_low")
                    ci_hi = m.get("weighted_ppl_ci95_high")
                    if mean is not None and ci_lo is not None:
                        half = mean - ci_lo
                        print(f" | {_fmt(mean)} ± {_fmt(half, '.3f'):>18s}", end="")
                    else:
                        print(f" | {_fmt(m.get('weighted_ppl')):>26s}", end="")
                else:
                    print(f" | {_fmt(m.get('weighted_ppl')):>14s}", end="")
            print()


if __name__ == "__main__":
    main()
