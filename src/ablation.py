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
import os
import sys
from pathlib import Path

import torch

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
) -> dict | None:
    """Run a single ablation evaluation: load model, inject mode, validate."""
    from .perplexity import _evaluate_single

    logger.info("")
    logger.info(f"{'='*70}")
    logger.info(f"  [{entry.name}] mode={mode}")
    logger.info(f"{'='*70}")

    # Load model fresh (ensures clean state)
    model, trainer, train_config, ckpt_path = _load_model_and_trainer(entry)

    if limit_val_batches is not None:
        trainer.limit_val_batches = limit_val_batches

    # Inject ablation mode
    _inject_ablation_mode(model, mode)

    # Determine output path
    output_path = os.path.join(results_dir, f"{entry.name}_{mode}.json")

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
            logger.info(
                f"  ✓ raw_ppl={metrics.get('raw_ppl', 'N/A'):.4f}, "
                f"weighted_ppl={metrics.get('weighted_ppl', 'N/A'):.4f}"
            )
            return metrics
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


def run_ablation_suite(
    models=None,
    modes=None,
    wikitext_dir: str = "data/wikitext",
    results_dir: str = "data/ablation_results",
    limit_val_batches: int | float | None = None,
    checkpoint_dir_overrides: dict[str, list[str]] | None = None,
) -> dict:
    """Run full ablation suite: all kernels × all modes on WikiText.

    Returns:
        dict: {kernel: {mode: metrics_dict}}
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

    total = len(models) * len(modes)
    logger.info(f"Running {total} evaluations ({len(models)} kernels × {len(modes)} modes)")

    all_results: dict[str, dict[str, dict]] = {}

    for entry in models:
        kernel = entry.kernel
        logger.info(f"\n{'#'*70}")
        logger.info(f"# KERNEL: {kernel}")
        logger.info(f"{'#'*70}")

        kernel_results: dict[str, dict] = {}

        for mode in modes:
            result = _run_ablation_evaluation(
                entry=entry,
                mode=mode,
                wikitext_benchmark_dir=wikitext_benchmark_dir,
                results_dir=results_dir,
                limit_val_batches=limit_val_batches,
            )
            if result:
                kernel_results[mode] = result

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


def save_results(all_results: dict, results_dir: str = "data/ablation_results") -> str:
    """Save ablation results as CSV and Markdown table.

    Returns:
        str: Path to the Markdown summary file.
    """
    import csv
    from datetime import datetime

    os.makedirs(results_dir, exist_ok=True)

    # ---- Flatten into rows ----
    rows: list[dict] = []
    for kernel, mode_dict in all_results.items():
        for mode, metrics in mode_dict.items():
            pm = MODE_PARAMS.get(mode, {})
            rows.append({
                "kernel": kernel,
                "mode": mode,
                "A": _checkmark(pm.get("A", True)),
                "B": _checkmark(pm.get("B", True)),
                "C": _checkmark(pm.get("C", True)),
                "dt": _checkmark(pm.get("dt", True)),
                "D": _checkmark(pm.get("D", True)),
                "raw_nll": metrics.get("raw_nll"),
                "raw_ppl": metrics.get("raw_ppl"),
                "weighted_nll": metrics.get("weighted_nll"),
                "weighted_ppl": metrics.get("weighted_ppl"),
                "mdlm_masked_ppl": metrics.get("mdlm_masked_ppl"),
                "error": metrics.get("error", ""),
            })

    # ---- CSV ----
    csv_path = os.path.join(results_dir, "ablation_results.csv")
    fieldnames = [
        "kernel", "mode", "A", "B", "C", "dt", "D",
        "raw_nll", "raw_ppl", "weighted_nll", "weighted_ppl",
        "mdlm_masked_ppl", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"CSV saved to {csv_path}")

    # ---- Markdown Table ----
    md_path = os.path.join(results_dir, "ablation_summary.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w") as f:
        f.write("# MaRK SSM Parameter Ablation — Validation Loss on WikiText\n\n")
        f.write(f"> Generated: {timestamp}\n\n")

        # Main table
        f.write("## Results: All Kernels × All Ablation Modes\n\n")
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
            f.write(
                f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                f"| {row['A']} | {row['B']} | {row['C']} | {row['dt']} | {row['D']} "
                f"| {_fmt(row['weighted_ppl']):>13s} | {_fmt(row['raw_ppl']):>9s} | {_fmt(row['weighted_nll']):>14s} |\n"
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
            full_val = full_m.get("weighted_ppl")
            noA_val = noA_m.get("weighted_ppl")
            if full_val is not None and noA_val is not None:
                delta = float(noA_val) - float(full_val)
                f.write(
                    f"| {kernel:>9s} | {_fmt(full_val):>9s} | {_fmt(noA_val):>17s} "
                    f"| {_fmt(delta):>7s} | {delta:>+.4f} |\n"
                )
            else:
                f.write(f"| {kernel:>9s} | — | — | — | — |\n")
        f.write("\n")

        # ---- Analysis: BC_only vs full ----
        f.write("## Analysis: Mamba-Style Selection (BC_only vs full)\n\n")
        f.write(
            "If `BC_only` (B and C modulated, A/Δ/D frozen) performs close to `full`, "
            "then Mamba-style selection through B and C already captures much of "
            "MaRK's benefit. If `A_only` is close to `full`, then A-modulation "
            "is the key contribution.\n\n"
        )
        f.write("| Kernel | Full W-PPL | BC_only W-PPL | A_only W-PPL | dt_only W-PPL |\n")
        f.write("|--------|-----------|--------------|-------------|--------------|\n")
        for kernel in sorted(all_results.keys()):
            full_m = all_results[kernel].get("full", {})
            bc_m = all_results[kernel].get("BC_only", {})
            a_m = all_results[kernel].get("A_only", {})
            dt_m = all_results[kernel].get("dt_only", {})
            f.write(
                f"| {kernel:>9s} | {_fmt(full_m.get('weighted_ppl')):>9s} "
                f"| {_fmt(bc_m.get('weighted_ppl')):>12s} "
                f"| {_fmt(a_m.get('weighted_ppl')):>11s} "
                f"| {_fmt(dt_m.get('weighted_ppl')):>12s} |\n"
            )
        f.write("\n")

        # ---- Full table with all metrics ----
        f.write("## Full Results (All Metrics)\n\n")
        full_header = (
            "| Kernel | Mode | Raw NLL | Raw PPL | "
            "Weighted NLL | Weighted PPL | MDLM PPL |\n"
        )
        full_sep = (
            "|--------|------|---------|---------|"
            "-------------|-------------|----------|\n"
        )
        f.write(full_header)
        f.write(full_sep)
        for row in rows:
            f.write(
                f"| {row['kernel']:>9s} | {row['mode']:<14s} "
                f"| {_fmt(row['raw_nll']):>7s} | {_fmt(row['raw_ppl']):>7s} "
                f"| {_fmt(row['weighted_nll']):>11s} | {_fmt(row['weighted_ppl']):>11s} "
                f"| {_fmt(row['mdlm_masked_ppl']):>8s} |\n"
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
    return parser


def main(argv: list[str] | None = None) -> None:
    from .perplexity import parse_checkpoint_dir_overrides

    parser = build_arg_parser()
    args = parser.parse_args(argv)

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
    logger.info(f"Total evaluations: {len(models) * len(modes)}")

    all_results = run_ablation_suite(
        models=models,
        modes=modes,
        wikitext_dir=args.wikitext_dir,
        results_dir=args.output_dir,
        limit_val_batches=args.limit_val_batches,
        checkpoint_dir_overrides=checkpoint_dir_overrides,
    )

    md_path = save_results(all_results, results_dir=args.output_dir)

    # Print summary to console
    print("\n" + "=" * 80)
    print("ABLATION COMPLETE")
    print("=" * 80)
    print(f"Markdown summary: {md_path}")
    print(f"CSV results:      {os.path.join(args.output_dir, 'ablation_results.csv')}")

    # Print mini-table
    kernels_sorted = sorted(all_results.keys())
    if kernels_sorted:
        print(f"\n{'Kernel':>10s}", end="")
        for mode in modes:
            print(f" | {mode:<14s}", end="")
        print()
        print("-" * (10 + 18 * len(modes)))
        for kernel in kernels_sorted:
            print(f"{kernel:>10s}", end="")
            for mode in modes:
                m = all_results[kernel].get(mode, {})
                print(f" | {_fmt(m.get('weighted_ppl')):>14s}", end="")
            print()


if __name__ == "__main__":
    main()
