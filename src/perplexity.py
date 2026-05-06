import argparse
import torch
import gc
import json
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

from omegaconf import DictConfig

from .utils import load_config, log_setup, get_best_checkpoint, arrow_dataloader, load_compatible_state_dict
from .hydra_model import HydraForMaskedLM
from .nemo import NemoForMaskedLM

from nemo import lightning as nl

torch.serialization.add_safe_globals([DictConfig])
torch.manual_seed(42)

LOG_DIR = "./logs"
BENCHMARKS_DIR = "./data/benchmarks"
RESULTS_DIR = "./data/benchmark_results"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = log_setup(log_name="PerplexityLogger", log_file=os.path.join(LOG_DIR, "perplexity.log"), level=logging.INFO)
SUPPORTED_KERNELS = ("chebyshev", "dct", "hypernet")


def resolve_runtime_path(path_str: str) -> str:
    """Map a host-style repository path into the active runtime's project root when needed."""
    path = Path(os.path.expanduser(path_str))
    if path.exists():
        return str(path)

    if not path.is_absolute():
        candidate = (PROJECT_ROOT / path).resolve()
        return str(candidate) if candidate.exists() else str(path)

    for ancestor in path.parents:
        candidate = PROJECT_ROOT / path.relative_to(ancestor)
        if candidate.exists():
            return str(candidate)

    return str(path)


@dataclass
class ModelEntry:
    """Registry entry for a single model checkpoint to benchmark."""
    kernel: str
    stage: int
    config_path: str
    checkpoint_suffix: str  # filename suffix inside checkpoint dir, e.g. "" for ``best_hydra_mark.ckpt``
    checkpoint_search_dirs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return f"{self.kernel}_stage{self.stage}"

    @property
    def checkpoint_dirs(self) -> list[str]:
        if self.checkpoint_search_dirs:
            return [resolve_runtime_path(directory) for directory in self.checkpoint_search_dirs]

        return [
            resolve_runtime_path(f"./models/hydra_mark_{self.kernel}"),
            resolve_runtime_path(f"./models/hydra_mark_{self.kernel}_apr"),
        ]

    def with_checkpoint_dirs(self, checkpoint_dirs: list[str]) -> "ModelEntry":
        return ModelEntry(
            kernel=self.kernel,
            stage=self.stage,
            config_path=self.config_path,
            checkpoint_suffix=self.checkpoint_suffix,
            checkpoint_search_dirs=tuple(checkpoint_dirs),
        )

    @property
    def checkpoint_path(self) -> str:
        checkpoint_name = f"best_hydra_mark{self.checkpoint_suffix}.ckpt"

        for checkpoint_dir in self.checkpoint_dirs:
            candidate = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.exists(candidate):
                return candidate

        for checkpoint_dir in self.checkpoint_dirs:
            fallback = get_best_checkpoint(checkpoint_dir)
            if fallback is not None:
                return fallback

        return os.path.join(self.checkpoint_dirs[0], checkpoint_name)


# Released evaluation set: one checkpoint per kernel (matches the paper ablation table).
MODEL_REGISTRY: list[ModelEntry] = [
    ModelEntry("hypernet", 1, "configs/benchmark_config_hypernet_stage1.yaml", ""),
    ModelEntry("chebyshev", 1, "configs/benchmark_config_chebyshev_stage1.yaml", ""),
    ModelEntry("dct", 1, "configs/benchmark_config_dct_stage1.yaml", ""),
]


def parse_checkpoint_dir_overrides(values: list[str]) -> dict[str, list[str]]:
    """Parse repeated ``KERNEL=DIR`` CLI overrides into a kernel-to-directories mapping."""
    overrides: dict[str, list[str]] = {kernel: [] for kernel in SUPPORTED_KERNELS}

    for value in values:
        kernel, separator, directory = value.partition("=")
        kernel = kernel.strip().lower()
        directory = directory.strip()

        if separator != "=" or not kernel or not directory:
            raise ValueError(
                "Checkpoint directory overrides must use the format KERNEL=DIR, "
                "for example: --checkpoint-dir chebyshev=/models/run_a"
            )
        if kernel not in overrides:
            raise ValueError(
                f"Unsupported kernel '{kernel}'. Expected one of: {', '.join(SUPPORTED_KERNELS)}"
            )

        overrides[kernel].append(directory)

    return {kernel: directories for kernel, directories in overrides.items() if directories}


def apply_checkpoint_dir_overrides(
    models: list[ModelEntry],
    checkpoint_dir_overrides: dict[str, list[str]] | None,
) -> list[ModelEntry]:
    """Return model entries with per-kernel checkpoint directory overrides applied."""
    if not checkpoint_dir_overrides:
        return models

    return [
        entry.with_checkpoint_dirs(checkpoint_dir_overrides[entry.kernel])
        if entry.kernel in checkpoint_dir_overrides
        else entry
        for entry in models
    ]


def discover_datasets(benchmarks_dir: str = BENCHMARKS_DIR) -> list[str]:
    """Auto-discover benchmark dataset subdirectories.

    Args:
        benchmarks_dir (str, optional): Root directory containing dataset folders. Defaults to BENCHMARKS_DIR.

    Returns:
        list[str]: Sorted list of dataset directory names.
    """
    base = Path(benchmarks_dir)
    if not base.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")
    datasets = sorted(
        [
            d.name
            for d in base.iterdir()
            if d.is_dir() and not d.name.startswith(".") and any(d.rglob("*.parquet"))
        ]
    )
    if not datasets:
        raise FileNotFoundError(f"No dataset subdirectories found in {benchmarks_dir}")
    return datasets


def _load_model_and_trainer(
    entry: ModelEntry,
) -> tuple[NemoForMaskedLM, nl.Trainer, DictConfig, str | None]:
    """Load a model from checkpoint and create a trainer for benchmark evaluation.

    Args:
        entry (ModelEntry): Registry entry specifying kernel, stage, and config.

    Returns:
        tuple[NemoForMaskedLM, nl.Trainer, DictConfig, str | None]: The NeMo model wrapper,
        trainer, train config, and checkpoint path.
    """
    train_config = load_config(entry.config_path, dict_config=True)
    model_config = load_config(entry.config_path, pretrained_config=True)

    weights = torch.load(train_config["weights_path"], map_location="cpu", weights_only=False)
    model_base = HydraForMaskedLM(config=model_config)
    missing_keys, unexpected_keys, mismatched_keys = load_compatible_state_dict(model_base, weights)
    logger.info(
        f"[{entry.name}] Base-weight compatibility: missing={len(missing_keys)} "
        f"unexpected={len(unexpected_keys)} mismatched={len(mismatched_keys)}"
    )
    if missing_keys:
        logger.info(f"[{entry.name}] Missing keys from base weights (using initialized values): {missing_keys}")
    if unexpected_keys:
        logger.warning(f"[{entry.name}] Unexpected keys in base weights: {unexpected_keys}")
    if mismatched_keys:
        logger.warning(f"[{entry.name}] Skipped {len(mismatched_keys)} mismatched base-weight tensors")
        logger.warning(f"[{entry.name}] Example mismatch: {mismatched_keys[0]}")
    del weights

    model_base = model_base.to(device="cuda")

    if train_config["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(train_config.get("matmul_precision", "high"))

    trainer_kwargs = dict(train_config["trainer"])
    trainer_kwargs.pop("strategy", None)

    trainer = nl.Trainer(
        **trainer_kwargs,
        max_epochs=1,
    )

    model = NemoForMaskedLM(config=train_config, tracker=None, trainer=trainer, model=model_base)
    model.to(device="cuda")

    # Load the Lightning checkpoint directly into the NemoForMaskedLM model.
    # trainer.validate(ckpt_path=...) does not reliably restore weights before running
    # validation — relying on it leaves MaRK adapter weights randomly initialized.
    # Loading explicitly with strict=False loads all matching keys (base Hydra + MaRK
    # adapters) and silently skips any shape mismatches, which is the desired behaviour.
    ckpt_path = entry.checkpoint_path
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys, mismatched_keys = load_compatible_state_dict(model, ckpt["state_dict"])
        logger.info(
            f"[{entry.name}] Checkpoint compatibility: missing={len(missing_keys)} "
            f"unexpected={len(unexpected_keys)} mismatched={len(mismatched_keys)}"
        )
        if missing_keys:
            logger.info(f"[{entry.name}] Missing keys from checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"[{entry.name}] Unexpected keys from checkpoint: {unexpected_keys}")
        if mismatched_keys:
            logger.warning(f"[{entry.name}] Skipped {len(mismatched_keys)} mismatched checkpoint tensors")
            logger.warning(f"[{entry.name}] Example checkpoint mismatch: {mismatched_keys[0]}")
        if missing_keys or unexpected_keys or mismatched_keys:
            raise RuntimeError(
                f"Checkpoint load for {entry.name} is incomplete: "
                f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} mismatched={len(mismatched_keys)}. "
                "Refusing to benchmark a partially initialized model."
            )
        logger.info(f"[{entry.name}] Loaded checkpoint: {ckpt_path}")
        del ckpt
    else:
        logger.warning(f"[{entry.name}] Checkpoint not found: {ckpt_path}, using base weights only")

    return model, trainer, train_config, None  # Checkpoint already loaded; pass None to trainer.validate.


def _evaluate_single(
    model: NemoForMaskedLM,
    trainer: nl.Trainer,
    train_config: DictConfig,
    ckpt_path: str | None,
    dataset_name: str,
    dataset_dir: str,
    output_path: str,
) -> dict | None:
    """Run validation on a single dataset and return metrics.

    Args:
        model (NemoForMaskedLM): The NeMo model wrapper.
        trainer (nl.Trainer): Lightning trainer instance.
        train_config (DictConfig): Training configuration.
        ckpt_path (str | None): Checkpoint path for loading, or None.
        dataset_name (str): Name of the benchmark dataset.
        dataset_dir (str): Path to the dataset directory.
        output_path (str): Path to write the JSON results.

    Returns:
        dict | None: Benchmark metrics dict, or None on failure.
    """
    model.reset_benchmark_state()
    model.val_dir = dataset_dir
    model.benchmark_path = output_path
    model._val_dl = None  # force dataloader rebuild

    val_dl = arrow_dataloader(
        data_dir=dataset_dir,
        split="validation",
        batch_size=train_config.get("batch_size", 1),
        num_workers=train_config.get("num_workers", 4),
        keep_in_memory=True,
    )

    with torch.inference_mode():
        trainer.validate(model, val_dl, ckpt_path=ckpt_path)

    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return None


def main(
    models: list[ModelEntry] | None = None,
    benchmarks_dir: str = BENCHMARKS_DIR,
    results_dir: str = RESULTS_DIR,
    limit_val_batches: int | float | None = None,
    checkpoint_dir_overrides: dict[str, list[str]] | None = None,
) -> dict:
    """Run perplexity benchmarks across all models and datasets.

    Args:
        models (list[ModelEntry] | None, optional): Models to benchmark. Defaults to MODEL_REGISTRY.
        benchmarks_dir (str, optional): Directory containing benchmark datasets. Defaults to BENCHMARKS_DIR.
        results_dir (str, optional): Directory to write JSON results. Defaults to RESULTS_DIR.
        limit_val_batches (int | float | None, optional): Limit validation batches (for smoke tests). Defaults to None.
        checkpoint_dir_overrides (dict[str, list[str]] | None, optional): Per-kernel checkpoint search
            directories supplied by the CLI. Defaults to None.

    Returns:
        dict: Nested dict of {model_name: {dataset_name: metrics_dict}}.
    """
    if models is None:
        models = MODEL_REGISTRY

    models = apply_checkpoint_dir_overrides(models, checkpoint_dir_overrides)

    datasets = discover_datasets(benchmarks_dir)
    logger.info(f"Discovered {len(datasets)} datasets: {datasets}")
    logger.info(
        f"Running benchmarks for {len(models)} models × {len(datasets)} datasets = {len(models) * len(datasets)} evaluations"
    )

    os.makedirs(results_dir, exist_ok=True)
    all_results: dict[str, dict[str, dict]] = {}

    for entry in models:
        logger.info(f"=== Loading model: {entry.name} ===")
        model, trainer, train_config, ckpt_path = _load_model_and_trainer(entry)

        if limit_val_batches is not None:
            trainer.limit_val_batches = limit_val_batches

        model_results: dict[str, dict] = {}

        for ds_name in datasets:
            ds_dir = os.path.join(benchmarks_dir, ds_name)
            output_path = os.path.join(results_dir, f"{entry.name}_{ds_name}.json")

            logger.info(f"  Evaluating {entry.name} on {ds_name}...")
            try:
                metrics = _evaluate_single(model, trainer, train_config, ckpt_path, ds_name, ds_dir, output_path)
                if metrics:
                    model_results[ds_name] = metrics
                    logger.info(f"  {ds_name}: raw_ppl={metrics.get('raw_ppl', 'N/A'):.4f}, weighted_ppl={metrics.get('weighted_ppl', 'N/A'):.4f}")
                else:
                    logger.warning(f"  {ds_name}: No metrics produced")
            except Exception as e:
                logger.error(f"  {ds_name}: Evaluation failed: {e}")
                model_results[ds_name] = {"error": str(e)}

        all_results[entry.name] = model_results

        # Cleanup model to free VRAM before loading next
        del model, trainer, train_config
        torch.cuda.empty_cache()
        gc.collect()

    # Write aggregated results
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``python -m src.perplexity``."""
    parser = argparse.ArgumentParser(description="Run perplexity benchmarks across Hydra/MaRK checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        metavar="KERNEL=DIR",
        help=(
            "Checkpoint directory override for a kernel. Repeat the flag to search multiple directories or "
            "configure multiple kernels, for example: --checkpoint-dir chebyshev=/models/a "
            "--checkpoint-dir chebyshev=/models/b --checkpoint-dir dct=/models/c"
        ),
    )
    parser.add_argument("--benchmarks-dir", default=BENCHMARKS_DIR, help="Benchmark dataset root directory.")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory for JSON benchmark outputs.")
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Optional validation batch limit for smoke tests. Accepts Lightning int/float-compatible values.",
    )
    return parser


def cli(argv: list[str] | None = None) -> dict:
    """CLI entrypoint for ``python -m src.perplexity``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        checkpoint_dir_overrides = parse_checkpoint_dir_overrides(args.checkpoint_dir)
    except ValueError as exc:
        parser.error(str(exc))

    return main(
        benchmarks_dir=args.benchmarks_dir,
        results_dir=args.results_dir,
        limit_val_batches=args.limit_val_batches,
        checkpoint_dir_overrides=checkpoint_dir_overrides,
    )


if __name__ == "__main__":
    cli()