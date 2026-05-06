# ================================================
# DeepEval Benchmark Harness for MaRK Adapter Ablation
#
# Runs downstream task benchmarks via DeepEval to compare MaRK kernel
# variants (Hypernet, Chebyshev, DCT) against each other and the base Hydra.
#
# Supported benchmarks:
#   - MMLU (Massive Multitask Language Understanding)
#   - HellaSwag (Commonsense NLI)
#   - ARC (AI2 Reasoning Challenge)
#   - GLUE CoLA (Linguistic Acceptability)
#   - GLUE SST-2 (Sentiment Analysis)
#
# Run: cd /home/ibrahim/Desktop/experiment && python -m src.benchmark
# ================================================

import os
import json
import logging
from pathlib import Path

from .eval import EvalModel
from .utils import log_setup

LOG_FILE = os.path.join("logs", "benchmark.log")
LOG_LEVEL = logging.INFO
Path("logs").mkdir(exist_ok=True)
logger = log_setup("BenchmarkLogger", LOG_FILE, LOG_LEVEL)

RESULTS_DIR = "analysis/results"

# Model weight paths (MaRK checkpoints match ``analysis/dynamics_analysis``)
WEIGHT_PATHS: dict[str, str] = {
    "base":      "models/hydra_bert_23layers.pt",
    "hypernet":  "models/hydra_mark_hypernet_apr/best_hydra_mark.ckpt",
    "chebyshev": "models/hydra_mark_chebyshev_apr/best_hydra_mark.ckpt",
    "dct":       "models/hydra_mark_dct_apr/best_hydra_mark.ckpt",
}

KERNEL_MAP: dict[str, str] = {
    "base":      "chebyshev",
    "hypernet":  "hypernet",
    "chebyshev": "chebyshev",
    "dct":       "dct",
}


def get_benchmark(name: str):
    """
    Lazily import and return a DeepEval benchmark instance.

    Args:
        name (str): Benchmark name (mmlu, hellaswag, arc, cola, sst2).

    Returns:
        Benchmark instance with evaluate() method.
    """
    if name == "mmlu":
        from deepeval.benchmarks import MMLU
        return MMLU()
    elif name == "hellaswag":
        from deepeval.benchmarks import HellaSwag
        return HellaSwag()
    elif name == "arc":
        from deepeval.benchmarks import ARC
        return ARC()
    elif name == "cola":
        from deepeval.benchmarks import GLUE
        return GLUE(task="cola")
    elif name == "sst2":
        from deepeval.benchmarks import GLUE
        return GLUE(task="sst2")
    else:
        raise ValueError(f"Unknown benchmark: {name}. Choose from: mmlu, hellaswag, arc, cola, sst2")


def run_benchmark(
    variant: str,
    benchmark_name: str,
    config_path: str = "configs/hydra.yaml",
    device: str = "cuda",
) -> dict:
    """
    Run a single benchmark on a single model variant.

    Args:
        variant (str): Model variant ("base", "hypernet", "chebyshev", "dct").
        benchmark_name (str): Benchmark name.
        config_path (str): Path to Hydra config YAML.
        device (str): Compute device.

    Returns:
        dict: Benchmark results with scores and metadata.
    """
    weights_path = WEIGHT_PATHS[variant]
    kernel = KERNEL_MAP[variant]

    logger.info(f"Loading model: {variant} (kernel={kernel})")
    eval_model = EvalModel(
        weights_path=weights_path,
        config_path=config_path,
        kernel=kernel,
        device=device,
    )

    logger.info(f"Running benchmark: {benchmark_name}")
    benchmark = get_benchmark(benchmark_name)

    try:
        benchmark.evaluate(model=eval_model)
        score = benchmark.overall_score
        logger.info(f"  [{variant}] {benchmark_name}: {score:.4f}")
        return {
            "variant": variant,
            "benchmark": benchmark_name,
            "score": score,
            "task_scores": getattr(benchmark, "task_scores", None),
        }
    except Exception as e:
        logger.error(f"  [{variant}] {benchmark_name} FAILED: {e}")
        return {
            "variant": variant,
            "benchmark": benchmark_name,
            "score": None,
            "error": str(e),
        }


def run_all_benchmarks(
    variants: list[str] | None = None,
    benchmarks: list[str] | None = None,
    config_path: str = "configs/hydra.yaml",
    device: str = "cuda",
) -> dict[str, dict[str, dict]]:
    """
    Run all benchmarks on all model variants.

    Args:
        variants (list[str] | None): Model variants. Defaults to all 4.
        benchmarks (list[str] | None): Benchmarks. Defaults to recommended set.
        config_path (str): Path to Hydra config YAML.
        device (str): Compute device.

    Returns:
        dict[str, dict[str, dict]]: {variant: {benchmark: results}}
    """
    variants = variants or list(WEIGHT_PATHS.keys())
    benchmarks = benchmarks or ["mmlu", "hellaswag", "arc"]

    all_results = {}
    for variant in variants:
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating variant: {variant}")
        logger.info(f"{'='*60}")
        all_results[variant] = {}
        for bench_name in benchmarks:
            result = run_benchmark(
                variant=variant,
                benchmark_name=bench_name,
                config_path=config_path,
                device=device,
            )
            all_results[variant][bench_name] = result

    return all_results


def save_benchmark_results(results: dict, output_path: str | None = None) -> str:
    """
    Save benchmark results to JSON.

    Args:
        results (dict): Full benchmark results.
        output_path (str | None): Output path.

    Returns:
        str: Path to saved JSON.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = output_path or os.path.join(RESULTS_DIR, "benchmark_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Results Summary")
    logger.info("=" * 80)

    variants = list(results.keys())
    benchmarks = list(next(iter(results.values())).keys()) if results else []

    header = f"{'Benchmark':>15s}"
    for v in variants:
        header += f" | {v:>12s}"
    logger.info(header)
    logger.info("-" * len(header))

    for bench in benchmarks:
        row = f"{bench:>15s}"
        for v in variants:
            r = results.get(v, {}).get(bench, {})
            score = r.get("score")
            if score is not None:
                row += f" | {score:>12.4f}"
            else:
                row += f" | {'FAILED':>12s}"
        logger.info(row)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepEval benchmarks for MaRK")
    parser.add_argument("--variants", nargs="+", default=None, help="Model variants to evaluate")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Benchmarks to run")
    parser.add_argument("--config", default="configs/hydra.yaml", help="Model config path")
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = run_all_benchmarks(
        variants=args.variants,
        benchmarks=args.benchmarks,
        config_path=args.config,
        device=args.device,
    )

    save_benchmark_results(results, output_path=args.output)
    logger.info("Done.")
