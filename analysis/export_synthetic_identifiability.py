"""CSV/JSON export helpers for the synthetic LPV identifiability figures.

The three synthetic scripts share this module so their audit tables have the
same schema: per-run rows, per-dataset summary statistics, and a small config
JSON describing the protocol.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any


def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def write_runs_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_out_dir(os.path.dirname(path) or ".")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(
    path: str,
    dataset_sizes: list[int],
    mean_errors,
    std_errors,
    ci_margin,
) -> None:
    """Summary rows match save_recovery_plot theory curve: theory_constant = mean[0]*sqrt(N0)."""
    ensure_out_dir(os.path.dirname(path) or ".")
    n0 = float(dataset_sizes[0])
    theory_constant = float(mean_errors[0]) * (n0**0.5)
    rows = []
    for i, N in enumerate(dataset_sizes):
        me = float(mean_errors[i])
        sm = float(std_errors[i]) if std_errors is not None else float("nan")
        ci = float(ci_margin[i])
        theory = theory_constant / (float(N) ** 0.5)
        rows.append(
            {
                "dataset_size": int(N),
                "mean_error": me,
                "std_error": sm,
                "ci_margin": ci,
                "ci_low": me - ci,
                "ci_high": me + ci,
                "theory_O_sqrt_N": theory,
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_size",
                "mean_error",
                "std_error",
                "ci_margin",
                "ci_low",
                "ci_high",
                "theory_O_sqrt_N",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def write_config_json(path: str, config: dict[str, Any]) -> None:
    ensure_out_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
