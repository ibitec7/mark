"""CSV export for the paper's Markov norm-vs-lag dynamics figure."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DYNAMICS_OUT = Path("analysis/results/dynamics")
KERNEL_NAMES = ("Hypernet", "Chebyshev", "DCT")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        logger.warning("No rows to write for %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote CSV: %s", path)


def export_markov_norm_vs_lag_csv(
    curve_data: dict[str, dict],
    output_csv: str = "analysis/results/dynamics/markov_norm_vs_lag_k4096.csv",
) -> None:
    """Write the exact curves shown in `plots/markov_norm_vs_lag_k4096.png`."""
    if not curve_data:
        logger.warning("Missing Markov norm curves; skipping CSV export")
        return

    rows = []
    for kernel_name in KERNEL_NAMES:
        if kernel_name not in curve_data:
            continue
        data = curve_data[kernel_name]

        for idx, timestep, norm_curve in zip(
            data["snapshot_indices"],
            data["snapshot_timesteps"],
            data["modulated_curves"],
            strict=True,
        ):
            for lag_i, value in enumerate(norm_curve):
                rows.append(
                    {
                        "lag_index": int(lag_i),
                        "kernel": kernel_name,
                        "snapshot_t": float(timestep),
                        "snapshot_idx": int(idx),
                        "markov_norm": float(value),
                        "series": "modulated",
                    }
                )

        for lag_i, value in enumerate(data["frozen_curve"]):
            rows.append(
                {
                    "lag_index": int(lag_i),
                    "kernel": kernel_name,
                    "snapshot_t": "",
                    "snapshot_idx": "",
                    "markov_norm": float(value),
                    "series": "frozen_backbone",
                }
            )

    _write_csv(
        Path(output_csv),
        ["lag_index", "kernel", "snapshot_t", "snapshot_idx", "markov_norm", "series"],
        rows,
    )
