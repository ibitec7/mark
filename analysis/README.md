# Analysis Artifact Manifest

This directory contains the reviewer-facing analysis code used for the paper figures. The active figure list is taken from `drafts/paper/research_paper.tex`; exploratory plots and benchmark helpers that are not part of the submission are intentionally omitted from this surface.

## Paper Artifacts

| Paper figure | Canonical output | Generator |
| --- | --- | --- |
| AQS certificate | `plots/aqs_certificate.png` | `python -m analysis.aqs_certificate` |
| Hypernet synthetic LPV recovery | `plots/hypernet_synthetic_recovery.png` | `python analysis/hypernet_synthetic_lpv.py` |
| Chebyshev synthetic LPV recovery | `plots/chebyshev_synthetic_recovery.png` | `python analysis/chebyshev_synthetic_lpv.py` |
| DCT synthetic LPV recovery | `plots/dct_synthetic_recovery.png` | `python analysis/dct_synthetic_lpv.py` |
| Markov norm vs lag, `K=4096` | `plots/markov_norm_vs_lag_k4096.png` | `python -m analysis.dynamics_analysis` |

The scripts also write compact CSV or JSON audit trails under `analysis/results/` where useful:

- `analysis/results/aqs_certificate/`: per-layer margins and run metadata for the AQS certificate.
- `analysis/results/synthetic_identifiability/`: run-level and summary tables for the synthetic LPV recovery plots.
- `analysis/results/dynamics/markov_norm_vs_lag_k4096.csv`: plot-ready Markov norm curves for the dynamics figure.

## Reproduction Notes

Run commands from the repository root. The AQS and dynamics scripts require the local model checkpoints named in each script. The synthetic LPV scripts are self-contained apart from the standard Python, NumPy, PyTorch, and Matplotlib dependencies.

## Cleanup Policy

Generated figures are canonical under `plots/`. Avoid storing duplicate PNG copies directly in `analysis/`. Large or exploratory intermediate files should be regenerated from scripts or archived outside the supplementary analysis surface unless they are listed above as audit data.
