# MaRK CART-Weighted Validation Ablation — All Benchmark Datasets

> Generated: 2026-09-28 17:39:34
> Seeds: 10 (seed i = 42 + i×100)
> Estimator: sample std with `ddof=1`; 95% CI = mean ± 1.96 × SEM (same estimator as the WikiText leave-one-out SSM ablation)
> Benchmark root: `./data/benchmarks`
> Datasets (6): ag_news, arxiv, lambada, ptb, pubmed, wikitext
> Modes: full
> Evaluator: NeMo `Trainer.validate` with diffusion masking and CART weights
> RNG protocol: `loo` — each dataset is validated from the RNG state captured right after the model is built, matching the leave-one-out harness draw-for-draw (so WikiText is directly comparable with the LOO table)
> **Primary metric:** `weighted_nll` = the **CART loss**. `weighted_ppl = exp(weighted_nll)` is the same number exponentiated.
> **Validation cap:** first 500 validation batches per dataset (datasets with fewer batches are evaluated in full)

## Results: CART loss (weighted NLL) per dataset

`weighted_nll` is the CART loss — the reportable number. `weighted_ppl = exp(weighted_nll)` is shown only as a convenience.

| Kernel | Dataset | n | CART loss (W-NLL) ↓ | W-PPL (=e^NLL) | Raw PPL ↓ |
|--------|---------|---|---|--------------------|---------------|-----------|
| chebyshev | ag_news   | 10 |     2.8494 ± 0.139 | 17.6923 ± 2.660 | 58.4697 ± 11.016 |
| chebyshev | arxiv     | 10 |     2.5530 ± 0.058 | 12.8978 ± 0.772 | 56.0764 ± 3.381 |
| chebyshev | lambada   | 10 |     2.3954 ± 0.098 | 11.1013 ± 1.160 | 32.3033 ± 4.524 |
| chebyshev | ptb       | 10 |     2.4194 ± 0.257 | 12.2622 ± 3.760 | 67.3572 ± 21.205 |
| chebyshev | pubmed    | 10 |     2.5536 ± 0.057 | 12.9037 ± 0.757 | 55.0711 ± 3.215 |
| chebyshev | wikitext  | 10 |     2.5466 ± 0.155 | 13.1277 ± 2.044 | 45.0968 ± 7.148 |
|       dct | ag_news   | 10 |     2.8604 ± 0.142 | 17.9002 ± 2.694 | 53.7538 ± 10.627 |
|       dct | arxiv     | 10 |     2.6135 ± 0.065 | 13.7148 ± 0.910 | 52.6284 ± 2.789 |
|       dct | lambada   | 10 |     2.4707 ± 0.109 | 11.9986 ± 1.343 | 33.2838 ± 5.119 |
|       dct | ptb       | 10 |     2.4768 ± 0.271 | 13.1039 ± 4.219 | 57.3567 ± 21.061 |
|       dct | pubmed    | 10 |     2.5639 ± 0.063 | 13.0489 ± 0.844 | 48.4929 ± 2.410 |
|       dct | wikitext  | 10 |     2.5519 ± 0.138 | 13.1258 ± 1.843 | 41.9370 ± 6.302 |
|  hypernet | ag_news   | 10 |     3.9094 ± 0.198 | 52.3444 ± 11.242 | 269.9076 ± 82.336 |
|  hypernet | arxiv     | 10 |     4.1237 ± 0.102 | 62.5684 ± 6.737 | 517.2608 ± 41.933 |
|  hypernet | lambada   | 10 |     3.7035 ± 0.171 | 42.0634 ± 7.726 | 191.9839 ± 48.151 |
|  hypernet | ptb       | 10 |     3.3904 ± 0.306 | 33.4623 ± 11.788 | 461.5699 ± 326.237 |
|  hypernet | pubmed    | 10 |     3.9374 ± 0.097 | 51.8750 ± 5.318 | 421.7540 ± 33.931 |
|  hypernet | wikitext  | 10 |     3.7330 ± 0.228 | 44.3678 ± 9.836 | 297.1969 ± 73.487 |

## Best kernel per dataset (lowest CART loss)

| Dataset | Best kernel | CART loss | 2nd | 3rd |
|---------|-------------|-----------|-----|-----|
| ag_news   | chebyshev   |    2.8494 | dct (2.8604) | hypernet (3.9094) |
| arxiv     | chebyshev   |    2.5530 | dct (2.6135) | hypernet (4.1237) |
| lambada   | chebyshev   |    2.3954 | dct (2.4707) | hypernet (3.7035) |
| ptb       | chebyshev   |    2.4194 | dct (2.4768) | hypernet (3.3904) |
| pubmed    | chebyshev   |    2.5536 | dct (2.5639) | hypernet (3.9374) |
| wikitext  | chebyshev   |    2.5466 | dct (2.5519) | hypernet (3.7330) |

## Kernel average across datasets (CART loss)

| Kernel | Mean CART loss | Min | Max | Datasets |
|--------|---------------|-----|-----|----------|
| chebyshev |         2.5529 | 2.3954 | 2.8494 | 6 |
|       dct |         2.5895 | 2.4707 | 2.8604 | 6 |
|  hypernet |         3.7996 | 3.3904 | 4.1237 | 6 |

## Artifacts

- `cart_raw.csv` — one row per kernel × dataset × mode × seed; `weighted_nll` is the CART loss
- `cart_benchmark_results.csv` — mean, std and 95% CI per kernel × dataset × mode
- `cart_<kernel>_stage1_<dataset>_<mode>_seed<seed>.json` — per-run evaluator output
