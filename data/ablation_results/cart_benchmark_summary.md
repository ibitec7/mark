# MaRK CART-Weighted Validation Ablation — All Benchmark Datasets

> Generated: 2026-09-28 16:04:17
> Seeds: 10 (seed i = 42 + i×100)
> Estimator: sample std with `ddof=1`; 95% CI = mean ± 1.96 × SEM (same estimator as the WikiText leave-one-out SSM ablation)
> Benchmark root: `./data/benchmarks`
> Datasets (6): ag_news, arxiv, lambada, ptb, pubmed, wikitext
> Modes: full
> Evaluator: NeMo `Trainer.validate` with diffusion masking and CART weights
> **Primary metric:** `weighted_nll` = the **CART loss**. `weighted_ppl = exp(weighted_nll)` is the same number exponentiated.
> **Validation cap:** first 500 validation batches per dataset (datasets with fewer batches are evaluated in full)

## Results: CART loss (weighted NLL) per dataset

`weighted_nll` is the CART loss — the reportable number. `weighted_ppl = exp(weighted_nll)` is shown only as a convenience.

| Kernel | Dataset | n | CART loss (W-NLL) ↓ | W-PPL (=e^NLL) | Raw PPL ↓ |
|--------|---------|---|---|--------------------|---------------|-----------|
| chebyshev | ag_news   | 10 |     2.8025 ± 0.158 | 17.0007 ± 2.904 | 62.4774 ± 12.801 |
| chebyshev | arxiv     | 10 |     2.5491 ± 0.066 | 12.8630 ± 0.879 | 56.1032 ± 3.417 |
| chebyshev | lambada   | 10 |     2.3597 ± 0.112 | 10.7504 ± 1.271 | 33.5684 ± 5.057 |
| chebyshev | ptb       | 10 |     2.4286 ± 0.286 | 12.5884 ± 4.142 | 67.3230 ± 23.650 |
| chebyshev | pubmed    | 10 |     2.5514 ± 0.067 | 12.8940 ± 0.890 | 55.1051 ± 3.331 |
| chebyshev | wikitext  | 10 |     2.5068 ± 0.175 | 12.7079 ± 2.177 | 47.4108 ± 8.057 |
|       dct | ag_news   | 10 |     2.8229 ± 0.163 | 17.3888 ± 3.077 | 56.2247 ± 10.668 |
|       dct | arxiv     | 10 |     2.6060 ± 0.068 | 13.6207 ± 0.964 | 53.1435 ± 2.619 |
|       dct | lambada   | 10 |     2.4448 ± 0.118 | 11.7262 ± 1.473 | 34.4325 ± 4.999 |
|       dct | ptb       | 10 |     2.4368 ± 0.287 | 12.7016 ± 4.192 | 58.8911 ± 19.391 |
|       dct | pubmed    | 10 |     2.5545 ± 0.067 | 12.9337 ± 0.897 | 49.0520 ± 2.389 |
|       dct | wikitext  | 10 |     2.5174 ± 0.176 | 12.8466 ± 2.208 | 44.6406 ± 7.375 |
|  hypernet | ag_news   | 10 |     3.8700 ± 0.212 | 50.6894 ± 11.728 | 285.8680 ± 79.899 |
|  hypernet | arxiv     | 10 |     4.1260 ± 0.109 | 62.8333 ± 7.234 | 517.7661 ± 37.866 |
|  hypernet | lambada   | 10 |     3.6685 ± 0.176 | 40.7140 ± 7.720 | 194.6358 ± 40.461 |
|  hypernet | ptb       | 10 |     3.3192 ± 0.362 | 32.6882 ± 13.763 | 462.1609 ± 252.031 |
|  hypernet | pubmed    | 10 |     3.9411 ± 0.102 | 52.1267 ± 5.576 | 422.6095 ± 31.105 |
|  hypernet | wikitext  | 10 |     3.6755 ± 0.245 | 42.2653 ± 9.970 | 315.3795 ± 84.497 |

## Best kernel per dataset (lowest CART loss)

| Dataset | Best kernel | CART loss | 2nd | 3rd |
|---------|-------------|-----------|-----|-----|
| ag_news   | chebyshev   |    2.8025 | dct (2.8229) | hypernet (3.8700) |
| arxiv     | chebyshev   |    2.5491 | dct (2.6060) | hypernet (4.1260) |
| lambada   | chebyshev   |    2.3597 | dct (2.4448) | hypernet (3.6685) |
| ptb       | chebyshev   |    2.4286 | dct (2.4368) | hypernet (3.3192) |
| pubmed    | chebyshev   |    2.5514 | dct (2.5545) | hypernet (3.9411) |
| wikitext  | chebyshev   |    2.5068 | dct (2.5174) | hypernet (3.6755) |

## Kernel average across datasets (CART loss)

| Kernel | Mean CART loss | Min | Max | Datasets |
|--------|---------------|-----|-----|----------|
| chebyshev |         2.5330 | 2.3597 | 2.8025 | 6 |
|       dct |         2.5637 | 2.4368 | 2.8229 | 6 |
|  hypernet |         3.7667 | 3.3192 | 4.1260 | 6 |

## Artifacts

- `cart_raw.csv` — one row per kernel × dataset × mode × seed; `weighted_nll` is the CART loss
- `cart_benchmark_results.csv` — mean, std and 95% CI per kernel × dataset × mode
- `cart_<kernel>_stage1_<dataset>_<mode>_seed<seed>.json` — per-run evaluator output
