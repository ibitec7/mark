# MaRK CART-Weighted Validation Ablation — All Benchmark Datasets

> Generated: 2026-09-28 15:43:23
> Seeds: 10 (seed i = 42 + i×100)
> Estimator: sample std with `ddof=1`; 95% CI = mean ± 1.96 × SEM (same estimator as the WikiText leave-one-out SSM ablation)
> Benchmark root: `./data/benchmarks`
> Datasets (6): ag_news, arxiv, lambada, ptb, pubmed, wikitext
> Modes: full
> Evaluator: NeMo `Trainer.validate` with diffusion masking and CART weights
> **Validation cap:** first 500 validation batches per dataset (datasets with fewer batches are evaluated in full)

## Results: CART-weighted (and raw) perplexity per dataset

| Kernel | Dataset | n | Weighted PPL ↓ | Raw PPL ↓ | Weighted NLL ↓ |
|--------|---------|---|---|---------------|-----------|----------------|
| chebyshev | ag_news   | 10 |  17.0007 ± 2.904 | 62.4774 ± 12.801 | 2.8025 ± 0.158 |
| chebyshev | arxiv     | 10 |  12.8630 ± 0.879 | 56.1032 ± 3.417 | 2.5491 ± 0.066 |
| chebyshev | lambada   | 10 |  10.7504 ± 1.271 | 33.5684 ± 5.057 | 2.3597 ± 0.112 |
| chebyshev | ptb       | 10 |  12.5884 ± 4.142 | 67.3230 ± 23.650 | 2.4286 ± 0.286 |
| chebyshev | pubmed    | 10 |  12.8940 ± 0.890 | 55.1051 ± 3.331 | 2.5514 ± 0.067 |
| chebyshev | wikitext  | 10 |  12.7079 ± 2.177 | 47.4108 ± 8.057 | 2.5068 ± 0.175 |
|       dct | ag_news   | 10 |  17.3888 ± 3.077 | 56.2247 ± 10.668 | 2.8229 ± 0.163 |
|       dct | arxiv     | 10 |  13.6207 ± 0.964 | 53.1435 ± 2.619 | 2.6060 ± 0.068 |
|       dct | lambada   | 10 |  11.7262 ± 1.473 | 34.4325 ± 4.999 | 2.4448 ± 0.118 |
|       dct | ptb       | 10 |  12.7016 ± 4.192 | 58.8911 ± 19.391 | 2.4368 ± 0.287 |
|       dct | pubmed    | 10 |  12.9337 ± 0.897 | 49.0520 ± 2.389 | 2.5545 ± 0.067 |
|       dct | wikitext  | 10 |  12.8466 ± 2.208 | 44.6406 ± 7.375 | 2.5174 ± 0.176 |
|  hypernet | ag_news   | 10 | 50.6894 ± 11.728 | 285.8680 ± 79.899 | 3.8700 ± 0.212 |
|  hypernet | arxiv     | 10 |  62.8333 ± 7.234 | 517.7661 ± 37.866 | 4.1260 ± 0.109 |
|  hypernet | lambada   | 10 |  40.7140 ± 7.720 | 194.6358 ± 40.461 | 3.6685 ± 0.176 |
|  hypernet | ptb       | 10 | 32.6882 ± 13.763 | 462.1609 ± 252.031 | 3.3192 ± 0.362 |
|  hypernet | pubmed    | 10 |  52.1267 ± 5.576 | 422.6095 ± 31.105 | 3.9411 ± 0.102 |
|  hypernet | wikitext  | 10 |  42.2653 ± 9.970 | 315.3795 ± 84.497 | 3.6755 ± 0.245 |

## Best kernel per dataset (lowest CART-weighted PPL)

| Dataset | Best kernel | Weighted PPL | 2nd | 3rd |
|---------|-------------|--------------|-----|-----|
| ag_news   | chebyshev   |      17.0007 | dct (17.3888) | hypernet (50.6894) |
| arxiv     | chebyshev   |      12.8630 | dct (13.6207) | hypernet (62.8333) |
| lambada   | chebyshev   |      10.7504 | dct (11.7262) | hypernet (40.7140) |
| ptb       | chebyshev   |      12.5884 | dct (12.7016) | hypernet (32.6882) |
| pubmed    | chebyshev   |      12.8940 | dct (12.9337) | hypernet (52.1267) |
| wikitext  | chebyshev   |      12.7079 | dct (12.8466) | hypernet (42.2653) |

## Kernel average across datasets (CART-weighted PPL)

| Kernel | Mean W-PPL | Min | Max | Datasets |
|--------|-----------|-----|-----|----------|
| chebyshev |   13.1341 | 10.7504 | 17.0007 | 6 |
|       dct |   13.5363 | 11.7262 | 17.3888 | 6 |
|  hypernet |   46.8861 | 32.6882 | 62.8333 | 6 |

## Artifacts

- `cart_raw.csv` — one row per kernel × dataset × mode × seed (raw metrics)
- `cart_benchmark_results.csv` — mean, std and 95% CI per kernel × dataset × mode
- `cart_<kernel>_stage1_<dataset>_<mode>_seed<seed>.json` — per-run evaluator output
