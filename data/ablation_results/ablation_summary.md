# MaRK SSM Parameter Ablation — Validation Loss on WikiText

> Generated: 2026-07-27 06:19:49
> Seeds: 10  |  Seed offset: 42 + i×100

**Note:** Results are reported as mean ± 95% CI across 10 seeds. Seed 1 uses `torch.manual_seed(42)`, subsequent seeds use 42 + i×100.

## Results: All Kernels × All Ablation Modes

| Kernel | Mode | A | B | C | dt | D | n | Weighted PPL ↓ | Raw PPL ↓ | Weighted NLL ↓ |
|--------|------|---|---|---|----|---|---|---------------|-----------|----------------|
|  hypernet | full           | ✓ | ✓ | ✓ | ✓ | ✓ | 10 | 44.3824 ± 15.879 | 297.2076 ± 118.371 | 3.7333 ± 0.367 |
|  hypernet | all_except_A   | ✗ | ✓ | ✓ | ✓ | ✓ | 10 | 53.2564 ± 19.594 | 379.5826 ± 156.721 | 3.9118 ± 0.379 |
|  hypernet | A_only         | ✓ | ✗ | ✗ | ✗ | ✗ | 10 | 63.3135 ± 25.352 | 384.8417 ± 145.620 | 4.0735 ± 0.411 |
|  hypernet | dt_only        | ✗ | ✗ | ✗ | ✓ | ✗ | 10 | 61.2869 ± 24.373 | 435.2180 ± 175.766 | 4.0421 ± 0.408 |
|  hypernet | BC_only        | ✗ | ✓ | ✓ | ✗ | ✗ | 10 | 63.1964 ± 24.742 | 456.6219 ± 190.343 | 4.0747 ± 0.403 |
|  hypernet | all_except_dt  | ✓ | ✓ | ✓ | ✗ | ✓ | 10 | 44.7728 ± 16.066 | 299.5020 ± 123.077 | 3.7417 ± 0.368 |
|  hypernet | D_only         | ✗ | ✗ | ✗ | ✗ | ✓ | 10 | 46.5930 ± 16.496 | 341.0452 ± 143.035 | 3.7834 ± 0.362 |
|  hypernet | none           | ✗ | ✗ | ✗ | ✗ | ✗ | 10 | 58.3424 ± 22.795 | 407.3940 ± 161.289 | 3.9957 ± 0.400 |
| chebyshev | full           | ✓ | ✓ | ✓ | ✓ | ✓ | 10 | 13.1270 ± 3.296 | 45.0050 ± 11.634 | 2.5465 ± 0.250 |
| chebyshev | all_except_A   | ✗ | ✓ | ✓ | ✓ | ✓ | 10 | 16.6531 ± 4.435 | 70.7997 ± 20.934 | 2.7808 ± 0.265 |
| chebyshev | A_only         | ✓ | ✗ | ✗ | ✗ | ✗ | 10 | 32.3204 ± 9.834 | 228.8546 ± 102.246 | 3.4346 ± 0.301 |
| chebyshev | dt_only        | ✗ | ✗ | ✗ | ✓ | ✗ | 10 | 53.4661 ± 19.808 | 377.2061 ± 159.773 | 3.9174 ± 0.370 |
| chebyshev | BC_only        | ✗ | ✓ | ✓ | ✗ | ✗ | 10 | 22.1444 ± 6.589 | 102.8428 ± 32.581 | 3.0580 ± 0.296 |
| chebyshev | all_except_dt  | ✓ | ✓ | ✓ | ✗ | ✓ | 10 | 13.2504 ± 3.325 | 46.4961 ± 12.235 | 2.5559 ± 0.249 |
| chebyshev | D_only         | ✗ | ✗ | ✗ | ✗ | ✓ | 10 | 52.2861 ± 19.216 | 370.8523 ± 158.005 | 3.8960 ± 0.367 |
| chebyshev | none           | ✗ | ✗ | ✗ | ✗ | ✗ | 10 | 58.4640 ± 22.406 | 413.3920 ± 170.696 | 4.0026 ± 0.382 |
|       dct | full           | ✓ | ✓ | ✓ | ✓ | ✓ | 10 | 13.1259 ± 2.972 | 41.9436 ± 10.176 | 2.5520 ± 0.223 |
|       dct | all_except_A   | ✗ | ✓ | ✓ | ✓ | ✓ | 10 | 16.8730 ± 3.855 | 78.1390 ± 24.534 | 2.8025 ± 0.227 |
|       dct | A_only         | ✓ | ✗ | ✗ | ✗ | ✗ | 10 | 26.2726 ± 6.380 | 173.5442 ± 64.356 | 3.2424 ± 0.240 |
|       dct | dt_only        | ✗ | ✗ | ✗ | ✓ | ✗ | 10 | 56.7725 ± 19.382 | 391.2852 ± 151.016 | 3.9875 ± 0.337 |
|       dct | BC_only        | ✗ | ✓ | ✓ | ✗ | ✗ | 10 | 21.7474 ± 5.181 | 120.0516 ± 40.127 | 3.0542 ± 0.236 |
|       dct | all_except_dt  | ✓ | ✓ | ✓ | ✗ | ✓ | 10 | 13.6418 ± 3.178 | 43.7888 ± 10.538 | 2.5892 ± 0.230 |
|       dct | D_only         | ✗ | ✗ | ✗ | ✗ | ✓ | 10 | 51.7190 ± 17.334 | 336.7832 ± 125.902 | 3.8960 ± 0.331 |
|       dct | none           | ✗ | ✗ | ✗ | ✗ | ✗ | 10 | 57.1807 ± 19.183 | 394.3660 ± 153.894 | 3.9965 ± 0.331 |

## Detailed Statistics (per kernel/mode)

| Kernel | Mode | Metric | Mean | ±95% CI | Std |
|--------|------|--------|------|---------|-----|
| chebyshev | A_only         | raw_nll         | 5.3458 | ±0.2712 | 0.4375 |
| chebyshev | A_only         | raw_ppl         | 228.8546 | ±63.3725 | 102.2456 |
| chebyshev | A_only         | weighted_nll    | 3.4346 | ±0.1867 | 0.3012 |
| chebyshev | A_only         | weighted_ppl    | 32.3204 | ±6.0950 | 9.8337 |
| chebyshev | BC_only        | raw_nll         | 4.5877 | ±0.1972 | 0.3181 |
| chebyshev | BC_only        | raw_ppl         | 102.8428 | ±20.1936 | 32.5805 |
| chebyshev | BC_only        | weighted_nll    | 3.0580 | ±0.1836 | 0.2962 |
| chebyshev | BC_only        | weighted_ppl    | 22.1444 | ±4.0840 | 6.5892 |
| chebyshev | D_only         | raw_nll         | 5.8352 | ±0.2616 | 0.4220 |
| chebyshev | D_only         | raw_ppl         | 370.8523 | ±97.9325 | 158.0051 |
| chebyshev | D_only         | weighted_nll    | 3.8960 | ±0.2276 | 0.3672 |
| chebyshev | D_only         | weighted_ppl    | 52.2861 | ±11.9104 | 19.2164 |
| chebyshev | all_except_A   | raw_nll         | 4.2199 | ±0.1854 | 0.2991 |
| chebyshev | all_except_A   | raw_ppl         | 70.7997 | ±12.9747 | 20.9335 |
| chebyshev | all_except_A   | weighted_nll    | 2.7808 | ±0.1644 | 0.2653 |
| chebyshev | all_except_A   | weighted_ppl    | 16.6531 | ±2.7491 | 4.4354 |
| chebyshev | all_except_dt  | raw_nll         | 3.8085 | ±0.1620 | 0.2613 |
| chebyshev | all_except_dt  | raw_ppl         | 46.4961 | ±7.5834 | 12.2352 |
| chebyshev | all_except_dt  | weighted_nll    | 2.5559 | ±0.1546 | 0.2494 |
| chebyshev | all_except_dt  | weighted_ppl    | 13.2504 | ±2.0607 | 3.3247 |
| chebyshev | dt_only        | raw_nll         | 5.8522 | ±0.2620 | 0.4227 |
| chebyshev | dt_only        | raw_ppl         | 377.2061 | ±99.0286 | 159.7735 |
| chebyshev | dt_only        | weighted_nll    | 3.9174 | ±0.2291 | 0.3696 |
| chebyshev | dt_only        | weighted_ppl    | 53.4661 | ±12.2768 | 19.8076 |
| chebyshev | full           | raw_nll         | 3.7770 | ±0.1591 | 0.2567 |
| chebyshev | full           | raw_ppl         | 45.0050 | ±7.2108 | 11.6340 |
| chebyshev | full           | weighted_nll    | 2.5465 | ±0.1547 | 0.2496 |
| chebyshev | full           | weighted_ppl    | 13.1270 | ±2.0430 | 3.2961 |
| chebyshev | none           | raw_nll         | 5.9481 | ±0.2548 | 0.4112 |
| chebyshev | none           | raw_ppl         | 413.3920 | ±105.7983 | 170.6957 |
| chebyshev | none           | weighted_nll    | 4.0026 | ±0.2366 | 0.3818 |
| chebyshev | none           | weighted_ppl    | 58.4640 | ±13.8872 | 22.4057 |
|       dct | A_only         | raw_nll         | 5.0917 | ±0.2372 | 0.3826 |
|       dct | A_only         | raw_ppl         | 173.5442 | ±39.8884 | 64.3562 |
|       dct | A_only         | weighted_nll    | 3.2424 | ±0.1486 | 0.2397 |
|       dct | A_only         | weighted_ppl    | 26.2726 | ±3.9543 | 6.3798 |
|       dct | BC_only        | raw_nll         | 4.7366 | ±0.2103 | 0.3393 |
|       dct | BC_only        | raw_ppl         | 120.0516 | ±24.8710 | 40.1271 |
|       dct | BC_only        | weighted_nll    | 3.0542 | ±0.1464 | 0.2362 |
|       dct | BC_only        | weighted_ppl    | 21.7474 | ±3.2114 | 5.1814 |
|       dct | D_only         | raw_nll         | 5.7553 | ±0.2346 | 0.3785 |
|       dct | D_only         | raw_ppl         | 336.7832 | ±78.0346 | 125.9016 |
|       dct | D_only         | weighted_nll    | 3.8960 | ±0.2053 | 0.3312 |
|       dct | D_only         | weighted_ppl    | 51.7190 | ±10.7439 | 17.3344 |
|       dct | all_except_A   | raw_nll         | 4.3141 | ±0.1947 | 0.3142 |
|       dct | all_except_A   | raw_ppl         | 78.1390 | ±15.2061 | 24.5337 |
|       dct | all_except_A   | weighted_nll    | 2.8025 | ±0.1405 | 0.2266 |
|       dct | all_except_A   | weighted_ppl    | 16.8730 | ±2.3894 | 3.8551 |
|       dct | all_except_dt  | raw_nll         | 3.7531 | ±0.1497 | 0.2416 |
|       dct | all_except_dt  | raw_ppl         | 43.7888 | ±6.5314 | 10.5378 |
|       dct | all_except_dt  | weighted_nll    | 2.5892 | ±0.1423 | 0.2296 |
|       dct | all_except_dt  | weighted_ppl    | 13.6418 | ±1.9695 | 3.1776 |
|       dct | dt_only        | raw_nll         | 5.9005 | ±0.2438 | 0.3933 |
|       dct | dt_only        | raw_ppl         | 391.2852 | ±93.6004 | 151.0156 |
|       dct | dt_only        | weighted_nll    | 3.9875 | ±0.2089 | 0.3370 |
|       dct | dt_only        | weighted_ppl    | 56.7725 | ±12.0132 | 19.3822 |
|       dct | full           | raw_nll         | 3.7096 | ±0.1511 | 0.2439 |
|       dct | full           | raw_ppl         | 41.9436 | ±6.3074 | 10.1764 |
|       dct | full           | weighted_nll    | 2.5520 | ±0.1384 | 0.2232 |
|       dct | full           | weighted_ppl    | 13.1259 | ±1.8421 | 2.9720 |
|       dct | none           | raw_nll         | 5.9061 | ±0.2480 | 0.4002 |
|       dct | none           | raw_ppl         | 394.3660 | ±95.3846 | 153.8941 |
|       dct | none           | weighted_nll    | 3.9965 | ±0.2051 | 0.3309 |
|       dct | none           | weighted_ppl    | 57.1807 | ±11.8896 | 19.1828 |
|  hypernet | A_only         | raw_nll         | 5.8856 | ±0.2418 | 0.3900 |
|  hypernet | A_only         | raw_ppl         | 384.8417 | ±90.2564 | 145.6204 |
|  hypernet | A_only         | weighted_nll    | 4.0735 | ±0.2548 | 0.4111 |
|  hypernet | A_only         | weighted_ppl    | 63.3135 | ±15.7131 | 25.3517 |
|  hypernet | BC_only        | raw_nll         | 6.0440 | ±0.2619 | 0.4226 |
|  hypernet | BC_only        | raw_ppl         | 456.6219 | ±117.9756 | 190.3427 |
|  hypernet | BC_only        | weighted_nll    | 4.0747 | ±0.2495 | 0.4026 |
|  hypernet | BC_only        | weighted_ppl    | 63.1964 | ±15.3353 | 24.7421 |
|  hypernet | D_only         | raw_nll         | 5.7526 | ±0.2600 | 0.4194 |
|  hypernet | D_only         | raw_ppl         | 341.0452 | ±88.6538 | 143.0346 |
|  hypernet | D_only         | weighted_nll    | 3.7834 | ±0.2245 | 0.3622 |
|  hypernet | D_only         | weighted_ppl    | 46.5930 | ±10.2242 | 16.4957 |
|  hypernet | all_except_A   | raw_nll         | 5.8604 | ±0.2602 | 0.4198 |
|  hypernet | all_except_A   | raw_ppl         | 379.5826 | ±97.1368 | 156.7211 |
|  hypernet | all_except_A   | weighted_nll    | 3.9118 | ±0.2349 | 0.3790 |
|  hypernet | all_except_A   | weighted_ppl    | 53.2564 | ±12.1446 | 19.5942 |
|  hypernet | all_except_dt  | raw_nll         | 5.6262 | ±0.2540 | 0.4098 |
|  hypernet | all_except_dt  | raw_ppl         | 299.5020 | ±76.2836 | 123.0765 |
|  hypernet | all_except_dt  | weighted_nll    | 3.7417 | ±0.2281 | 0.3680 |
|  hypernet | all_except_dt  | weighted_ppl    | 44.7728 | ±9.9581 | 16.0665 |
|  hypernet | dt_only        | raw_nll         | 6.0015 | ±0.2524 | 0.4072 |
|  hypernet | dt_only        | raw_ppl         | 435.2180 | ±108.9410 | 175.7662 |
|  hypernet | dt_only        | weighted_nll    | 4.0421 | ±0.2529 | 0.4081 |
|  hypernet | dt_only        | weighted_ppl    | 61.2869 | ±15.1067 | 24.3733 |
|  hypernet | full           | raw_nll         | 5.6227 | ±0.2472 | 0.3988 |
|  hypernet | full           | raw_ppl         | 297.2076 | ±73.3673 | 118.3713 |
|  hypernet | full           | weighted_nll    | 3.7333 | ±0.2275 | 0.3671 |
|  hypernet | full           | weighted_ppl    | 44.3824 | ±9.8421 | 15.8794 |
|  hypernet | none           | raw_nll         | 5.9380 | ±0.2481 | 0.4003 |
|  hypernet | none           | raw_ppl         | 407.3940 | ±99.9678 | 161.2887 |
|  hypernet | none           | weighted_nll    | 3.9957 | ±0.2478 | 0.3998 |
|  hypernet | none           | weighted_ppl    | 58.3424 | ±14.1286 | 22.7952 |

## Analysis: A-Modulation Contribution (Reviewer Axfu Q2)

Comparing `full` (all 5 params modulated) vs `all_except_A` (A frozen, B/C/D/Δ modulated). The gap isolates how much the recurrence parameter A contributes beyond Mamba-style selection (which already modulates Δ, B, and C).

| Kernel | Full W-PPL | All-except-A W-PPL | Δ W-PPL | A Contribution |
|--------|-----------|-------------------|---------|---------------|
| chebyshev | 13.1270 ± 3.296 |   16.6531 ± 4.435 |  3.5261 | +3.5261 |
|       dct | 13.1259 ± 2.972 |   16.8730 ± 3.855 |  3.7471 | +3.7471 |
|  hypernet | 44.3824 ± 15.879 |  53.2564 ± 19.594 |  8.8740 | +8.8740 |

## Analysis: Mamba-Style Selection

If `BC_only` performs close to `full`, Mamba-style selection through B and C already captures much of MaRK's benefit. If `A_only` is close to `full`, then A-modulation is the key contribution.

| Kernel | Full W-PPL | BC_only W-PPL | A_only W-PPL | dt_only W-PPL |
|--------|-----------|--------------|-------------|--------------|
| chebyshev |   13.1270 |      22.1444 |     32.3204 |      53.4661 |
|       dct |   13.1259 |      21.7474 |     26.2726 |      56.7725 |
|  hypernet |   44.3824 |      63.1964 |     63.3135 |      61.2869 |

## Full Results (All Metrics)

| Kernel | Mode | Raw NLL | Raw PPL | Weighted NLL | Weighted PPL |
|--------|------|---------|---------|-------------|-------------|
|  hypernet | full           | 5.6227 ± 0.399 | 297.2076 ± 118.371 | 3.7333 ± 0.367 | 44.3824 ± 15.879 |
|  hypernet | all_except_A   | 5.8604 ± 0.420 | 379.5826 ± 156.721 | 3.9118 ± 0.379 | 53.2564 ± 19.594 |
|  hypernet | A_only         | 5.8856 ± 0.390 | 384.8417 ± 145.620 | 4.0735 ± 0.411 | 63.3135 ± 25.352 |
|  hypernet | dt_only        | 6.0015 ± 0.407 | 435.2180 ± 175.766 | 4.0421 ± 0.408 | 61.2869 ± 24.373 |
|  hypernet | BC_only        | 6.0440 ± 0.423 | 456.6219 ± 190.343 | 4.0747 ± 0.403 | 63.1964 ± 24.742 |
|  hypernet | all_except_dt  | 5.6262 ± 0.410 | 299.5020 ± 123.077 | 3.7417 ± 0.368 | 44.7728 ± 16.066 |
|  hypernet | D_only         | 5.7526 ± 0.419 | 341.0452 ± 143.035 | 3.7834 ± 0.362 | 46.5930 ± 16.496 |
|  hypernet | none           | 5.9380 ± 0.400 | 407.3940 ± 161.289 | 3.9957 ± 0.400 | 58.3424 ± 22.795 |
| chebyshev | full           | 3.7770 ± 0.257 | 45.0050 ± 11.634 | 2.5465 ± 0.250 | 13.1270 ± 3.296 |
| chebyshev | all_except_A   | 4.2199 ± 0.299 | 70.7997 ± 20.934 | 2.7808 ± 0.265 | 16.6531 ± 4.435 |
| chebyshev | A_only         | 5.3458 ± 0.438 | 228.8546 ± 102.246 | 3.4346 ± 0.301 | 32.3204 ± 9.834 |
| chebyshev | dt_only        | 5.8522 ± 0.423 | 377.2061 ± 159.773 | 3.9174 ± 0.370 | 53.4661 ± 19.808 |
| chebyshev | BC_only        | 4.5877 ± 0.318 | 102.8428 ± 32.581 | 3.0580 ± 0.296 | 22.1444 ± 6.589 |
| chebyshev | all_except_dt  | 3.8085 ± 0.261 | 46.4961 ± 12.235 | 2.5559 ± 0.249 | 13.2504 ± 3.325 |
| chebyshev | D_only         | 5.8352 ± 0.422 | 370.8523 ± 158.005 | 3.8960 ± 0.367 | 52.2861 ± 19.216 |
| chebyshev | none           | 5.9481 ± 0.411 | 413.3920 ± 170.696 | 4.0026 ± 0.382 | 58.4640 ± 22.406 |
|       dct | full           | 3.7096 ± 0.244 | 41.9436 ± 10.176 | 2.5520 ± 0.223 | 13.1259 ± 2.972 |
|       dct | all_except_A   | 4.3141 ± 0.314 | 78.1390 ± 24.534 | 2.8025 ± 0.227 | 16.8730 ± 3.855 |
|       dct | A_only         | 5.0917 ± 0.383 | 173.5442 ± 64.356 | 3.2424 ± 0.240 | 26.2726 ± 6.380 |
|       dct | dt_only        | 5.9005 ± 0.393 | 391.2852 ± 151.016 | 3.9875 ± 0.337 | 56.7725 ± 19.382 |
|       dct | BC_only        | 4.7366 ± 0.339 | 120.0516 ± 40.127 | 3.0542 ± 0.236 | 21.7474 ± 5.181 |
|       dct | all_except_dt  | 3.7531 ± 0.242 | 43.7888 ± 10.538 | 2.5892 ± 0.230 | 13.6418 ± 3.178 |
|       dct | D_only         | 5.7553 ± 0.379 | 336.7832 ± 125.902 | 3.8960 ± 0.331 | 51.7190 ± 17.334 |
|       dct | none           | 5.9061 ± 0.400 | 394.3660 ± 153.894 | 3.9965 ± 0.331 | 57.1807 ± 19.183 |

