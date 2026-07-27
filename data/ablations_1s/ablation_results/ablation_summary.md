# MaRK SSM Parameter Ablation — Validation Loss on WikiText

> Generated: 2026-07-27 05:31:23

## Results: All Kernels × All Ablation Modes

| Kernel | Mode | A | B | C | dt | D | Weighted PPL ↓ | Raw PPL ↓ | Weighted NLL ↓ |
|--------|------|---|---|---|----|---|---------------|-----------|----------------|
|  hypernet | full           | ✓ | ✓ | ✓ | ✓ | ✓ |       43.4079 |  300.7303 |         3.7706 |
|  hypernet | all_except_A   | ✗ | ✓ | ✓ | ✓ | ✓ |       70.0998 |  258.8043 |         4.2499 |
|  hypernet | A_only         | ✓ | ✗ | ✗ | ✗ | ✗ |       50.3840 |  413.3946 |         3.9197 |
|  hypernet | dt_only        | ✗ | ✗ | ✗ | ✓ | ✗ |       42.2725 |  454.0857 |         3.7441 |
|  hypernet | BC_only        | ✗ | ✓ | ✓ | ✗ | ✗ |       49.7470 |  763.5111 |         3.9069 |
|  hypernet | all_except_dt  | ✓ | ✓ | ✓ | ✗ | ✓ |       54.8039 |  317.5383 |         4.0038 |
|  hypernet | D_only         | ✗ | ✗ | ✗ | ✗ | ✓ |       55.5601 |  294.1071 |         4.0175 |
|  hypernet | none           | ✗ | ✗ | ✗ | ✗ | ✗ |       57.7564 |  307.2272 |         4.0562 |
| chebyshev | full           | ✓ | ✓ | ✓ | ✓ | ✓ |       10.4748 |   59.7916 |         2.3490 |
| chebyshev | all_except_A   | ✗ | ✓ | ✓ | ✓ | ✓ |       15.9085 |   79.4965 |         2.7669 |
| chebyshev | A_only         | ✓ | ✗ | ✗ | ✗ | ✗ |       29.5789 |  149.0750 |         3.3871 |
| chebyshev | dt_only        | ✗ | ✗ | ✗ | ✓ | ✗ |       40.3394 |  486.8895 |         3.6973 |
| chebyshev | BC_only        | ✗ | ✓ | ✓ | ✗ | ✗ |       39.9448 |   52.7435 |         3.6875 |
| chebyshev | all_except_dt  | ✓ | ✓ | ✓ | ✗ | ✓ |       14.6511 |   41.8654 |         2.6845 |
| chebyshev | D_only         | ✗ | ✗ | ✗ | ✗ | ✓ |       80.4337 |  234.9127 |         4.3874 |
| chebyshev | none           | ✗ | ✗ | ✗ | ✗ | ✗ |       36.6007 |  686.2868 |         3.6001 |
|       dct | full           | ✓ | ✓ | ✓ | ✓ | ✓ |       17.3016 |   32.0454 |         2.8508 |
|       dct | all_except_A   | ✗ | ✓ | ✓ | ✓ | ✓ |       18.8523 |   56.4827 |         2.9366 |
|       dct | A_only         | ✓ | ✗ | ✗ | ✗ | ✗ |       32.4890 |   96.3118 |         3.4809 |
|       dct | dt_only        | ✗ | ✗ | ✗ | ✓ | ✗ |       56.6487 |  336.5798 |         4.0369 |
|       dct | BC_only        | ✗ | ✓ | ✓ | ✗ | ✗ |       22.4113 |   96.8010 |         3.1096 |
|       dct | all_except_dt  | ✓ | ✓ | ✓ | ✗ | ✓ |       14.6323 |   37.6985 |         2.6832 |
|       dct | D_only         | ✗ | ✗ | ✗ | ✗ | ✓ |      107.8219 |  219.6458 |         4.6805 |
|       dct | none           | ✗ | ✗ | ✗ | ✗ | ✗ |       53.9311 |  401.1647 |         3.9877 |

## Analysis: A-Modulation Contribution (Reviewer Axfu Q2)

Comparing `full` (all 5 params modulated) vs `all_except_A` (A frozen, B/C/D/Δ modulated). The gap isolates how much the recurrence parameter A contributes beyond Mamba-style selection (which already modulates Δ, B, and C).

| Kernel | Full W-PPL | All-except-A W-PPL | Δ W-PPL | A Contribution |
|--------|-----------|-------------------|---------|---------------|
| chebyshev |   10.4748 |           15.9085 |  5.4337 | +5.4337 |
|       dct |   17.3016 |           18.8523 |  1.5507 | +1.5507 |
|  hypernet |   43.4079 |           70.0998 | 26.6918 | +26.6918 |

## Analysis: Mamba-Style Selection (BC_only vs full)

If `BC_only` (B and C modulated, A/Δ/D frozen) performs close to `full`, then Mamba-style selection through B and C already captures much of MaRK's benefit. If `A_only` is close to `full`, then A-modulation is the key contribution.

| Kernel | Full W-PPL | BC_only W-PPL | A_only W-PPL | dt_only W-PPL |
|--------|-----------|--------------|-------------|--------------|
| chebyshev |   10.4748 |      39.9448 |     29.5789 |      40.3394 |
|       dct |   17.3016 |      22.4113 |     32.4890 |      56.6487 |
|  hypernet |   43.4079 |      49.7470 |     50.3840 |      42.2725 |

## Full Results (All Metrics)

| Kernel | Mode | Raw NLL | Raw PPL | Weighted NLL | Weighted PPL | MDLM PPL |
|--------|------|---------|---------|-------------|-------------|----------|
|  hypernet | full           |  5.7062 | 300.7303 |      3.7706 |     43.4079 | 50001939381.7935 |
|  hypernet | all_except_A   |  5.5561 | 258.8043 |      4.2499 |     70.0998 | 107894091781.5679 |
|  hypernet | A_only         |  6.0244 | 413.3946 |      3.9197 |     50.3840 | 25644790589.3336 |
|  hypernet | dt_only        |  6.1183 | 454.0857 |      3.7441 |     42.2725 | 220768649053.7494 |
|  hypernet | BC_only        |  6.6379 | 763.5111 |      3.9069 |     49.7470 | 8892059911.9041 |
|  hypernet | all_except_dt  |  5.7606 | 317.5383 |      4.0038 |     54.8039 | 3494355727.9782 |
|  hypernet | D_only         |  5.6839 | 294.1071 |      4.0175 |     55.5601 | 14756569493.5906 |
|  hypernet | none           |  5.7276 | 307.2272 |      4.0562 |     57.7564 | 85347042961.4548 |
| chebyshev | full           |  4.0909 | 59.7916 |      2.3490 |     10.4748 | 2046850.8779 |
| chebyshev | all_except_A   |  4.3757 | 79.4965 |      2.7669 |     15.9085 | 25556419.0284 |
| chebyshev | A_only         |  5.0044 | 149.0750 |      3.3871 |     29.5789 | 3989267465.2700 |
| chebyshev | dt_only        |  6.1880 | 486.8895 |      3.6973 |     40.3394 | 6604460425.5887 |
| chebyshev | BC_only        |  3.9654 | 52.7435 |      3.6875 |     39.9448 | 24350896554.3065 |
| chebyshev | all_except_dt  |  3.7345 | 41.8654 |      2.6845 |     14.6511 | 12834164.1584 |
| chebyshev | D_only         |  5.4592 | 234.9127 |      4.3874 |     80.4337 | 66591124565.7541 |
| chebyshev | none           |  6.5313 | 686.2868 |      3.6001 |     36.6007 | 18803876184.1784 |
|       dct | full           |  3.4672 | 32.0454 |      2.8508 |     17.3016 | 45990754.3324 |
|       dct | all_except_A   |  4.0339 | 56.4827 |      2.9366 |     18.8523 | 140152904.5256 |
|       dct | A_only         |  4.5676 | 96.3118 |      3.4809 |     32.4890 | 5458683343.2512 |
|       dct | dt_only        |  5.8188 | 336.5798 |      4.0369 |     56.6487 | 50175280124.8931 |
|       dct | BC_only        |  4.5727 | 96.8010 |      3.1096 |     22.4113 | 256181253.4344 |
|       dct | all_except_dt  |  3.6296 | 37.6985 |      2.6832 |     14.6323 | 11736792.7503 |
|       dct | D_only         |  5.3920 | 219.6458 |      4.6805 |    107.8219 | 153120111407.9435 |
|       dct | none           |  5.9944 | 401.1647 |      3.9877 |     53.9311 | 48484028232.0960 |

