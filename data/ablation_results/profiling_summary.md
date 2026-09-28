# MaRK Profiling Ablation — AdaLN-Zero vs Input-Injection vs MaRK Kernels

> Source: `/workspace/experiment` (`profiling.profile_multiseed`)
> Seeds: 15 | Latency iterations/seed: 100 | Warmup: 20
> Mode: both | Batch sizes: [1, 4, 8] | Seq len: 512
> Artifacts: `data/ablation_results/profiling/a100_neurips_512`

## adapter_table.md

<!-- Per-adapter block profiling — copy into Table X of the appendix -->

| Adapter | Latency (ms) | Std (ms) | Peak Alloc (MiB) | Peak Reserv (MiB) | Params (M) | Trainable (M) |
|---------|-------------:|---------:|-----------------:|------------------:|-----------:|--------------:|
| Frozen-Hydra | 0.000 | 0.0000 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| Hypernet | 0.734 | 0.0195 | 10.18 | 24.00 | 0.2726 | 0.0000 |
| Chebyshev | 1.305 | 0.0505 | 10.96 | 25.60 | 0.4771 | 0.0000 |
| DCT | 1.068 | 0.0577 | 10.32 | 24.00 | 0.3075 | 0.0000 |
| AdaLN-Zero | 0.159 | 0.0060 | 15.27 | 22.00 | 0.2972 | 0.0000 |
| Input-Injection | 0.063 | 0.0037 | 11.51 | 22.00 | 0.0991 | 0.0000 |

*15 seeds per adapter, 100 latency iterations per seed. Measured on NVIDIA A100-SXM4-80GB.*

## e2e_table_bs1.md

<!-- Inference profiling: batch=1, seq_len=512 -->

| Variant | Latency (ms) | 95% CI | Alloc (MiB) | 95% CI | Throughput (samples/s) | 95% CI | Params (M) | Adapter (M) |
|---------|-------------:|-------:|------------:|-------:|----------------------:|-------:|-----------:|------------:|
| Frozen-Hydra | 43.80 | ±0.07 | 500.97 | ±0.03 | 22.83 | ±0.03 | 111.68 | 0.00 |
| Hypernet | 73.86 | ±0.09 | 524.97 | ±0.05 | 13.54 | ±0.02 | 117.95 | 6.27 |
| Chebyshev | 91.74 | ±0.18 | 543.02 | ±0.07 | 10.90 | ±0.02 | 122.65 | 10.97 |
| DCT | 84.44 | ±0.11 | 528.38 | ±0.06 | 11.84 | ±0.02 | 118.75 | 7.07 |
| AdaLN-Zero | 47.81 | ±0.25 | 528.62 | ±0.15 | 20.92 | ±0.11 | 118.51 | 6.84 |
| Input-Injection | 45.32 | ±0.11 | 509.89 | ±0.05 | 22.07 | ±0.05 | 113.96 | 2.28 |

*15 independent seeds, 100 latency iterations per seed (+ 20 warmup). Mean ± 95% CI (t-distribution, 14 df). Measured on NVIDIA A100-SXM4-80GB.*

## e2e_table_bs4.md

<!-- Inference profiling: batch=4, seq_len=512 -->

| Variant | Latency (ms) | 95% CI | Alloc (MiB) | 95% CI | Throughput (samples/s) | 95% CI | Params (M) | Adapter (M) |
|---------|-------------:|-------:|------------:|-------:|----------------------:|-------:|-----------:|------------:|
| Frozen-Hydra | 44.71 | ±0.08 | 688.38 | ±0.11 | 89.46 | ±0.15 | 111.68 | 0.00 |
| Hypernet | 76.37 | ±0.18 | 712.47 | ±0.11 | 52.38 | ±0.12 | 117.95 | 6.27 |
| Chebyshev | 94.93 | ±0.15 | 730.42 | ±0.11 | 42.14 | ±0.07 | 122.65 | 10.97 |
| DCT | 87.70 | ±0.23 | 715.78 | ±0.11 | 45.61 | ±0.12 | 118.75 | 7.07 |
| AdaLN-Zero | 48.92 | ±0.11 | 716.45 | ±0.48 | 81.77 | ±0.19 | 118.51 | 6.84 |
| Input-Injection | 46.60 | ±0.11 | 697.10 | ±0.11 | 85.83 | ±0.20 | 113.96 | 2.28 |

*15 independent seeds, 100 latency iterations per seed (+ 20 warmup). Mean ± 95% CI (t-distribution, 14 df). Measured on NVIDIA A100-SXM4-80GB.*

## e2e_table_bs8.md

<!-- Inference profiling: batch=8, seq_len=512 -->

| Variant | Latency (ms) | 95% CI | Alloc (MiB) | 95% CI | Throughput (samples/s) | 95% CI | Params (M) | Adapter (M) |
|---------|-------------:|-------:|------------:|-------:|----------------------:|-------:|-----------:|------------:|
| Frozen-Hydra | 47.75 | ±0.09 | 938.90 | ±0.13 | 167.55 | ±0.32 | 111.68 | 0.00 |
| Hypernet | 79.08 | ±0.12 | 962.98 | ±0.13 | 101.16 | ±0.16 | 117.95 | 6.27 |
| Chebyshev | 97.39 | ±0.14 | 980.93 | ±0.13 | 82.14 | ±0.12 | 122.65 | 10.97 |
| DCT | 90.08 | ±0.11 | 966.30 | ±0.13 | 88.81 | ±0.11 | 118.75 | 7.07 |
| AdaLN-Zero | 51.38 | ±0.09 | 966.96 | ±0.46 | 155.70 | ±0.27 | 118.51 | 6.84 |
| Input-Injection | 49.45 | ±0.05 | 947.62 | ±0.13 | 161.79 | ±0.15 | 113.96 | 2.28 |

*15 independent seeds, 100 latency iterations per seed (+ 20 warmup). Mean ± 95% CI (t-distribution, 14 df). Measured on NVIDIA A100-SXM4-80GB.*

