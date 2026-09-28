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
