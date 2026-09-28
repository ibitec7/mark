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
