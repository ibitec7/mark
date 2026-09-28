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
