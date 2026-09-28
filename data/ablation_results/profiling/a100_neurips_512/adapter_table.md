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
