<!-- Overhead vs Frozen-Hydra (frozen backbone) — batch=4 -->

| Variant | Latency +% | Memory +% | Throughput −% | +Adapter (M) |
|---------|----------:|---------:|-------------:|-------------:|
| Frozen-Hydra | 0.0 | 0.0 | 0.0 | 0.00 |
| Hypernet | +70.8 | +3.5 | 41.5 | 6.27 |
| Chebyshev | +112.3 | +6.1 | 52.9 | 10.97 |
| DCT | +96.1 | +4.0 | 49.0 | 7.07 |
| AdaLN-Zero | +9.4 | +4.1 | 8.6 | 6.84 |
| Input-Injection | +4.2 | +1.3 | 4.1 | 2.28 |

*Overhead computed as (variant / Frozen-Hydra) − 1. Measured on NVIDIA A100-SXM4-80GB (15 seeds, 95% CI).*
