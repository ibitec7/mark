<!-- Overhead vs Frozen-Hydra (frozen backbone) — batch=1 -->

| Variant | Latency +% | Memory +% | Throughput −% | +Adapter (M) |
|---------|----------:|---------:|-------------:|-------------:|
| Frozen-Hydra | 0.0 | 0.0 | 0.0 | 0.00 |
| Hypernet | +68.6 | +4.8 | 40.7 | 6.27 |
| Chebyshev | +109.4 | +8.4 | 52.3 | 10.97 |
| DCT | +92.8 | +5.5 | 48.1 | 7.07 |
| AdaLN-Zero | +9.2 | +5.5 | 8.4 | 6.84 |
| Input-Injection | +3.5 | +1.8 | 3.3 | 2.28 |

*Overhead computed as (variant / Frozen-Hydra) − 1. Measured on NVIDIA A100-SXM4-80GB (15 seeds, 95% CI).*
