<!-- Overhead vs Frozen-Hydra (frozen backbone) — batch=8 -->

| Variant | Latency +% | Memory +% | Throughput −% | +Adapter (M) |
|---------|----------:|---------:|-------------:|-------------:|
| Frozen-Hydra | 0.0 | 0.0 | 0.0 | 0.00 |
| Hypernet | +65.6 | +2.6 | 39.6 | 6.27 |
| Chebyshev | +104.0 | +4.5 | 51.0 | 10.97 |
| DCT | +88.7 | +2.9 | 47.0 | 7.07 |
| AdaLN-Zero | +7.6 | +3.0 | 7.1 | 6.84 |
| Input-Injection | +3.6 | +0.9 | 3.4 | 2.28 |

*Overhead computed as (variant / Frozen-Hydra) − 1. Measured on NVIDIA A100-SXM4-80GB (15 seeds, 95% CI).*
