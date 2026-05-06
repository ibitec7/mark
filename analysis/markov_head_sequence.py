"""Head-resolved Markov coefficients for the paper dynamics figure.

The coefficient convention is
    m_{l,h}(k) = <B'_{l,h}, C'_{l,h}> * |lambda_{l,h}|**k
under the same diagonal-per-head abstraction used in `dynamics_analysis.py`.
The B/C state dimension is split across heads with `numpy.array_split`; this is
a geometric partition for analysis, not a Triton memory-layout assumption.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def head_index_splits(d_state: int, n_heads: int) -> list[slice]:
    """Partition [0, d_state) into n_heads contiguous index ranges (unequal if d_state % n_heads != 0)."""
    chunks = np.array_split(np.arange(d_state, dtype=np.int64), n_heads)
    return [slice(int(c[0]), int(c[-1]) + 1) for c in chunks]


def frozen_lambda_per_head(sweep_kernel: dict[str, Any]) -> np.ndarray:
    """|λ| per (layer, head) with frozen base dynamics (no MaRK on A, dt). Shape (L, H)."""
    base_A_log = sweep_kernel["base_A_log"]
    base_dt = sweep_kernel["base_dt"]
    A_cont = -np.exp(base_A_log)
    delta = np.log1p(np.exp(base_dt))
    A_frozen = np.exp(A_cont * delta)
    return np.abs(A_frozen)


def compute_markov_head_sequence(
    B_stack: np.ndarray,
    C_stack: np.ndarray,
    sweep_kernel: dict[str, Any],
    k_max: int,
) -> np.ndarray:
    """
    Per diffusion index τ, layer ℓ, head h, lag k:
        m[τ,ℓ,h,k] = dot(B'_{ℓ,h}, C'_{ℓ,h}) * λ_{ℓ,h,τ}^k
    with modulated B', C' from the sweep and λ = |A_disc|.

    Returns:
        m: float64 array of shape (T, L, H, k_max)
    """
    A_disc = np.abs(sweep_kernel["A_disc"])
    T, L, H = A_disc.shape
    B_scale = sweep_kernel["B_scale"]
    B_shift = sweep_kernel["B_shift"]
    C_scale = sweep_kernel["C_scale"]
    C_shift = sweep_kernel["C_shift"]

    d_state = B_stack.shape[1]
    splits = head_index_splits(d_state, H)
    m = np.zeros((T, L, H, k_max), dtype=np.float64)
    k_axis = np.arange(k_max, dtype=np.float64)

    for tau in range(T):
        Bp = B_stack * B_scale[tau] + B_shift[tau]
        Cp = C_stack * C_scale[tau] + C_shift[tau]
        for l in range(L):
            for h in range(H):
                sl = splits[h]
                dh = float(np.dot(Bp[l, sl], Cp[l, sl]))
                lam = float(A_disc[tau, l, h])
                m[tau, l, h, :] = dh * (lam**k_axis)
    return m


def compute_frozen_markov_head_sequence(
    B_stack: np.ndarray,
    C_stack: np.ndarray,
    sweep_kernel: dict[str, Any],
    k_max: int,
) -> np.ndarray:
    """
    Same as compute_markov_head_sequence but unmodulated B,C and λ from frozen recurrence.
    Returns shape (L, H, k_max).
    """
    lam_f = frozen_lambda_per_head(sweep_kernel)
    L, H = lam_f.shape
    d_state = B_stack.shape[1]
    splits = head_index_splits(d_state, H)
    m = np.zeros((L, H, k_max), dtype=np.float64)
    k_axis = np.arange(k_max, dtype=np.float64)

    for l in range(L):
        for h in range(H):
            sl = splits[h]
            dh = float(np.dot(B_stack[l, sl], C_stack[l, sl]))
            lam = float(lam_f[l, h])
            m[l, h, :] = dh * (lam**k_axis)
    return m


def network_norm_vs_k(m_tau: np.ndarray) -> np.ndarray:
    """
    For one τ, R[k] = sqrt( sum_{l,h} m[l,h,k]^2 ). m_tau shape (L, H, k_max).
    """
    s = np.sum(m_tau**2, axis=(0, 1))
    return np.sqrt(np.maximum(s, 0.0))
