"""Generate the AQS stability certificate figure and audit tables.

Paper artifact:
    plots/aqs_certificate.png

Run from the repository root:
    python -m analysis.aqs_certificate

The certificate uses the Hydra base recurrence parameters and the shared MaRK
modulation bound to verify a common P=I Lyapunov certificate layer by layer.
"""

# ==================================================================================================================================== #
# AQS Certificate: Constructive Stability Verification for MaRK-Modulated Hydra SSM
#
# This script provides a rigorous Affine Quadratic Stability (AQS) certificate for the MaRK
# adapter framework by exploiting the constructive parameterization chain:
#
#   A_log' = A_log + beta * tanh(psi(c_t))    (bounded shift, |shift| < 0.5)
#   A_continuous = -exp(A_log')                (strictly negative)
#   dt = softplus(dt_input + dt_bias)          (strictly positive)
#   A_discrete = exp(A_continuous * dt)        (element-wise in (0, 1))
#
# For diagonal A with spectral radius rho < 1, the identity matrix P = I is a valid common
# Lyapunov function. The Lyapunov decrease condition A^T P A - P = diag(a_i^2) - I is
# negative definite with margin epsilon = 1 - max(a_i^2).
#
# The certificate is kernel-independent: all three MaRK variants (Hypernet, Chebyshev, DCT)
# share max_A_beta = 0.5, so the hyper-rectangle of modulations is identical.
# ==================================================================================================================================== #

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path
try:
    from .utils import setup_logger
except ImportError:  # Allows direct execution as `python analysis/aqs_certificate.py`.
    from utils import setup_logger

Path("logs").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
logger = setup_logger("logs/aqs_certificate.log", write_console=True)

# ── Constants from MaRK architecture ──────────────────────────────────────────
MAX_A_BETA = 0.5          # max modulation bound (mark.py: self.max_A_beta = 0.5)
DT_MIN = 0.001            # minimum discretization step (hydra.py: dt_min default)
NUM_LAYERS = 23            # Hydra BERT 111M layers
WEIGHT_PATH = "models/hydra_bert_23layers.pt"
A_LOG_KEY = "model.bert.encoder.layer.{}.layer.mixer.A_log"
DT_BIAS_KEY = "model.bert.encoder.layer.{}.layer.mixer.dt_bias"
RESULTS_DIR = Path("analysis/results/aqs_certificate")


def load_pretrained_parameters(weight_path: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Load pretrained A_log and dt_bias parameters from the Hydra checkpoint.

    Args:
        weight_path (str): Path to the pretrained .pt checkpoint file.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: Per-layer A_log and dt_bias tensors.
    """
    logger.info(f"Loading pretrained weights from {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    # Navigate nested checkpoint structure: {state: {model: {key: tensor}}}
    if "state" in checkpoint:
        weights = checkpoint["state"]
        if isinstance(weights, dict) and "model" in weights:
            weights = weights["model"]
    else:
        weights = checkpoint

    a_logs, dt_biases = [], []
    for i in range(NUM_LAYERS):
        a_key = A_LOG_KEY.format(i)
        dt_key = DT_BIAS_KEY.format(i)

        assert a_key in weights, f"Key not found: {a_key}"
        a_logs.append(weights[a_key].float())

        if dt_key in weights:
            dt_biases.append(weights[dt_key].float())
        else:
            # dt_bias may be absent in some checkpoints; use zeros as fallback
            logger.warning(f"dt_bias not found for layer {i}, using zeros")
            dt_biases.append(torch.zeros_like(weights[a_key]).float())

    logger.info(f"Loaded parameters for {len(a_logs)} layers, head dim = {a_logs[0].shape[0]}")
    return a_logs, dt_biases


def compute_discrete_A(A_log: torch.Tensor, shift: float, dt_min: float) -> torch.Tensor:
    """
    Compute the discrete-time state transition eigenvalues at a given modulation vertex.

    The ZOH discretization chain is:
        A_log' = A_log + shift
        A_continuous = -exp(A_log')       (strictly negative)
        A_discrete = exp(A_continuous * dt_min)   (element-wise in (0, 1))

    We use dt_min (the smallest possible discretization step) as the worst case because
    smaller dt produces less decay, pushing A_discrete closer to 1.

    Args:
        A_log (torch.Tensor): Base log-space recurrence parameter (per head).
        shift (float): Additive modulation vertex value (e.g., -0.5 or +0.5).
        dt_min (float): Minimum discretization step size.

    Returns:
        torch.Tensor: Discrete eigenvalues A_bar, element-wise in (0, 1).
    """
    A_log_prime = A_log + shift
    A_continuous = -torch.exp(A_log_prime)
    A_discrete = torch.exp(A_continuous * dt_min)
    return A_discrete


def check_spectral_radius_at_vertices(
    a_logs: list[torch.Tensor],
    dt_min: float = DT_MIN,
    beta: float = MAX_A_BETA,
) -> dict:
    """
    Compute the spectral radius at both vertices of the MaRK hyper-rectangle for each layer.

    Because A is diagonal and A_discrete(v) is monotonically decreasing in v, the worst-case
    spectral radius (closest to 1) occurs at v = -beta (the lower bound).

    Args:
        a_logs (list[torch.Tensor]): Per-layer pretrained A_log parameters.
        dt_min (float): Minimum discretization step (worst case for stability).
        beta (float): Maximum modulation bound (0.5 for all MaRK kernels).

    Returns:
        dict: Per-layer spectral radii at both vertices and epsilon margins.
    """
    results = {"layers": [], "overall_stable": True}

    for layer_idx, A_log in enumerate(a_logs):
        # Worst-case vertex: shift = -beta (least decay, A_discrete closest to 1)
        A_disc_worst = compute_discrete_A(A_log, shift=-beta, dt_min=dt_min)
        rho_worst = torch.max(torch.abs(A_disc_worst)).item()

        # Best-case vertex: shift = +beta (most decay, A_discrete closest to 0)
        A_disc_best = compute_discrete_A(A_log, shift=+beta, dt_min=dt_min)
        rho_best = torch.max(torch.abs(A_disc_best)).item()

        # Epsilon margin: stability buffer at worst case
        epsilon = 1.0 - rho_worst ** 2
        is_stable = rho_worst < 1.0

        results["layers"].append({
            "layer": layer_idx,
            "rho_worst": rho_worst,
            "rho_best": rho_best,
            "epsilon": epsilon,
            "stable": is_stable,
        })

        if not is_stable:
            results["overall_stable"] = False

        logger.info(
            f"Layer {layer_idx:2d}: ρ_worst = {rho_worst:.6f}, "
            f"ρ_best = {rho_best:.6f}, ε = {epsilon:.6f} {'✓' if is_stable else '✗'}"
        )

    return results


def verify_lyapunov_certificate(
    a_logs: list[torch.Tensor],
    dt_min: float = DT_MIN,
    beta: float = MAX_A_BETA,
) -> dict:
    """
    Verify the P = I Lyapunov certificate by demonstrating energy dissipation.

    For the energy function V(x) = x^T P x = ||x||^2 (with P = I), stability requires:
        V(x_{t+1}) < V(x_t)
    i.e., ||A_bar' x||^2 < ||x||^2 for all nonzero x and all admissible A_bar'.

    For diagonal A_bar' = diag(a_1, ..., a_n), this reduces to:
        a_i^2 < 1 for all i

    The Lyapunov decrease matrix is:
        A^T P A - P = A^T I A - I = diag(a_i^2 - 1)

    This must be negative definite, i.e., all diagonal entries < 0.
    The stability margin epsilon = min_i(1 - a_i^2) > 0.

    Args:
        a_logs (list[torch.Tensor]): Per-layer pretrained A_log parameters.
        dt_min (float): Minimum discretization step.
        beta (float): Maximum modulation bound.

    Returns:
        dict: Certificate results including per-layer epsilon, Lyapunov decrease eigenvalues.
    """
    logger.info("\n" + "=" * 70)
    logger.info("P = I LYAPUNOV CERTIFICATE VERIFICATION")
    logger.info("=" * 70)
    logger.info("Energy function: V(x) = ||x||^2  (i.e., P = I)")
    logger.info("Stability condition: A^T A - I ≺ 0 (negative definite)")
    logger.info(f"Hyper-rectangle: A_log ± {beta} (kernel-independent)")
    logger.info(f"Worst-case dt: {dt_min} (minimum discretization step)")
    logger.info("=" * 70)

    certificate = {
        "P": "I (identity matrix)",
        "layers": [],
        "overall_certified": True,
        "global_epsilon": float("inf"),
    }

    for layer_idx, A_log in enumerate(a_logs):
        # Compute A_discrete at worst-case vertex v = -beta
        A_disc = compute_discrete_A(A_log, shift=-beta, dt_min=dt_min)

        # Lyapunov decrease: diag(a_i^2 - 1) — must all be < 0
        lyapunov_diag = A_disc ** 2 - 1.0

        # All entries must be strictly negative for negative definiteness
        max_lyapunov_entry = torch.max(lyapunov_diag).item()
        is_neg_def = max_lyapunov_entry < 0.0

        # Epsilon margin: smallest |a_i^2 - 1| = min(1 - a_i^2)
        epsilon = torch.min(1.0 - A_disc ** 2).item()

        # Spectral radius
        rho = torch.max(torch.abs(A_disc)).item()

        certificate["layers"].append({
            "layer": layer_idx,
            "negative_definite": is_neg_def,
            "max_lyapunov_entry": max_lyapunov_entry,
            "epsilon": epsilon,
            "rho": rho,
            "n_heads": A_log.shape[0],
        })

        if epsilon < certificate["global_epsilon"]:
            certificate["global_epsilon"] = epsilon

        if not is_neg_def:
            certificate["overall_certified"] = False

        status = "CERTIFIED" if is_neg_def else "FAILED"
        logger.info(
            f"Layer {layer_idx:2d}: A^T A - I max entry = {max_lyapunov_entry:.8f}, "
            f"ε = {epsilon:.6f}, ρ = {rho:.6f}  [{status}]"
        )

    logger.info("-" * 70)
    if certificate["overall_certified"]:
        logger.info(
            f"AQS CERTIFICATE VERIFIED: P = I is a valid common Lyapunov function "
            f"across all {NUM_LAYERS} layers."
        )
        logger.info(
            f"Global stability margin: ε = {certificate['global_epsilon']:.6f}"
        )
        logger.info(
            "The MaRK adapters are robustly stable — the modulated SSM dynamics are "
            "strictly dissipative for all conditioning contexts c_t within the "
            f"hyper-rectangle A_log ± {beta}."
        )
    else:
        logger.warning("AQS CERTIFICATE FAILED: Not all layers satisfy the Lyapunov condition.")

    return certificate


def generate_certificate_plot(
    results: dict,
    certificate: dict,
    output_path: str = "plots/aqs_certificate.png",
) -> None:
    """
    Generate a visualization of the AQS certificate showing per-layer stability margins.

    Args:
        results (dict): Output from check_spectral_radius_at_vertices.
        certificate (dict): Output from verify_lyapunov_certificate.
        output_path (str): Path to save the plot.
    """
    layers = [r["layer"] for r in results["layers"]]
    epsilons = [c["epsilon"] for c in certificate["layers"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        r"AQS Stability Certificate ($P = I$): Per-Layer Dissipative Margins",
        fontsize=14,
        fontweight="bold",
    )

    x = np.arange(len(layers))
    colors = ["#1976d2" if e > 0 else "#d32f2f" for e in epsilons]
    ax.bar(x, epsilons, color=colors, alpha=0.85)
    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel(r"Stability Margin $\varepsilon_\ell = 1 - \rho_\ell^2$", fontsize=12)
    ax.set_xlabel("Hydra Layer Index", fontsize=12)
    ax.set_title(
        r"$\varepsilon_\ell > 0 \;\Rightarrow\; \bar{A}^{\!\top}\! \bar{A} - I \prec 0$ (energy strictly decreasing)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Certificate plot saved to {output_path}")
    plt.close(fig)


def export_certificate_data(results: dict, certificate: dict) -> None:
    """
    Save the per-layer AQS certificate data used by generate_certificate_plot.

    CSV rows are plot-ready while the JSON summary preserves run-level metadata.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spectral_by_layer = {r["layer"]: r for r in results["layers"]}
    rows = []
    for layer_cert in certificate["layers"]:
        layer = layer_cert["layer"]
        spectral = spectral_by_layer[layer]
        rows.append(
            {
                "layer": int(layer),
                "epsilon": float(layer_cert["epsilon"]),
                "rho": float(layer_cert["rho"]),
                "rho_worst": float(spectral["rho_worst"]),
                "rho_best": float(spectral["rho_best"]),
                "spectral_epsilon": float(spectral["epsilon"]),
                "max_lyapunov_entry": float(layer_cert["max_lyapunov_entry"]),
                "negative_definite": bool(layer_cert["negative_definite"]),
                "stable": bool(spectral["stable"]),
                "n_heads": int(layer_cert["n_heads"]),
            }
        )

    csv_path = RESULTS_DIR / "aqs_certificate_layers.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer",
            "epsilon",
            "rho",
            "rho_worst",
            "rho_best",
            "spectral_epsilon",
            "max_lyapunov_entry",
            "negative_definite",
            "stable",
            "n_heads",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "weight_path": WEIGHT_PATH,
        "max_A_beta": MAX_A_BETA,
        "dt_min": DT_MIN,
        "num_layers": NUM_LAYERS,
        "P": certificate["P"],
        "overall_stable": bool(results["overall_stable"]),
        "overall_certified": bool(certificate["overall_certified"]),
        "global_epsilon": float(certificate["global_epsilon"]),
        "layers_certified": sum(1 for row in rows if row["negative_definite"]),
        "plot_data_csv": str(csv_path),
    }
    json_path = RESULTS_DIR / "aqs_certificate_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(f"Certificate layer data saved to {csv_path}")
    logger.info(f"Certificate summary saved to {json_path}")


def main():
    logger.info("AQS Certificate: Constructive Stability Verification")
    logger.info(f"Hyper-rectangle bound: β = {MAX_A_BETA} (shared across all MaRK kernels)")
    logger.info(f"Worst-case dt: {DT_MIN}")
    logger.info(f"Loading pretrained weights from: {WEIGHT_PATH}\n")

    # ── Step 1: Load real pretrained parameters ──
    a_logs, _dt_biases = load_pretrained_parameters(WEIGHT_PATH)

    # ── Step 2: Spectral radius verification at vertices ──
    logger.info("\n--- Spectral Radius at Hyper-Rectangle Vertices ---")
    results = check_spectral_radius_at_vertices(a_logs, dt_min=DT_MIN, beta=MAX_A_BETA)

    # ── Step 3: P = I Lyapunov certificate with energy dissipation proof ──
    certificate = verify_lyapunov_certificate(a_logs, dt_min=DT_MIN, beta=MAX_A_BETA)

    # ── Step 4: Save plot-ready certificate data ──
    export_certificate_data(results, certificate)

    # ── Step 5: Generate certificate visualization ──
    generate_certificate_plot(results, certificate)

    # ── Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("CERTIFICATE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Lyapunov function:       P = I (identity matrix)")
    logger.info(f"  Energy function:         V(x) = ||x||²")
    logger.info(f"  Certificate status:      {'VERIFIED' if certificate['overall_certified'] else 'FAILED'}")
    logger.info(f"  Global stability margin: ε = {certificate['global_epsilon']:.6f}")
    logger.info(f"  Layers certified:        {sum(1 for l in certificate['layers'] if l['negative_definite'])}/{NUM_LAYERS}")
    logger.info(f"  Kernel independence:     All MaRK variants share β = {MAX_A_BETA}")
    logger.info(
        f"  Conclusion:              {'The modulated dynamics are strictly dissipative.' if certificate['overall_certified'] else 'Certificate failed — investigate individual layers.'}"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
