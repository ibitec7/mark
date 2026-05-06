"""Generate the paper's Markov norm-vs-lag dynamics figure.

Paper artifact:
    plots/markov_norm_vs_lag_k4096.png

Run from the repository root:
    python -m analysis.dynamics_analysis

The script reconstructs MaRK adapters from the three checkpoints, captures real
pre-MaRK B/C vectors with one model forward per kernel, and plots the
head-resolved Markov norm at the timestep snapshots used in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import export_dynamics_csv as edc
from . import markov_head_sequence as mhs
from . import markov_io_capture as mio
from .utils import setup_logger
from src.mark import ChebyshevPolynomial, DCTKernel, Hypernet

Path("logs").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
logger = setup_logger("logs/dynamics_analysis.log", write_console=True)

NUM_LAYERS = 23
N_HEADS = 12
N_GROUPS = 1
D_STATE = 64
RANK = 2
COND_DIM = 128
HIDDEN_DIM = 256
N_TIMESTEPS = 101
MAX_TIMESTEP = 1000
USE_SCALED_ENCODING = True

LAYER_PREFIX = "inner.hydra.encoder.layer.{}.layer.mixer"
TIME_MLP_PREFIX = "inner.hydra.embeddings"

CHECKPOINT_PATHS = {
    "Hypernet": "models/hydra_mark_hypernet_apr/best_hydra_mark.ckpt",
    "Chebyshev": "models/hydra_mark_chebyshev_apr/best_hydra_mark.ckpt",
    "DCT": "models/hydra_mark_dct_apr/best_hydra_mark.ckpt",
}

KERNEL_CLASSES = {
    "Hypernet": Hypernet,
    "Chebyshev": ChebyshevPolynomial,
    "DCT": DCTKernel,
}

KERNEL_EXTRA_KWARGS = {
    "Hypernet": {},
    "Chebyshev": {"degree": 5},
    "DCT": {"n_freqs": 8, "L_timepoints": 256},
}

KERNEL_COLORS = {
    "Hypernet": "#1f77b4",
    "Chebyshev": "#2ca02c",
    "DCT": "#d62728",
}

PAGE_WIDTH_IN = 13.5
FIG_SAVE_DPI = 300
MARKOV_K_MAX = 4096
MARKOV_OUTPUT = "plots/markov_norm_vs_lag_k4096.png"


def configure_neurips_style() -> None:
    """Publication-oriented matplotlib defaults used by the paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=FIG_SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", path)


def load_checkpoint(path: str) -> dict:
    """Load a Lightning checkpoint and return its flat state dict."""
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if "state" in ckpt:
        state = ckpt["state"]
        if isinstance(state, dict) and "model" in state:
            return state["model"]
        return state
    return ckpt


def build_time_mlp(sd: dict) -> tuple[nn.Sequential, torch.Tensor]:
    """Reconstruct the timestep MLP and sinusoidal frequency buffer."""
    time_mlp = nn.Sequential(
        nn.Linear(COND_DIM, COND_DIM),
        nn.SiLU(),
        nn.Linear(COND_DIM, COND_DIM),
    )
    prefix = f"{TIME_MLP_PREFIX}.time_mlp"
    time_mlp[0].weight.data.copy_(sd[f"{prefix}.0.weight"])
    time_mlp[0].bias.data.copy_(sd[f"{prefix}.0.bias"])
    time_mlp[2].weight.data.copy_(sd[f"{prefix}.2.weight"])
    time_mlp[2].bias.data.copy_(sd[f"{prefix}.2.bias"])
    time_mlp.eval()
    return time_mlp, sd[f"{TIME_MLP_PREFIX}.freqs"].float()


def build_adapter(kernel_name: str, layer_idx: int, sd: dict) -> nn.Module:
    """Instantiate one MaRK adapter and load the corresponding layer weights."""
    common_kwargs = dict(
        cond_dim=COND_DIM,
        n_heads=N_HEADS,
        n_groups=N_GROUPS,
        d_state=D_STATE,
        rank=RANK,
        hidden_dim=HIDDEN_DIM,
        factory_kwargs={"device": "cpu"},
    )
    adapter = KERNEL_CLASSES[kernel_name](**common_kwargs, **KERNEL_EXTRA_KWARGS[kernel_name])

    prefix = f"{LAYER_PREFIX.format(layer_idx)}.mark."
    adapter_sd = {}
    for key, value in sd.items():
        if key.startswith(prefix):
            adapter_sd[key[len(prefix) :]] = value.float()

    if not adapter_sd:
        raise KeyError(f"No adapter keys found for layer {layer_idx} with prefix {prefix}")
    adapter.load_state_dict(adapter_sd)
    adapter.float().eval()
    return adapter


def extract_base_params(layer_idx: int, sd: dict) -> dict[str, torch.Tensor]:
    """Extract frozen recurrence parameters needed for the Markov sequence."""
    prefix = LAYER_PREFIX.format(layer_idx)
    return {
        "A_log": sd[f"{prefix}.A_log"].float(),
        "dt_bias": sd[f"{prefix}.dt_bias"].float(),
    }


@torch.no_grad()
def generate_conditioning(t: float, freqs: torch.Tensor, time_mlp: nn.Sequential) -> torch.Tensor:
    """Reproduce the checkpoint's timestep encoding at ratio `t` in [0, 1]."""
    t_tensor = torch.tensor(t, dtype=torch.float32)
    scale = MAX_TIMESTEP if USE_SCALED_ENCODING else 1.0
    args = (t_tensor * scale) * freqs
    sin_embed = torch.cat([args.sin(), args.cos()], dim=-1)
    return time_mlp(sin_embed)


@torch.no_grad()
def sweep_all_kernels() -> dict[str, dict]:
    """Sweep timestep ratios and collect the parameters needed by the Markov figure."""
    timesteps = np.linspace(0.0, 1.0, N_TIMESTEPS)
    all_results: dict[str, dict] = {}
    reference_base_a_log: np.ndarray | None = None

    for kernel_name, checkpoint_path in CHECKPOINT_PATHS.items():
        logger.info("Processing %s", kernel_name)
        sd = load_checkpoint(checkpoint_path)
        time_mlp, freqs = build_time_mlp(sd)

        a_shift = np.zeros((N_TIMESTEPS, NUM_LAYERS, N_HEADS))
        dt_shift = np.zeros((N_TIMESTEPS, NUM_LAYERS, N_HEADS))
        b_scale = np.zeros((N_TIMESTEPS, NUM_LAYERS, N_GROUPS * D_STATE))
        b_shift = np.zeros((N_TIMESTEPS, NUM_LAYERS, N_GROUPS * D_STATE))
        c_scale = np.zeros((N_TIMESTEPS, NUM_LAYERS, N_GROUPS * D_STATE))
        c_shift = np.zeros((N_TIMESTEPS, NUM_LAYERS, N_GROUPS * D_STATE))

        base_params = []
        for layer_idx in range(NUM_LAYERS):
            adapter = build_adapter(kernel_name, layer_idx, sd)
            base_params.append(extract_base_params(layer_idx, sd))

            for ti, t in enumerate(timesteps):
                out = adapter(generate_conditioning(t, freqs, time_mlp))
                a_sh, b_sc, b_sh, c_sc, c_sh, dt_sh, _d_sh = out
                a_shift[ti, layer_idx] = a_sh.numpy()
                dt_shift[ti, layer_idx] = dt_sh.numpy()
                b_scale[ti, layer_idx] = b_sc.numpy()
                b_shift[ti, layer_idx] = b_sh.numpy()
                c_scale[ti, layer_idx] = c_sc.numpy()
                c_shift[ti, layer_idx] = c_sh.numpy()

        base_a_log = np.stack([bp["A_log"].numpy() for bp in base_params])
        base_dt = np.stack([bp["dt_bias"].numpy() for bp in base_params])
        if reference_base_a_log is None:
            reference_base_a_log = base_a_log
        else:
            diff = float(np.max(np.abs(base_a_log - reference_base_a_log)))
            logger.info("Base A_log max diff vs first kernel: %.2e", diff)
            if diff >= 1e-4:
                raise ValueError(f"Base A_log mismatch across kernels: {diff}")

        a_log_mod = base_a_log[None, :, :] + a_shift
        dt_mod = base_dt[None, :, :] + dt_shift
        a_cont = -np.exp(a_log_mod)
        delta = np.log1p(np.exp(dt_mod))
        a_disc = np.exp(a_cont * delta)

        max_eig = float(np.max(a_disc))
        max_a_shift = float(np.max(np.abs(a_shift)))
        logger.info("  A_disc max %.6f, max |A_shift| %.6f", max_eig, max_a_shift)
        if max_eig >= 1.0:
            raise ValueError(f"AQS violated: max eigenvalue {max_eig} >= 1")
        if max_a_shift > 0.5 + 1e-6:
            raise ValueError(f"|A_shift| exceeds bound: {max_a_shift}")

        all_results[kernel_name] = {
            "timesteps": timesteps,
            "A_disc": a_disc,
            "B_scale": b_scale,
            "B_shift": b_shift,
            "C_scale": c_scale,
            "C_shift": c_shift,
            "base_A_log": base_a_log,
            "base_dt": base_dt,
        }

    return all_results


def compute_markov_norm_curves(results: dict[str, dict]) -> dict[str, dict]:
    """Compute only the norm curves used by the paper, discarding large tensors promptly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curve_data: dict[str, dict] = {}
    logger.info("Capturing B/C and computing k=%d Markov curves on %s", MARKOV_K_MAX, device)

    for kernel_name, checkpoint_path in CHECKPOINT_PATHS.items():
        try:
            b_stack, c_stack = mio.capture_pre_mark_bc_stack(
                checkpoint_path,
                kernel_name,
                device,
                seq_len=mio.MARKOV_SEQ_LEN,
                token_pos=mio.MARKOV_TOKEN_POS,
            )
            head_m = mhs.compute_markov_head_sequence(b_stack, c_stack, results[kernel_name], MARKOV_K_MAX)
            head_frozen = mhs.compute_frozen_markov_head_sequence(
                b_stack,
                c_stack,
                results[kernel_name],
                MARKOV_K_MAX,
            )
        except Exception as exc:
            logger.exception("Skipping %s dynamics figure data: %s", kernel_name, exc)
            continue

        timesteps = results[kernel_name]["timesteps"]
        snapshot_indices = mio.snapshot_indices(timesteps, mio.T_SNAPSHOTS)
        modulated_curves = [
            mhs.network_norm_vs_k(head_m[idx, :, :, :MARKOV_K_MAX]) for idx in snapshot_indices
        ]
        frozen_curve = mhs.network_norm_vs_k(head_frozen[:, :, :MARKOV_K_MAX])
        curve_data[kernel_name] = {
            "snapshot_indices": snapshot_indices,
            "snapshot_timesteps": [float(timesteps[idx]) for idx in snapshot_indices],
            "modulated_curves": modulated_curves,
            "frozen_curve": frozen_curve,
        }

        logger.info("  %s: collected %d timestep curves", kernel_name, len(modulated_curves))
        del head_m, head_frozen
        if device == "cuda":
            torch.cuda.empty_cache()

    return curve_data


def plot_markov_norm_vs_lag(curve_data: dict[str, dict], output: str = MARKOV_OUTPUT) -> None:
    """Plot the head-resolved Markov norm curves referenced by the paper."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_IN, 4.5), constrained_layout=True)
    for col, kernel_name in enumerate(CHECKPOINT_PATHS):
        ax = axes[col]
        if kernel_name not in curve_data:
            ax.set_visible(False)
            continue

        data = curve_data[kernel_name]
        k_axis = np.arange(len(data["frozen_curve"]))
        cmap = plt.cm.viridis
        for j, (timestep, norm_curve) in enumerate(
            zip(data["snapshot_timesteps"], data["modulated_curves"], strict=True)
        ):
            color = cmap(0.15 + 0.75 * j / max(len(data["modulated_curves"]) - 1, 1))
            ax.semilogy(
                k_axis,
                np.maximum(norm_curve, 1e-30),
                color=color,
                linewidth=1.5,
                label=rf"$t={timestep:.2f}$",
            )
        ax.semilogy(
            k_axis,
            np.maximum(data["frozen_curve"], 1e-30),
            color="0.45",
            linestyle="--",
            linewidth=1.8,
            label="Frozen backbone",
        )
        ax.set_xlabel(r"Lag $k$", fontsize=15)
        ax.set_title(kernel_name, fontsize=15)
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
        ax.tick_params(axis="both", which="major", labelsize=10)

    for ax in axes:
        if ax.get_visible():
            ax.set_ylabel(r"$||m_{\ell,h}(k)||^2$", fontsize=15)
            break
    fig.suptitle("Head-resolved Markov norm vs lag", fontsize=15)
    _savefig(fig, output)


def main() -> None:
    logger.info("Starting paper dynamics analysis")
    configure_neurips_style()
    results = sweep_all_kernels()
    curve_data = compute_markov_norm_curves(results)
    edc.export_markov_norm_vs_lag_csv(curve_data)
    plot_markov_norm_vs_lag(curve_data)
    logger.info("Dynamics analysis complete")


if __name__ == "__main__":
    main()
