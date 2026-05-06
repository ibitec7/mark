"""
Synthetic identifiability experiment for Hypernet-style MaRK modulation.

Paper artifact:
    plots/hypernet_synthetic_recovery.png

Run from the repository root:
    python analysis/hypernet_synthetic_lpv.py

This mirrors the Chebyshev/DCT synthetic protocol while using Hypernet geometry:
tau -> sinusoidal embedding -> 4-layer SiLU MLP -> A-shift + B UV scale/shift.
The measurement target is frozen-time Markov operator error versus dataset size.
"""

import os
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

try:
    from .utils import setup_logger
    from .export_synthetic_identifiability import write_config_json, write_runs_csv, write_summary_csv
except ImportError:  # Allows direct execution as `python analysis/hypernet_synthetic_lpv.py`.
    from utils import setup_logger
    from export_synthetic_identifiability import write_config_json, write_runs_csv, write_summary_csv


os.makedirs("logs", exist_ok=True)
logger = setup_logger("logs/synthetic_lpv_hypernet.log", write_console=True)


def configure_plot_style():
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


def save_recovery_plot(dataset_sizes, mean_errors, ci_margin, output_path, title):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    main_color = "b"
    theory_color = "r"

    ax.plot(
        dataset_sizes,
        mean_errors,
        marker="o",
        color=main_color,
        linewidth=2.5,
        markersize=8,
        label="Markov Operator Error",
    )
    ax.fill_between(
        dataset_sizes,
        mean_errors - ci_margin,
        mean_errors + ci_margin,
        color=main_color,
        alpha=0.25,
        label="95% CI (Normal approx.)",
    )

    theory_constant = mean_errors[0] * np.sqrt(dataset_sizes[0])
    theory_curve = [theory_constant / np.sqrt(n) for n in dataset_sizes]
    ax.plot(dataset_sizes, theory_curve, linestyle="--", color=theory_color, linewidth=2.5, label=r"Theory $\mathcal{O}(1/\sqrt{N})$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (N)", fontsize=15)
    ax.set_ylabel(r"Operator Error $\|M(\tau)-\hat{M}(\tau)\|_F$", fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_factors_near_sqrt(length):
    best = None
    for p in range(int(math.sqrt(length)), 0, -1):
        if length % p == 0:
            q = length // p
            best = (p, q)
            break
    if best is None:
        p = int(math.sqrt(length))
        q = int(math.ceil(length / p))
        best = (p, q)
    return best


def sinusoidal_embedding(tau, n_freqs=64):
    """
    tau: [batch, seq_len] in [0, 1]
    returns: [batch, seq_len, 2*n_freqs]
    """
    freqs = torch.arange(n_freqs, device=tau.device, dtype=tau.dtype)
    angles = tau.unsqueeze(-1) * freqs.view(1, 1, -1) * (2.0 * torch.pi)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class HypernetSyntheticLPV(nn.Module):
    """
    Diagonal LPV-SSM with Hypernet-style modulation for A and B.
    """

    def __init__(
        self,
        d_state=4,
        rank=2,
        cond_dim=128,
        hidden_dim=256,
        alpha_init=0.2,
        beta_init=0.2,
        dt=0.1,
    ):
        super().__init__()
        self.d_state = d_state
        self.rank = rank
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.dt = dt

        self.A_log_base = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.A_shift_proj = nn.Linear(hidden_dim, d_state)

        self.U_dim, self.V_dim = pick_factors_near_sqrt(d_state)
        self.rank_dim = (self.U_dim + self.V_dim) * rank

        self.B_scale_proj = nn.Linear(hidden_dim, self.rank_dim)
        self.B_shift_proj = nn.Linear(hidden_dim, self.rank_dim)

        self.C_scale_proj = nn.Linear(hidden_dim, self.rank_dim)
        self.C_shift_proj = nn.Linear(hidden_dim, self.rank_dim)

        self.U_base = nn.Parameter(torch.randn(self.U_dim, rank) * 0.5)
        self.V_base = nn.Parameter(torch.randn(rank, self.V_dim) * 0.5)

        self.max_A_beta = 0.5
        self.A_beta_param = nn.Parameter(torch.tensor(beta_init))
        self.B_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.B_beta = nn.Parameter(torch.tensor(beta_init))

        self.C_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.C_beta = nn.Parameter(torch.tensor(beta_init))

        with torch.no_grad():
            self.A_log_base.copy_(torch.linspace(-2.0, 0.0, d_state))

        # C(tau) will be modulated, but we keep a stable base readout.
        self.register_buffer("C_base", torch.ones(d_state))

    def _uv_to_flat(self, uv):
        u = uv[..., : self.U_dim * self.rank].reshape(*uv.shape[:-1], self.U_dim, self.rank)
        v = uv[..., self.U_dim * self.rank :].reshape(*uv.shape[:-1], self.rank, self.V_dim)
        flat = torch.matmul(u, v).reshape(*uv.shape[:-1], self.U_dim * self.V_dim)
        return flat[..., : self.d_state]

    def get_continuous_matrices(self, tau):
        cond = sinusoidal_embedding(tau, n_freqs=self.cond_dim // 2)
        h = self.mlp(cond)

        a_beta = self.max_A_beta * torch.sigmoid(self.A_beta_param)
        a_shift = self.A_shift_proj(h)
        a_c = -torch.exp(self.A_log_base.view(1, 1, -1) + a_beta * torch.tanh(a_shift))

        b_base = torch.matmul(self.U_base, self.V_base).reshape(1, 1, -1)[..., : self.d_state]

        b_scale_uv = self.B_scale_proj(h)
        b_scale = self._uv_to_flat(b_scale_uv)
        b_scale = 1.0 + self.B_alpha * torch.tanh(b_scale)

        b_shift_uv = self.B_shift_proj(h)
        b_shift = self._uv_to_flat(b_shift_uv)
        # Synthetic bounded-shift variant: tanh before beta (requested).
        b_shift = self.B_beta * torch.tanh(b_shift)

        b_c = b_base * b_scale + b_shift

        c_scale_uv = self.C_scale_proj(h)
        c_scale = self._uv_to_flat(c_scale_uv)
        c_scale = 1.0 + self.C_alpha * torch.tanh(c_scale)

        c_shift_uv = self.C_shift_proj(h)
        c_shift = self._uv_to_flat(c_shift_uv)
        # Synthetic bounded-shift variant: tanh before beta (requested).
        c_shift = self.C_beta * torch.tanh(c_shift)

        c_vec = self.C_base.view(1, 1, -1) * c_scale + c_shift
        return a_c, b_c, c_vec

    def get_discrete_matrices(self, tau):
        a_c, b_c, c_vec = self.get_continuous_matrices(tau)
        a_d = torch.exp(a_c * self.dt)
        b_d = (a_d - 1.0) / (a_c + 1e-7) * b_c
        return a_d, b_d, c_vec

    def forward(self, u, tau):
        batch_size, seq_len = u.shape
        x_t = torch.zeros(batch_size, self.d_state, device=u.device, dtype=u.dtype)
        y = []

        a_d, b_d, c_vec = self.get_discrete_matrices(tau)

        for t in range(seq_len):
            x_t = a_d[:, t, :] * x_t + b_d[:, t, :] * u[:, t].unsqueeze(1)
            y_t = (x_t * c_vec[:, t, :]).sum(dim=1)
            y.append(y_t)
        return torch.stack(y, dim=1)


def generate_synthetic_data(N, seq_len, d_state=4, rank=2, device="cuda"):
    true_model = HypernetSyntheticLPV(d_state=d_state, rank=rank).to(device)
    for p in true_model.parameters():
        p.requires_grad = False

    u = torch.randn(N, seq_len, device=device)
    tau = torch.empty(N, seq_len, device=device).uniform_(0, 1)

    with torch.no_grad():
        y = true_model(u, tau)
        y = y + 0.1 * torch.randn_like(y)

    return true_model, u, tau, y


def compute_operator_error(model_true, model_est):
    with torch.no_grad():
        tau_eval = torch.linspace(0, 1, steps=200, device=model_true.A_log_base.device).unsqueeze(0)
        ad_t, bd_t, c_t = model_true.get_discrete_matrices(tau_eval)
        ad_e, bd_e, c_e = model_est.get_discrete_matrices(tau_eval)

        err = 0.0
        for k in range(15):
            mk_t = (c_t * ((ad_t ** k) * bd_t)).sum(dim=-1)
            mk_e = (c_e * ((ad_e ** k) * bd_e)).sum(dim=-1)
            err += torch.norm(mk_t - mk_e, p=2)

        return err.item() / 15.0


def run_experiment():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    configure_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_state = 4
    rank = 2
    seq_len = 20
    dataset_sizes = [50, 100, 200, 400, 800, 1600]
    num_seeds = 15

    logger.info(f"Starting section 1.5: Hypernet LPV Parameter Recovery on {device}...")
    all_errors = np.zeros((num_seeds, len(dataset_sizes)))
    run_rows = []

    for seed in range(num_seeds):
        logger.info(f"--- Starting Seed {seed+1}/{num_seeds} ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)

        true_model, u_full, tau_full, y_full = generate_synthetic_data(
            max(dataset_sizes), seq_len, d_state, rank, device
        )

        for idx, N in enumerate(dataset_sizes):
            u = u_full[:N]
            tau = tau_full[:N]
            y_true = y_full[:N]

            est_model = HypernetSyntheticLPV(d_state=d_state, rank=rank).to(device)
            with torch.no_grad():
                for name, p in est_model.named_parameters():
                    true_p = dict(true_model.named_parameters())[name]
                    if p.ndim == 0:
                        noise_scale = 0.02
                    else:
                        noise_scale = 0.1
                    p.copy_(true_p + noise_scale * torch.randn_like(true_p))

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.AdamW(
                est_model.parameters(),
                lr=0.08,
                weight_decay=1e-4,
                fused=torch.cuda.is_available(),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)

            final_loss = 0.0
            for step in range(1500):
                optimizer.zero_grad()
                y_pred = est_model(u, tau)
                loss = loss_fn(y_pred, y_true)
                loss.backward()
                optimizer.step()
                scheduler.step()

                final_loss = loss.item()
                if final_loss < 1e-7:
                    break

            err = compute_operator_error(true_model, est_model)
            all_errors[seed, idx] = err
            run_rows.append(
                {
                    "seed": seed,
                    "dataset_size": int(N),
                    "operator_error": float(err),
                    "final_mse": float(final_loss),
                    "training_steps": int(step + 1),
                }
            )
            logger.info(
                f"Seed {seed+1:2d} | N={N:4d} | Steps: {step+1:4d} | MSE: {final_loss:.2e} | Oper Err: {err:.6e}"
            )

    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    ci_margin = 1.96 * (std_errors / np.sqrt(num_seeds))

    out_dir = os.path.join("analysis", "results", "synthetic_identifiability")
    runs_path = os.path.join(out_dir, "hypernet_runs.csv")
    summary_path = os.path.join(out_dir, "hypernet_summary.csv")
    config_path = os.path.join(out_dir, "hypernet_config.json")
    write_runs_csv(runs_path, run_rows)
    write_summary_csv(summary_path, dataset_sizes, mean_errors, std_errors, ci_margin)
    write_config_json(
        config_path,
        {
            "kernel": "hypernet",
            "d_state": d_state,
            "rank": rank,
            "seq_len": seq_len,
            "num_seeds": num_seeds,
            "dataset_sizes": list(dataset_sizes),
            "noise_sigma_output": 0.1,
        },
    )
    logger.info(f"Wrote identifiability CSVs: {runs_path}, {summary_path}, {config_path}")

    plot_path = os.path.join("plots", "hypernet_synthetic_recovery.png")
    save_recovery_plot(
        dataset_sizes=dataset_sizes,
        mean_errors=mean_errors,
        ci_margin=ci_margin,
        output_path=plot_path,
        title="Hypernet LPV Identifiability",
    )
    logger.info(f"Successfully saved experiment plot with CIs to {plot_path}")

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Hypernet System Operator Identifiability Analysis")
    logger.info("=" * 80)
    logger.info("Markov Operator Error (should decay ~O(1/sqrt(N))):")
    for i, N in enumerate(dataset_sizes):
        logger.info(f"  N={N:4d}: {mean_errors[i]:.6e} +/- {ci_margin[i]:.6e}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_experiment()
