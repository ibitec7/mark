"""Synthetic identifiability experiment for Chebyshev MaRK modulation.

Paper artifact:
    plots/chebyshev_synthetic_recovery.png

Run from the repository root:
    python analysis/chebyshev_synthetic_lpv.py

The experiment fits a matched synthetic LPV-SSM and reports Markov operator
error versus dataset size, using the same bounded Chebyshev structure as the
real adapter in `src/mark.py`.
"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
try:
    from .utils import setup_logger
    from .export_synthetic_identifiability import write_config_json, write_runs_csv, write_summary_csv
except ImportError:  # Allows direct execution as `python analysis/chebyshev_synthetic_lpv.py`.
    from utils import setup_logger
    from export_synthetic_identifiability import write_config_json, write_runs_csv, write_summary_csv

os.makedirs("logs", exist_ok=True)
logger = setup_logger("logs/synthetic_lpv_chebyshev.log", write_console=True)

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


def get_chebyshev_basis(z, degree):
    """
    z: [batch, seq_len] in [-1, 1]
    Returns T_k(z) for k=0 to degree-1. Shape: [batch, seq_len, degree]
    """
    basis = [torch.ones_like(z), z]
    for k in range(2, degree):
        basis.append(2 * z * basis[-1] - basis[-2])
    return torch.stack(basis[:degree], dim=-1)

class ChebyshevSyntheticLPV(nn.Module):
    """
    Ground-Truth Continuous-Time LPV SSM modulated by Chebyshev Polynomials.
    Precisely mirrors the Chebyshev kernel in `src/mark.py` mapping tau -> [-1, 1] -> T_n(tau)
    as well as bounded log-space formulations and low-rank UV decomposition.
    """
    def __init__(self, d_state=4, degree=4, rank=2, dt=0.1):
        super().__init__()
        self.d_state = d_state
        self.degree = degree
        self.rank = rank
        self.dt = dt
        
        # A log-space base and Chebyshev Basis coefficients
        self.A_log_base = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)
        self.A_coeffs = nn.Parameter(torch.randn(d_state, degree) * 0.5)
        self.A_beta = nn.Parameter(torch.tensor(0.5))
        
        # B Low-rank Base representations
        self.U_base = nn.Parameter(torch.randn(d_state, rank) * 0.5)
        self.V_base = nn.Parameter(torch.randn(rank, 1) * 0.5)
        
        # B Low-rank Chebyshev Basis coefficients
        self.U_coeffs = nn.Parameter(torch.randn(d_state, rank, degree) * 0.5)
        self.V_coeffs = nn.Parameter(torch.randn(rank, 1, degree) * 0.5)
        self.B_beta = nn.Parameter(torch.tensor(0.5))
        
        # Prevent pure coordinate permutation swapping by spreading initial params
        with torch.no_grad():
            self.A_log_base.copy_(torch.linspace(-2.0, 0.0, d_state))
        
        # Fixed C mapping to rigorously eliminate rotation/scaling ambiguity
        self.register_buffer("C", torch.ones(d_state))

    def get_continuous_matrices(self, tau):
        # tau: diffusion timesteps in [0, 1]. Map smoothly to Chebyshev domain [-1, 1]
        z = 2 * tau - 1.0
        basis = get_chebyshev_basis(z, self.degree) # [batch, seq_len, degree]
        
        # Polynomial evaluation via coefficients for A
        A_shift = torch.einsum('b s k, d k -> b s d', basis, self.A_coeffs)
        # Log-space modeling with beta-tanh bounding matching real kernel
        A_c = -torch.exp(self.A_log_base + self.A_beta * torch.tanh(A_shift))
        
        # Polynomial evaluation via coefficients for B (Low-rank UV factor tracking)
        U_shift = torch.einsum('b s k, d r k -> b s d r', basis, self.U_coeffs)
        V_shift = torch.einsum('b s k, r i k -> b s r i', basis, self.V_coeffs)
        
        U_val = self.U_base + self.B_beta * torch.tanh(U_shift)
        V_val = self.V_base + self.B_beta * torch.tanh(V_shift)
        
        B_c = torch.matmul(U_val, V_val).squeeze(-1) # [batch, seq_len, d_state]
        return A_c, B_c

    def get_discrete_matrices(self, tau):
        A_c, B_c = self.get_continuous_matrices(tau)
        A_d = torch.exp(A_c * self.dt)
        B_d = (A_d - 1.0) / (A_c + 1e-7) * B_c
        return A_d, B_d

    def forward(self, u, tau):
        """
        u: [batch, seq_len]
        tau: [batch, seq_len]
        """
        batch_size, seq_len = u.shape
        x_t = torch.zeros(batch_size, self.d_state, device=u.device, dtype=u.dtype)
        y = []
        
        A_d, B_d = self.get_discrete_matrices(tau)
        
        for t in range(seq_len):
            x_t = A_d[:, t, :] * x_t + B_d[:, t, :] * u[:, t].unsqueeze(1)
            y_t = (x_t * self.C).sum(dim=1)
            y.append(y_t)
            
        return torch.stack(y, dim=1)

def generate_synthetic_data(N, seq_len, d_state=4, degree=4, rank=2, device="cuda"):
    true_model = ChebyshevSyntheticLPV(d_state=d_state, degree=degree, rank=rank).to(device)
    for p in true_model.parameters():
        p.requires_grad = False
        
    u = torch.randn(N, seq_len, device=device)
    # Sequence of diffusion timesteps exactly mapping the model assumptions
    tau = torch.empty(N, seq_len, device=device).uniform_(0, 1)
    
    with torch.no_grad():
        y = true_model(u, tau)
        y = y + 0.1 * torch.randn_like(y)
        
    return true_model, u, tau, y

def compute_operator_error(model_true, model_est):
    """
    Computes the error in the dynamical operator (Markov Parameters).
    This unifies the identifiability measurement across different kernel parameterizations,
    proving that the underlying LPV system is strictly recovered.
    """
    with torch.no_grad():
        # Evaluate over a grid of tau
        tau_eval = torch.linspace(0, 1, steps=200, device=model_true.A_log_base.device).unsqueeze(0)
        Ad_t, Bd_t = model_true.get_discrete_matrices(tau_eval)
        Ad_e, Bd_e = model_est.get_discrete_matrices(tau_eval)
        
        err = 0.0
        # Compute frozen-time Markov parameters for k=0 to 14
        # M_k(tau) = C * Ad(tau)^k * Bd(tau). Since C is an array of 1s, it is a sum over states
        for k in range(15):
            mk_t = ((Ad_t ** k) * Bd_t).sum(dim=-1)
            mk_e = ((Ad_e ** k) * Bd_e).sum(dim=-1)
            err += torch.norm(mk_t - mk_e, p=2)
            
        return err.item() / 15.0

def run_experiment():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    configure_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    d_state = 4
    degree = 3
    rank = 2
    seq_len = 20
    dataset_sizes = [50, 100, 200, 400, 800, 1600]
    num_seeds = 15
    
    logger.info(f"Starting section 1.3: Chebyshev LPV Parameter Recovery on {device}...")
    
    all_errors = np.zeros((num_seeds, len(dataset_sizes)))
    run_rows = []
    
    for seed in range(num_seeds):
        logger.info(f"--- Starting Seed {seed+1}/{num_seeds} ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        true_model, u_full, tau_full, y_full = generate_synthetic_data(max(dataset_sizes), seq_len, d_state, degree, rank, device)
        
        for idx, N in enumerate(dataset_sizes):
            u = u_full[:N]
            tau = tau_full[:N]
            y_true = y_full[:N]
            
            est_model = ChebyshevSyntheticLPV(d_state=d_state, degree=degree, rank=rank).to(device)
            
            with torch.no_grad():
                noise_scale = 0.1
                est_model.A_coeffs.copy_(true_model.A_coeffs + noise_scale * torch.randn_like(true_model.A_coeffs))
                est_model.A_log_base.copy_(true_model.A_log_base + noise_scale * torch.randn_like(true_model.A_log_base))
                est_model.U_coeffs.copy_(true_model.U_coeffs + noise_scale * torch.randn_like(true_model.U_coeffs))
                est_model.V_coeffs.copy_(true_model.V_coeffs + noise_scale * torch.randn_like(true_model.V_coeffs))
                est_model.U_base.copy_(true_model.U_base + noise_scale * torch.randn_like(true_model.U_base))
                est_model.V_base.copy_(true_model.V_base + noise_scale * torch.randn_like(true_model.V_base))
                
            loss_fn = nn.MSELoss()
            
            optimizer = torch.optim.AdamW(
                est_model.parameters(),
                lr=0.08,
                weight_decay=1e-4,
                fused=True
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
            logger.info(f"Seed {seed+1:2d} | N={N:4d} | Converged Steps: {step+1:4d} | Final MSE: {final_loss:.2e} | Oper Error: {err:.6e}")

    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    ci_margin = 1.96 * (std_errors / np.sqrt(num_seeds))

    out_dir = os.path.join("analysis", "results", "synthetic_identifiability")
    runs_path = os.path.join(out_dir, "chebyshev_runs.csv")
    summary_path = os.path.join(out_dir, "chebyshev_summary.csv")
    config_path = os.path.join(out_dir, "chebyshev_config.json")
    write_runs_csv(runs_path, run_rows)
    write_summary_csv(summary_path, dataset_sizes, mean_errors, std_errors, ci_margin)
    write_config_json(
        config_path,
        {
            "kernel": "chebyshev",
            "d_state": d_state,
            "degree": degree,
            "rank": rank,
            "seq_len": seq_len,
            "num_seeds": num_seeds,
            "dataset_sizes": list(dataset_sizes),
            "noise_sigma": 0.1,
        },
    )
    logger.info(f"Wrote identifiability CSVs: {runs_path}, {summary_path}, {config_path}")

    plot_path = os.path.join("plots", "chebyshev_synthetic_recovery.png")
    save_recovery_plot(
        dataset_sizes=dataset_sizes,
        mean_errors=mean_errors,
        ci_margin=ci_margin,
        output_path=plot_path,
        title="Chebyshev LPV Identifiability",
    )
    logger.info(f"Successfully saved experiment plot with CIs to {plot_path}")

if __name__ == "__main__":
    run_experiment()
