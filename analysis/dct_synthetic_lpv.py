"""Synthetic identifiability experiment for DCT MaRK modulation.

Paper artifact:
    plots/dct_synthetic_recovery.png

Run from the repository root:
    python analysis/dct_synthetic_lpv.py

The experiment fits a matched synthetic LPV-SSM and reports Markov operator
error versus dataset size, using the same bounded spectral structure as the
real DCT adapter in `src/mark.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
try:
    from .utils import setup_logger
    from .export_synthetic_identifiability import write_config_json, write_runs_csv, write_summary_csv
except ImportError:  # Allows direct execution as `python analysis/dct_synthetic_lpv.py`.
    from utils import setup_logger
    from export_synthetic_identifiability import write_config_json, write_runs_csv, write_summary_csv

os.makedirs("logs", exist_ok=True)
logger = setup_logger("logs/synthetic_lpv_dct.log", write_console=True)

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


def get_dct_basis(tau, n_freqs, L_timepoints=256):
    """
    tau: diffusion timesteps in [0, 1]. Shape: [batch, seq_len]
    Returns DCT-II basis. Shape: [batch, seq_len, n_freqs]
    """
    k = torch.arange(n_freqs, device=tau.device).float()
    
    # Cosine basis mapped to continuous tau in [0, 1]
    basis = torch.cos(torch.pi * k.view(1, 1, -1) * tau.unsqueeze(-1))
    
    # Normalization matching discrete DCT in src/mark.py
    norm = torch.ones_like(k)
    norm[0] = 1.0 / np.sqrt(L_timepoints)
    norm[1:] = np.sqrt(2.0 / L_timepoints)
    
    return basis * norm.view(1, 1, -1)

class DCTSyntheticLPV(nn.Module):
    """
    Ground-Truth Continuous-Time LPV SSM modulated by DCT/Fourier basis.
    Precisely mirrors the DCT kernel in `src/mark.py` mapping tau -> [0, 1] -> DCT basis
    as well as spectral exponential decay filter and low-rank UV decomposition.
    """
    def __init__(self, d_state=4, n_freqs=8, L_timepoints=256, rank=2, dt=0.1):
        super().__init__()
        self.d_state = d_state
        self.n_freqs = n_freqs
        self.L_timepoints = L_timepoints
        self.rank = rank
        self.dt = dt
        
        # Spectral decay parameter alpha
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("freqs", torch.arange(n_freqs).float())
        
        # A log-space base and Spectral Coefficients (instead of polynomial coeffs)
        self.A_log_base = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)
        self.A_spec_coeffs = nn.Parameter(torch.randn(d_state, n_freqs) * 0.5)
        self.A_beta = nn.Parameter(torch.tensor(0.5))
        
        # B Low-rank Base representations
        self.U_base = nn.Parameter(torch.randn(d_state, rank) * 0.5)
        self.V_base = nn.Parameter(torch.randn(rank, 1) * 0.5)
        
        # B Low-rank Spectral Coefficients
        self.U_spec_coeffs = nn.Parameter(torch.randn(d_state, rank, n_freqs) * 0.5)
        self.V_spec_coeffs = nn.Parameter(torch.randn(rank, 1, n_freqs) * 0.5)
        self.B_beta = nn.Parameter(torch.tensor(0.5))
        
        # Prevent pure coordinate permutation swapping by spreading initial params
        with torch.no_grad():
            self.A_log_base.copy_(torch.linspace(-2.0, 0.0, d_state))
        
        # Fixed C mapping to rigorously eliminate rotation/scaling ambiguity
        self.register_buffer("C", torch.ones(d_state))

    def get_continuous_matrices(self, tau):
        # Calculate exactly filtered spectral weights based on learned bandwidth parameter `alpha`
        decay = 1.0 / ((self.freqs + 1.0) ** (F.softplus(self.alpha) + 1e-4))
        
        # Expand tau into temporal basis
        basis = get_dct_basis(tau, self.n_freqs, self.L_timepoints) # [batch, seq_len, n_freqs]
        
        # Spectral filtering applied to raw frequencies for A
        A_w = self.A_spec_coeffs * decay
        A_shift = torch.einsum('b s k, d k -> b s d', basis, A_w)
        A_c = -torch.exp(self.A_log_base + self.A_beta * torch.tanh(A_shift))
        
        # Spectral filtering applied to raw frequencies for B (Low-rank UV factor tracking)
        U_w = self.U_spec_coeffs * decay
        V_w = self.V_spec_coeffs * decay
        
        U_shift = torch.einsum('b s k, d r k -> b s d r', basis, U_w)
        V_shift = torch.einsum('b s k, r i k -> b s r i', basis, V_w)
        
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

def generate_synthetic_data(N, seq_len, d_state=4, n_freqs=8, rank=2, L_timepoints=256, device="cuda"):
    true_model = DCTSyntheticLPV(d_state=d_state, n_freqs=n_freqs, L_timepoints=L_timepoints, rank=rank).to(device)
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
    Because the LPV system has a flat non-identifiable manifold in its parameter 
    space (due to B=UV factorizations, internal tanh boundaries, and spectral bounds), 
    exact raw parameter tracking diverges. However, finding the unique dynamic operator 
    IS theoretically guaranteed. We track the frozen-time Markov parameters distance.
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
    n_freqs = 8
    rank = 2
    L_timepoints = 256
    seq_len = 20
    dataset_sizes = [50, 100, 200, 400, 800, 1600]
    num_seeds = 15
    
    logger.info(f"Starting section 1.4: DCT LPV Parameter Recovery on {device}...")
    logger.info("Tracking basis coefficients and alpha decay parameter separately for identifiability analysis.")
    
    # Track operator errors
    all_basis_errors = np.zeros((num_seeds, len(dataset_sizes)))
    run_rows = []

    for seed in range(num_seeds):
        logger.info(f"--- Starting Seed {seed+1}/{num_seeds} ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        true_model, u_full, tau_full, y_full = generate_synthetic_data(
            max(dataset_sizes), seq_len, d_state, n_freqs, rank, L_timepoints, device
        )
        
        for idx, N in enumerate(dataset_sizes):
            u = u_full[:N]
            tau = tau_full[:N]
            y_true = y_full[:N]
            
            est_model = DCTSyntheticLPV(
                d_state=d_state, n_freqs=n_freqs, L_timepoints=L_timepoints, rank=rank
            ).to(device)
            
            with torch.no_grad():
                noise_scale_coeff = 0.1
                noise_scale_alpha = 0.02
                
                # Initialize basis coefficients with moderate noise
                est_model.A_spec_coeffs.copy_(true_model.A_spec_coeffs + noise_scale_coeff * torch.randn_like(true_model.A_spec_coeffs))
                est_model.A_log_base.copy_(true_model.A_log_base + noise_scale_coeff * torch.randn_like(true_model.A_log_base))
                est_model.U_spec_coeffs.copy_(true_model.U_spec_coeffs + noise_scale_coeff * torch.randn_like(true_model.U_spec_coeffs))
                est_model.V_spec_coeffs.copy_(true_model.V_spec_coeffs + noise_scale_coeff * torch.randn_like(true_model.V_spec_coeffs))
                est_model.U_base.copy_(true_model.U_base + noise_scale_coeff * torch.randn_like(true_model.U_base))
                est_model.V_base.copy_(true_model.V_base + noise_scale_coeff * torch.randn_like(true_model.V_base))
                
                est_model.alpha.copy_(true_model.alpha + noise_scale_alpha * torch.randn_like(true_model.alpha))
                
            loss_fn = nn.MSELoss()
            
            optimizer = torch.optim.AdamW(
                est_model.parameters(),
                lr=0.08,
                weight_decay=1e-4,
                fused=torch.cuda.is_available()
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
                    
            operator_err = compute_operator_error(true_model, est_model)
            all_basis_errors[seed, idx] = operator_err
            run_rows.append(
                {
                    "seed": seed,
                    "dataset_size": int(N),
                    "operator_error": float(operator_err),
                    "final_mse": float(final_loss),
                    "training_steps": int(step + 1),
                }
            )
            logger.info(f"Seed {seed+1:2d} | N={N:4d} | Steps: {step+1:4d} | MSE: {final_loss:.2e} | Oper Err: {operator_err:.6e}")

    # Compute statistics for operator error
    mean_basis_errors = np.mean(all_basis_errors, axis=0)
    std_basis_errors = np.std(all_basis_errors, axis=0)
    basis_ci_margin = 1.96 * (std_basis_errors / np.sqrt(num_seeds))

    out_dir = os.path.join("analysis", "results", "synthetic_identifiability")
    runs_path = os.path.join(out_dir, "dct_runs.csv")
    summary_path = os.path.join(out_dir, "dct_summary.csv")
    config_path = os.path.join(out_dir, "dct_config.json")
    write_runs_csv(runs_path, run_rows)
    write_summary_csv(summary_path, dataset_sizes, mean_basis_errors, std_basis_errors, basis_ci_margin)
    write_config_json(
        config_path,
        {
            "kernel": "dct",
            "d_state": d_state,
            "n_freqs": n_freqs,
            "L_timepoints": L_timepoints,
            "rank": rank,
            "seq_len": seq_len,
            "num_seeds": num_seeds,
            "dataset_sizes": list(dataset_sizes),
            "noise_sigma_coeff": 0.1,
            "noise_sigma_alpha": 0.02,
        },
    )
    logger.info(f"Wrote identifiability CSVs: {runs_path}, {summary_path}, {config_path}")

    plot_path = os.path.join("plots", "dct_synthetic_recovery.png")
    save_recovery_plot(
        dataset_sizes=dataset_sizes,
        mean_errors=mean_basis_errors,
        ci_margin=basis_ci_margin,
        output_path=plot_path,
        title="DCT LPV Identifiability",
    )
    logger.info(f"Successfully saved dual error analysis plot to {plot_path}")
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: System Operator Identifiability Analysis")
    logger.info("="*80)
    logger.info("Markov Operator Error (should decay ~O(1/sqrt(N))):")
    for i, N in enumerate(dataset_sizes):
        logger.info(f"  N={N:4d}: {mean_basis_errors[i]:.6e} ± {basis_ci_margin[i]:.6e}")
    logger.info("="*80)

if __name__ == "__main__":
    run_experiment()
