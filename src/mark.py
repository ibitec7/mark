# ========================================================================================================================== #
###                                       MaRK - (M)arkov-(a)dapted (R)ecurrent (K)ernels                                       ###
# ========================================================================================================================== #

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .utils import pick_factors_near_sqrt
class Hypernet(nn.Module):
    """A Hypernet to reconstruct the kernels parameters A, B, C, dt based on conditioning input for Hydra SSM.

    Args:
        cond (torch.Tensor): Conditioning input tensor.
        A (torch.Tensor): State matrix A, to control the recurrence dynamics. (how much information to retain vs to forget)
        B (torch.Tensor): Input matrix B, to control the read-in dynamics of the input into the state dimension.
        C (torch.Tensor): Output matrix C, to control the read-out dynamics from the state dimension to the output.
        dt (torch.Tensor): Time step tensor, to control the discretization of the SSM.
    """
    def __init__(
        self,
        cond_dim: int = 128,
        n_heads: int = 1,
        n_groups: int = 1,
        d_state: int = 64,
        rank: int = 2,
        alpha_init: float = 0.2,
        beta_init: float = 0.2,
        hidden_dim: int = 256,
        factory_kwargs = None,
    ):
        super().__init__()
        self.factory_kwargs = factory_kwargs if factory_kwargs is not None else {}

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
        )

        self.A_shift_proj = nn.Linear(hidden_dim, n_heads, **self.factory_kwargs)
        self.dt_shift_proj = nn.Linear(hidden_dim, n_heads, **self.factory_kwargs)
        self.D_shift_proj = nn.Linear(hidden_dim, n_heads, **self.factory_kwargs)

        self.rank = rank
        self.U_dim, self.V_dim = pick_factors_near_sqrt(n_groups * d_state)
        self.rank_dim = (self.U_dim + self.V_dim) * self.rank

        self.B_scale_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.B_shift_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)

        self.C_scale_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.C_shift_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)

        self.max_A_beta = 0.5

        self.A_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)
        self.dt_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)
        self.D_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        self.B_alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.B_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        self.C_alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.C_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

    def forward(
        self,
        cond: torch.Tensor,
        A_log: torch.Tensor = None,
        B: torch.Tensor = None,
        C: torch.Tensor = None,
        dt: torch.Tensor = None,
        D: torch.Tensor = None,
    ):
        # Hidden state projection
        h = self.mlp(cond)

        # Find the shifts for A, dt, and D
        A_shift: torch.Tensor = self.A_shift_proj(h)
        dt_shift: torch.Tensor = self.dt_shift_proj(h)
        D_shift: torch.Tensor = self.D_shift_proj(h)

        A_beta: torch.Tensor = self.max_A_beta * torch.sigmoid(self.A_beta)

        # Find the U-V decompositions for B scales and shifts
        B_UV_scale: torch.Tensor = self.B_scale_proj(h)
        B_UV_shift: torch.Tensor = self.B_shift_proj(h)

        B_U_scale, B_V_scale = torch.split(B_UV_scale, [self.U_dim * self.rank, self.V_dim * self.rank], dim=-1)
        B_scale = torch.matmul(B_U_scale.view(self.U_dim, self.rank), B_V_scale.view(self.rank, self.V_dim))
        B_scale = 1.0 + self.B_alpha * torch.tanh(B_scale.flatten())

        B_U_shift, B_V_shift = torch.split(B_UV_shift, [self.U_dim * self.rank, self.V_dim * self.rank], dim=-1)
        B_shift = torch.matmul(B_U_shift.view(self.U_dim, self.rank), B_V_shift.view(self.rank, self.V_dim))
        B_shift = self.B_beta * torch.tanh(B_shift.flatten())

        # Find the U-V decompositions for C scales and shifts
        C_UV_scale: torch.Tensor = self.C_scale_proj(h)
        C_UV_shift: torch.Tensor = self.C_shift_proj(h)

        C_U_scale, C_V_scale = torch.split(C_UV_scale, [self.U_dim * self.rank, self.V_dim * self.rank], dim=-1)
        C_scale = torch.matmul(C_U_scale.view(self.U_dim, self.rank), C_V_scale.view(self.rank, self.V_dim))
        C_scale = 1.0 + self.C_alpha * torch.tanh(C_scale.flatten())

        C_U_shift, C_V_shift = torch.split(C_UV_shift, [self.U_dim * self.rank, self.V_dim * self.rank], dim=-1)
        C_shift = torch.matmul(C_U_shift.view(self.U_dim, self.rank), C_V_shift.view(self.rank, self.V_dim))
        C_shift = self.C_beta * torch.tanh(C_shift.flatten())

    # =======================================================================================================================#

        param_list: list = [A_log, B, C, dt, D]

        if all(p is None for p in param_list):
            # return the shifts and scales only

            A_shift = A_beta * torch.tanh(A_shift)
            dt_shift = self.dt_beta * torch.tanh(dt_shift)
            D_shift = self.D_beta * torch.tanh(D_shift)

            return A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift
            
        else:
            # Apply the shifts to A, dt, and D
            A_log: torch.Tensor = A_log + A_beta * torch.tanh(A_shift)
            dt: torch.Tensor = dt + self.dt_beta * torch.tanh(dt_shift)         # Constraint on dt to be positive
            D: torch.Tensor = D + self.D_beta * torch.tanh(D_shift)

            B = B * B_scale.expand_as(B) + B_shift.expand_as(B)

            C = C * C_scale.expand_as(C) + C_shift.expand_as(C)

            return A_log, B, C, dt, D


class ChebyshevPolynomial(nn.Module):
    """Chebyshev Polynomial conditioned basis expansion of the A, B, C and dt parameters of the Hydra SSM.

    Args:
        cond_dim (torch.Tensor): The concatenated timestep and mask embedding vector.

        n_heads (int): Number of heads for A and dt.

        n_groups (int): Number of groups for B and C.

        d_state (int): State dimension per group for B and C.

        degree (int): Degree of the Chebyshev polynomial expansion.

        rank (int): Rank for the low-rank decomposition of B and C.

        hidden_dim (int): Hidden dimension of the MLP.
    """
    def __init__(
        self,
        cond_dim: int = 128,
        n_heads: int = 1,
        n_groups: int = 1,
        d_state: int = 64,
        degree: int = 5,
        rank: int = 2,
        alpha_init: float = 0.2,
        beta_init: float = 0.2,
        hidden_dim: int = 256,
        factory_kwargs = None,
    ):
        
        super().__init__()
        self.factory_kwargs = factory_kwargs or {}

        self.degree = degree
        self.n_heads = n_heads
        self.rank = rank

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
        )

        self.z_proj = nn.Linear(hidden_dim, n_heads, **self.factory_kwargs)

        self.A_proj = nn.Linear(hidden_dim, n_heads * degree, **self.factory_kwargs)
        self.dt_proj = nn.Linear(hidden_dim, n_heads * degree, **self.factory_kwargs)
        self.D_proj = nn.Linear(hidden_dim, n_heads * degree, **self.factory_kwargs)

        self.U_dim, self.V_dim = pick_factors_near_sqrt(n_groups * d_state)
        self.rank_dim = (self.U_dim + self.V_dim) * self.rank

        self.B_scale_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.B_scale_basis_proj = nn.Linear(hidden_dim, self.rank_dim * degree, **self.factory_kwargs)

        self.B_shift_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.B_shift_basis_proj = nn.Linear(hidden_dim, self.rank_dim * degree, **self.factory_kwargs)

        self.C_scale_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.C_scale_basis_proj = nn.Linear(hidden_dim, self.rank_dim * degree, **self.factory_kwargs)

        self.C_shift_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.C_shift_basis_proj = nn.Linear(hidden_dim, self.rank_dim * degree, **self.factory_kwargs)

        self.max_A_beta = 0.5

        self.A_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)
        self.dt_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)
        self.D_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        self.B_alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.B_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        self.C_alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.C_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

    def _chebyshev_basis(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(-1)  # (n_heads, 1)

            B = z.shape[0]
        
            terms = []
            
            t0 = torch.ones(B, device=z.device, dtype=z.dtype)
            terms.append(t0)

            if self.degree > 1:
                z_flat = z.squeeze(-1) # Shape (B,)
                terms.append(z_flat)

                for k in range(1, self.degree - 1):
                    t_next = 2.0 * z_flat * terms[-1] - terms[-2]
                    terms.append(t_next)

            return torch.stack(terms, dim=1)
    
    def _make_uv(self, base_uv, basis_uv, basis_mean):
        # scalar modulation from basis and its projection
        # (K, rank_dim) * (K, 1) -> (K, rank_dim) -> sum over K -> (rank_dim,)
        uv_adj = torch.sum(basis_uv * basis_mean.unsqueeze(-1), dim=0)  # dim=0, not 1
        uv = base_uv + uv_adj  # (rank_dim,)

        U = uv[: self.U_dim * self.rank].view(self.U_dim, self.rank)
        V = uv[self.U_dim * self.rank :].view(self.rank, self.V_dim)

        return torch.matmul(U, V).reshape(self.U_dim * self.V_dim)

    def forward(
        self,
        cond: torch.Tensor = None,
        A_log: torch.Tensor = None,
        B: torch.Tensor = None,
        C: torch.Tensor = None,
        dt: torch.Tensor = None,
        D: torch.Tensor = None,
    ):
        h: torch.Tensor = self.mlp(cond)    # (hidden_dim,)

        z: torch.Tensor = self.z_proj(h)    # (n_heads,)

        basis: torch.Tensor = self._chebyshev_basis(z)  # (n_heads, degree)

        K = self.degree

        A_shifts = self.A_proj(h).view(self.n_heads, K)      # (n_heads, degree)
        dt_shifts = self.dt_proj(h).view(self.n_heads, K)   # (n_heads, degree)
        D_shifts = self.D_proj(h).view(self.n_heads, K)     # (n_heads, degree)

        # Find the A, dt, and D shifts using the basis expansion
        A_shift = torch.sum(A_shifts * basis, dim=-1)       # (n_heads,)
        dt_shift = torch.sum(dt_shifts * basis, dim=-1)      # (n_heads,)
        D_shift = torch.sum(D_shifts * basis, dim=-1)      # (n_heads,)

        # Bound A_beta to a stable range to prevent gradient explosion/vanishing
        A_beta: torch.Tensor = self.max_A_beta * torch.sigmoid(self.A_beta)

        basis_mean = basis.mean(dim=0)

        B_scale = self._make_uv(
            self.B_scale_base_proj(h),
            self.B_scale_basis_proj(h).view(K, self.rank_dim),
            basis_mean,
        )

        B_shift = self._make_uv(
            self.B_shift_base_proj(h),
            self.B_shift_basis_proj(h).view(K, self.rank_dim),
            basis_mean,
        )

        C_scale = self._make_uv(
            self.C_scale_base_proj(h),
            self.C_scale_basis_proj(h).view(K, self.rank_dim),
            basis_mean,
        )

        C_shift = self._make_uv(
            self.C_shift_base_proj(h),
            self.C_shift_basis_proj(h).view(K, self.rank_dim),
            basis_mean,
        )

# =======================================================================================================================#

        param_list: list = [A_log, B, C, dt, D]

        if all(p is None for p in param_list):
            # return the shifts and scales only

            A_shift = A_beta * torch.tanh(A_shift)
            dt_shift = self.dt_beta * torch.tanh(dt_shift)
            D_shift = self.D_beta * torch.tanh(D_shift)

            B_scale = 1.0 + self.B_alpha * torch.tanh(B_scale)
            B_shift = self.B_beta * torch.tanh(B_shift)

            C_scale = 1.0 + self.C_alpha * torch.tanh(C_scale)
            C_shift = self.C_beta * torch.tanh(C_shift)

            return A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift
        
        else:
            # Apply the shifts to A, dt, and D
            A_log = A_log + A_beta * torch.tanh(A_shift)       # (n_heads,)
            dt = dt + self.dt_beta * torch.tanh(dt_shift)           # (n_heads,)
            D = D + self.D_beta * torch.tanh(D_shift)               # (n_heads,)

            B = B * (1.0 + self.B_alpha * torch.tanh(B_scale)) + self.B_beta * torch.tanh(B_shift)

            C = C * (1.0 + self.C_alpha * torch.tanh(C_scale)) + self.C_beta * torch.tanh(C_shift)

            return A_log, B, C, dt, D

class DCTKernel(nn.Module):
    """
    Real-only FFT kernel implementation for MaRK adapters. (real Fourier basis -> time -> modulation).

    Using the FFT would give us complex spectral weights which would require complex arithmetic to implement a true IRFFT.
    This would complicate the implementation as complex gradients would be needed, and numerical instability could arise and they also require
    2x memory and computation compared to real-valued operations. 

    For the sake of stability, we use real FFT basis here using DCT-II basis. So we do not construct complex spectral weights, and don't try to
    implement a true IRFFT. Instead, we use cosine basis to approximate the time vectors which is Fourier in spirit. 
    This allows us to avoid complex arithmetic and keep the implementation simple and efficient, and prevent numerical instability
    and complex gradient issues.
    
    1. Initializes a DCT-II cosine basis table for time vector construction. (see cosine_basis buffer in __init__ method)
    2. Produces spectral weights using the hidden state from the MLP on the conditioning vector, and applies the decay to normalize the spectral weights.
    3. Matmul the spectral weights with the cosine basis to get the time vectors. (how much each frequency contributes to the final waveform)
    
    NOTE: Here L_timepoints will provide the resolution of the time vector, ie: the resolution of the final wave sampled.
          n_freqs will provide the number of frequencies used to construct the time vector, ie: the bandwidth of the final wave.
          Higher n_freqs will give more detailed frequency representation (but make it more sensitive to high frequency noise), 
          while higher L_timepoints will give more detailed time representation. 

    4. A projection layer on the time vector to get the shifts for A, dt, and D.
    5. The hidden state from the MLP is projected to get the base U-V decompositions for B and C. The time vector is projected to
        get the basis U-V decompositions for B and C. The Base is added to the Basis to get the final U-V decompositions for B and C.
    
    Args:
        cond_dim (int): Dimension of the conditioning vector.

        n_heads  (int): Number of heads for A and dt.

        n_groups (int): Number of groups for B and C.

        d_state  (int): State dimension per group for B and C.

        n_freqs  (int): Number of frequencies for the Fourier basis table.

        L_timepoints (int): Number of timepoints sampled for the Fourier basis table.

        rank     (int): Rank for the low-rank decomposition of B and C.

        hidden_dim (int): Hidden dimension of the MLP.

        alpha_init (float): Initial value for alpha parameter controlling modulation range.

        beta_init  (float): Initial value for beta parameter controlling modulation range.
    """

    def __init__(
        self,
        cond_dim: int = 128,
        n_heads: int = 1,
        n_groups: int = 1,
        d_state: int = 64,
        n_freqs: int = 8,
        L_timepoints: int = 256,   # Maximum number of timepoints for basis construction, if None use n_groups * d_state
        rank: int = 2,
        hidden_dim: int = 256,
        alpha_init: float = 0.2,
        beta_init: float = 0.2,
        factory_kwargs = None,
    ):
        super().__init__()
        self.factory_kwargs = factory_kwargs or {}

        self.n_freqs = n_freqs
        
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_state = d_state
        self.n_freqs = n_freqs
        self.rank = rank

        self.U_dim, self.V_dim = pick_factors_near_sqrt(n_groups * d_state)
        self.L_timepoints = L_timepoints or (self.U_dim * self.V_dim)

        # using a power-of-two for efficient FFT computation and better spectral coverage
        self.nfft = 1 << (self.L_timepoints - 1).bit_length()

        # Number of frequency bins for real FFT is N/2 + 1 because of symmetry, the negative frequencies are the complex conjugates of the positive frequencies so they are redundant.
        # Using a min to apply the Nyquist shannon sampling theorem limit, you can't have more frequency bins than the grid allows
        self.n_freqs = min(self.n_freqs, self.nfft // 2 + 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, **self.factory_kwargs),
            nn.SiLU(),
        )

        self.spec_proj = nn.Linear(hidden_dim, self.n_freqs, **self.factory_kwargs)

        # Projection heads to map time-domain vectors -> scalar shifts per head (A, dt, D)
        self.A_shift_proj = nn.Linear(self.L_timepoints, n_heads, **self.factory_kwargs)
        self.dt_shift_proj = nn.Linear(self.L_timepoints, n_heads, **self.factory_kwargs)
        self.D_shift_proj = nn.Linear(self.L_timepoints, n_heads, **self.factory_kwargs)

        self.rank_dim = (self.U_dim + self.V_dim) * self.rank
        
        # First the base and basis for the B_scale and B_shift
        self.B_scale_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.B_scale_basis_proj = nn.Linear(self.L_timepoints, self.rank_dim, **self.factory_kwargs)

        self.B_shift_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.B_shift_basis_proj = nn.Linear(self.L_timepoints, self.rank_dim, **self.factory_kwargs)

        # Then the base and basis for the C_scale and C_shift
        self.C_scale_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.C_scale_basis_proj = nn.Linear(self.L_timepoints, self.rank_dim, **self.factory_kwargs)

        self.C_shift_base_proj = nn.Linear(hidden_dim, self.rank_dim, **self.factory_kwargs)
        self.C_shift_basis_proj = nn.Linear(self.L_timepoints, self.rank_dim, **self.factory_kwargs)

        # Modulation parameter constraints for stability
        self.A_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)
        self.dt_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)
        self.D_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        self.max_A_beta = 0.5

        self.B_alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.B_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        self.C_alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
        self.C_beta = nn.Parameter(torch.tensor(beta_init), requires_grad=True)

        device = self.factory_kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        t = torch.arange(self.L_timepoints, device=device).float().unsqueeze(0)  # (1, L)
        n_freqs = torch.arange(self.n_freqs, device=device).float().unsqueeze(1)  # (F, 1)

        # Using the DCT Type-II basis (cosine basis) for real FFT approximation
        basis = torch.cos(torch.pi * n_freqs * (2 * t + 1) / (2 * float(self.L_timepoints)))  # (F, L)

        # Normalize the first frequency (DC component) differently
        basis[0, :] = basis[0, :] * (1.0 / torch.sqrt(torch.tensor(self.L_timepoints, device=device).float()))
        basis[1:, :] = basis[1:, :] * torch.sqrt(torch.tensor(2.0 / float(self.L_timepoints), device=device))

        self.register_buffer("cosine_basis", basis)  # (F, L)

        self.alpha = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=True)
        self.register_buffer("freqs", torch.arange(self.n_freqs, device=device).float())
    
    def _make_uv(self, base: torch.Tensor, basis: torch.Tensor):
        
        uv: torch.Tensor = base + basis  # (rank_dim,)

        U = uv[: self.U_dim * self.rank].view(self.U_dim, self.rank)
        V = uv[self.U_dim * self.rank :].view(self.rank, self.V_dim)

        return torch.matmul(U, V).reshape(self.U_dim * self.V_dim)
    
    def _spec_to_time(self, spec_weights):

        return torch.matmul(spec_weights, self.cosine_basis)  # (1, T)
    
    def forward(
        self,
        cond: torch.Tensor,
        A_log: torch.Tensor = None,
        B: torch.Tensor = None,
        C: torch.Tensor = None,
        dt: torch.Tensor = None,
        D: torch.Tensor = None,
    ):
        h: torch.Tensor = self.mlp(cond)   # (hidden_dim,)

        # Generate spectral weights with frequency decay
        decay: torch.Tensor = 1.0 / ((self.freqs + 1.0) ** (F.softplus(self.alpha) + 1e-4))   # (n_freqs,)
        spec_weights: torch.Tensor = self.spec_proj(h) * decay   # (n_freqs,)

        time_vec: torch.Tensor = self._spec_to_time(spec_weights) # (1, L_timepoints)

        A_beta: torch.Tensor = self.max_A_beta * torch.sigmoid(self.A_beta)

        A_shift: torch.Tensor = A_beta * torch.tanh(
            self.A_shift_proj(time_vec)
        )
        dt_shift: torch.Tensor = self.dt_beta * torch.tanh(
            self.dt_shift_proj(time_vec)
        )
        D_shift: torch.Tensor = self.D_beta * torch.tanh(
            self.D_shift_proj(time_vec)
        )

        B_scale: torch.Tensor = self._make_uv(
            self.B_scale_base_proj(h),
            self.B_scale_basis_proj(time_vec)
        )

        B_shift: torch.Tensor = self._make_uv(
            self.B_shift_base_proj(h),
            self.B_shift_basis_proj(time_vec)
        )

        B_scale = 1.0 + self.B_alpha * torch.tanh(B_scale)
        B_shift = self.B_beta * torch.tanh(B_shift)

        C_scale: torch.Tensor = self._make_uv(
            self.C_scale_base_proj(h),
            self.C_scale_basis_proj(time_vec)
        )

        C_shift: torch.Tensor = self._make_uv(
            self.C_shift_base_proj(h),
            self.C_shift_basis_proj(time_vec)
        )

        C_scale = 1.0 + self.C_alpha * torch.tanh(C_scale)
        C_shift = self.C_beta * torch.tanh(C_shift)

        # =======================================================================================================================#

        param_list: list = [A_log, B, C, dt, D]

        if all(p is None for p in param_list):
            # return the shifts and scales only
            return A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift
        
        # =======================================================================================================================#

        else:
            A_log: torch.Tensor = A_log + A_shift       # (n_heads,)
            dt: torch.Tensor = dt + dt_shift           # (n_heads,)
            D: torch.Tensor = D + D_shift               # (n_heads,)

            B: torch.Tensor = B * B_scale.expand_as(B) + B_shift.expand_as(B)
            C: torch.Tensor = C * C_scale.expand_as(C) + C_shift.expand_as(C)

            return A_log, B, C, dt, D

## [WIP] Implement CUDA streams for the parallel execution of the adapters. ##
class MarkEnsemble(nn.Module):
    """Ensemble of MaRK adapters for parallelism using CUDA streams.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_adapters: int = 23,
        adapter_class: str = "Hypernet",
        cond_dim: int = 128,
        n_heads: int = 1,
        n_groups: int = 1,
        d_state: int = 1,
        degree: int = 5,
        n_freqs: int = 8,
        L_timepoints: int = 256,
        rank: int = 2,
        hidden_dim: int = 768,
        ddp: bool = True,
        factory_kwargs: dict = None,
    ):
        super().__init__()

        self.num_adapters = num_adapters
        self.adapter_class = adapter_class
        self.rank = rank
        self.degree = degree
        self.L_timepoints = L_timepoints
        self.n_freqs = n_freqs
        self.mark_mlp_dim = hidden_dim

        self.cond_dim = cond_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_state = d_state
        self.degree = degree
        self.rank = rank
        self.hidden_dim = hidden_dim

        self._streams_initialized = False

        if adapter_class == "hypernet":
            self.mod_list = nn.ModuleList([
                Hypernet(
                    cond_dim = self.cond_dim,
                    n_heads = self.n_heads,
                    n_groups = self.n_groups,
                    d_state = self.d_state,
                    rank = self.rank,
                    hidden_dim = self.hidden_dim,
                    factory_kwargs = factory_kwargs,
                ) for _ in range(num_adapters)
            ])

        elif adapter_class == "chebyshev":
            self.mod_list = nn.ModuleList([
                ChebyshevPolynomial(
                    cond_dim = self.cond_dim,
                    n_heads = self.n_heads,
                    n_groups = self.n_groups,
                    d_state = self.d_state,
                    degree = self.degree,
                    rank = self.rank,
                    hidden_dim = self.hidden_dim,
                    factory_kwargs = factory_kwargs,
                ) for _ in range(num_adapters)
            ])

        elif adapter_class == "dct":
            self.mod_list = nn.ModuleList([
                DCTKernel(
                    cond_dim = self.cond_dim,
                    n_heads = self.n_heads,
                    n_groups = self.n_groups,
                    d_state = self.d_state,
                    n_freqs = self.n_freqs,
                    L_timepoints = self.L_timepoints,
                    rank = self.rank,
                    hidden_dim = self.hidden_dim,
                    factory_kwargs = factory_kwargs,
                ) for _ in range(num_adapters)
            ])

        else:
            raise ValueError(f"Unknown adapter class: {adapter_class}")
        
        if ddp == False:
            self.streams = [torch.cuda.Stream() for _ in range(num_adapters)]
            
            self._streams_initialized = True
        else:
            self.streams = None
        
    def forward(
        self,
        cond: torch.Tensor,
    ):
        outputs = []

        if self.streams is not None and self._streams_initialized == True:
            for stream in self.streams:
                cond.record_stream(stream)

            for i, module in enumerate(self.mod_list):
                with torch.cuda.stream(self.streams[i]):
                    out_tuple: tuple = module(cond)

                    # This serves as the synchronization barrier for the backward pass of the gradients
                    processed_out = []
                    for tensor in out_tuple:
                        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                            def get_sync_hook(stream):
                                return lambda grad: torch.cuda.current_stream().wait_stream(stream) or grad
                            tensor.register_hook(get_sync_hook(torch.cuda.current_stream(self.streams[i])))

                        processed_out.append(tensor)

                    outputs.append(tuple(processed_out))

            current_stream = torch.cuda.current_stream()
            for stream in self.streams:
                current_stream.wait_stream(stream)
        else:
            """Fallback to sequential execution if streams are not initialized."""
            for module in self.mod_list:
                outputs.append(module(cond))
                
        assert len(outputs) == self.num_adapters, "Number of outputs does not match number of adapters."

        return outputs