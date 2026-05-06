import torch
import torch.nn as nn
import torch.nn.functional as  F

import logging
import os

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from einops import rearrange, repeat
from .utils import log_setup

from .mark import Hypernet, ChebyshevPolynomial, DCTKernel

from .ops import hydra_split_conv1d_scan_combined

LOG_FILE = os.path.join("logs", "hydra.log")
LOG_LEVEL = logging.INFO

logger = log_setup("HydraLogger", LOG_FILE, LOG_LEVEL)

# d_conv must be odd for symmetrical padding
# The seq_len must be 512 if choosing use_eff_compute=True
class Hydra(nn.Module):
    def __init__ (
            self,
            d_model,                # The embedding size/dimension of input/output
            response_len=None,
            d_state=64,             # Internal State dimension (hidden layers)
            d_conv=17,               # Dimension of convolution receptive field before SSM block
            conv_init=None,         # Initialization method for convolution parameters
            head_dim=64 ,           # Dimension of each head of the SSM block
            expand=2,               # The expansion factor for intermediate channnels higher factors = more parameters
            n_groups=1,             # No. of groups in quasi-separable matrix blocks or group convolution
            dt_min=0.001,               # Minimum bound for discretization time constants in SSM
            dt_max=0.1,             # Maximum bound for discretization time constants in SSM
            dt_init_floor=1e-4,     # A small positive floor value for initializing time constants
            dt_limit=(0.0, torch.inf), # A tuple allowed range of max and min of time constants
            learnable_init_states=False, # Whether to make initial hidden states learnable
            activation="swish",          # Activation function choice
            bias=False,              # Introduce Bias terms in linear layers of Hydra
            conv_bias=True,         # Whether pre-SSM conv layer has bias
            mark_kernel: str="hypernet",  # The type of MaRK kernel to use for parameter reconstruction  hypernet | chebyshev | dct
            embedding_dim: int=128,      # The dimension of the conditioning embedding for MaRK (Changed from 256 since only timestep is passed)
            mark_ensemble: bool=True,    # Whether to use ensemble of MaRK adapters for better parallelism using CUDA streams
            rank: int=2,                  # The rank of the low-rank matrices B and C in MaRK hypernet
            degree: int=5,                # The degree of the Chebyshev polynomial in MaRK chebyshev kernel
            L_timepoints: int=256,        # Number of timepoints sampled for the DCT kernel in MaRK (tells the resolution of the kernel)
            n_freqs: int=8,               # Number of frequencies for the Fourier basis table in MaRK hypernet
            mark_mlp_dim: int=256,        # The hidden dimension of the MaRK hypernet MLP
            # Optimization parameters
            chunk_size=256,         # Split input sequences into chunks
            use_eff_compute=False,
            device=None,                    # Default device initialization
            dtype=torch.float32              # Data type of model parameters like torch.float32
    ):
        logger.debug("Initializing Hydra model with parameters")

        factory_kwargs={"device": device, "dtype": dtype} if device is not None else {"dtype": dtype}
        super().__init__()

        assert d_model % head_dim == 0

        self.d_model = d_model
        self.d_state = d_state
        self.response_len = response_len

        self.n_heads = d_model // head_dim
        
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model       # Calculate the size of inner projection layer that projects input x(t)
        self.head_dim = self.d_inner // self.n_heads
        self.n_groups = n_groups

        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.bias = bias
        self.chunk_size = chunk_size

        self.use_eff_compute = use_eff_compute

        self.mark_ensemble = mark_ensemble
        self.embedding_dim = embedding_dim

        if mark_kernel == "hypernet" and not self.mark_ensemble:
            self.mark = Hypernet(
                cond_dim = embedding_dim,
                n_heads = self.n_heads,
                n_groups = self.n_groups,
                d_state = self.d_state,
                rank = rank,
                hidden_dim = mark_mlp_dim,
                factory_kwargs = factory_kwargs,
            )
        elif mark_kernel == "chebyshev" and not self.mark_ensemble:
            self.mark = ChebyshevPolynomial(
                cond_dim = embedding_dim,
                n_heads = self.n_heads,
                n_groups = self.n_groups,
                d_state = self.d_state,
                degree = degree,
                rank = rank,
                hidden_dim = mark_mlp_dim,
                factory_kwargs = factory_kwargs,
            )
        elif mark_kernel == "dct" and not self.mark_ensemble:
            self.mark = DCTKernel(
                cond_dim = embedding_dim,
                n_heads = self.n_heads,
                n_groups = self.n_groups,
                d_state = self.d_state,
                n_freqs = n_freqs,
                L_timepoints = L_timepoints,
                rank = rank,
                hidden_dim = mark_mlp_dim,
                factory_kwargs = factory_kwargs,
            )
        else:
            pass

        ## d_in_proj:https://wandb.ai/ibitec-nvidia/hydra-training/runs/m60v5bey
        # 1. d_inner: The size of z(t) the hidden state from previous iteration
        # 2. d_inner + 2 * (2 * n_groups * d_state): The size of xBC matrices, which are the packed input signal x(t) and the B and C matrices
        # d_inner = x(t) ; B and C = 2 * (2 * n_groups * d_state)
        # 3. 2 * n_heads: The size of dt, which is the discretization time constants for each head
        # The total input dimension is the sum of these three
        # y(t): output signal

        d_in_proj = 2 * self.d_inner + 2 * (2 * self.n_groups * self.d_state) + 2 * self.n_heads    # calculate the input projection dimension for the projection layer
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory_kwargs)

        # A small convolution before the SSM scan.
        # This is done to parameterize the matrices
        conv_dim = self.d_inner + 2 * (2 * self.n_groups * self.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs
        )

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.n_heads, self.head_dim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.GELU() if self.activation == "gelu" else nn.SiLU() if self.activation == "swish" else nn.ReLU()

        ## Initialize the time constants dt and its bias

        # log uniform sampling of dt and clamp/limit it to the min.
        dt = torch.exp(
            torch.rand(self.n_heads, **factory_kwargs) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) + torch.log(torch.tensor(dt_min))
        )
        dt = torch.clamp(dt, min=dt_init_floor)

        # Solve Softplus(x) = dt for x, which is the inverse of dt with numerical stability
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of dt with numerical stability
        self.dt_bias = nn.Parameter(inv_dt)

        self.dt_bias._no_weight_decay = True

        # The A matrix parameter
        A = torch.ones(self.n_heads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # The D or "skip" matrix parameter
        self.D = nn.Parameter(torch.ones(self.n_heads, dtype=torch.float32, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.n_heads, bias=False, **factory_kwargs)
        
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.logits = None

        # Optional: dynamics / Markov analysis — capture pre-MaRK B, C at a fixed token index
        # Set `markov_capture_token_pos` before forward; read `_pre_mark_B`, `_pre_mark_C` after (CPU float32).
        self.markov_capture_token_pos: int | None = None
        self._pre_mark_B: torch.Tensor | None = None
        self._pre_mark_C: torch.Tensor | None = None

        logger.info("Hydra model initialized with parameters")

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
            self, 
            u: torch.Tensor,                    # The input to the SSM model containing z, xBC, dt
            timestep_cond: torch.Tensor,        # The timestep conditioning input for MaRK      Shape: (embedding_dim,)
            seq_idx: torch.Tensor=None,
            param_update: tuple=None,
    ) -> torch.Tensor:
        """
        The forward pass of the core SSM kernel of the Hydra model. This forward pass projects to a compatible 
        latent space using a linear layer, splits the latent vector to a skip parameter z, a packed input signal xBC, and a discretization time constant dt.
        The packed input signal is passed through a convolution before being split into the input signal x(t) and the B and C matrices.
        The mamba_ssd_chunk_scan_combined kernel is applied on the new signals and then the output is manually gated with the z parameter,
        and passed to the out_proj layer back to input.

        Args:
            u (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model) containing the input signal to the SSM model.
            seq_idx (int, optional): The sequence index for the SSM scan. Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model) after passing through the SSM model.
        """

        # u is the 1D input sequence of shape (batch_size, seq_len, d_model)

        # We map u(t) to an N-D latent space x(t) using a learned projection
        # x(t) will be mapped to a 1D output vector y(t) using a linear layer

        ## The SSM model is defined as follows:
        # h'(t) = A * h(t) + B * x(t)       ----    (1)
        # y(t) = C * h'(t) + D * x(t)       ----    (2)

        seq_idx=None

        batch_size, _, dim = u.shape

        assert dim == self.d_model, f"Input dimension {dim} does not match model dimension {self.d_model}"

        ## The projected zxBCdt vector from the input z, xBC and dt
        zxbcdt: torch.Tensor = self.in_proj(u)

        assert timestep_cond.shape[-1] == self.embedding_dim, f"timestep_cond dim {timestep_cond.shape[-1]} != embedding_dim {self.embedding_dim}"

        # Shape: (embedding_dim,)
        cond_embedding: torch.Tensor = timestep_cond if not self.mark_ensemble else None

        initial_states: torch.Tensor = repeat(self.init_states, "... -> b ...", b=2*batch_size) if self.learnable_init_states else None

        dt_limit_kwargs = {} if self.dt_limit == (0.0, torch.inf) else dict(dt_limit=self.dt_limit)

        if self.use_eff_compute:
            A: torch.Tensor = -torch.exp(self.A_log.float())          # Get the A matrix from the log space

            # [WIP] Maybe MaRK support will help or allow efficient compute mode soon using Ensemble of MaRK adapters
            assert self.mark is None, "MaRK support not available for efficient compute mode yet"

            logits: torch.Tensor = hydra_split_conv1d_scan_combined(
                zxbcdt,
                self.conv1d.weight,
                self.conv1d.bias,
                self.dt_limit,
                self.dt_bias,
                A,
                self.fc_D.weight,
                self.D,
                self.norm.weight,
                self.norm.eps,
                self.out_proj.weight,
                self.out_proj.bias,
                self.chunk_size,
                initial_states,
                seq_idx,
                self.d_inner,
                self.d_state,
                self.head_dim,
                self.n_groups,
            )
            
            self.logits = logits

            return logits

        # z = Gating
        # xBC = the packed input signal x and the B and C matrices
        # dt = Discretization time constants
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,   # Dimension of the hidden state z(t)
                self.d_inner + 2 * (2 * self.n_groups * self.d_state), # Dimension of the packed input signal x(t) and the B and C matrices
                2 * self.n_heads    # Dimension of the discretization time constants dt
            ],
            dim=-1
        )

        assert self.activation in ["silu", "swish", "relu", "gelu"]

        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )

        # Unpack the input signal x(t) from BC matrices
        # x(t) = self.d_inner (projected dimension of the input signal x(t))
        # BC = 2 * (2 * n_groups * d_state) (packed B and C matrices)
        x, BC = torch.split(xBC, [self.d_inner, 2 * (2 * self.n_groups * self.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)  # x(t) to the forward and reverse sequence

        # logger.debug(f"x shape: {x.shape}, x_og shape: {x_og.shape}, BC shape: {BC.shape}")

        # Concatenate the upper and lower halves of the BC matrices after reversing the upper half
        BC = torch.cat(
            (BC[:, :, :2 * self.n_groups * self.d_state],
             torch.flip(BC[:, :, 2 * self.n_groups * self.d_state:], (1,))),
             dim=0
        )
        # logger.debug(f"BC shape after concatenation: {BC.shape}")

        # Unpack the B and C matrices from BC
        # B = The learnable low-rank matrix for state update = Input projection to internal hidden state of SSM
        # C = The learnable low-rank matrix for the state readout = Projection of hidden state to the output vector
        B, C = torch.split(BC, [self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1)

        # Pre-MaRK capture for analysis: shapes (B, L, n_groups * d_state); forward stream is batch index 0 when batch_size==1
        if self.markov_capture_token_pos is not None:
            p = self.markov_capture_token_pos
            if 0 <= p < B.shape[1]:
                self._pre_mark_B = B[0, p].detach().float().cpu().clone()
                self._pre_mark_C = C[0, p].detach().float().cpu().clone()
            else:
                self._pre_mark_B = None
                self._pre_mark_C = None

        # explore using different timestep conditioning within a batch for training here. It may mean repeated mamba kernel launches but will stabilize training even more.
        if not self.mark_ensemble:
            A_log, B, C, dt_bias, D = self.mark.forward(
                cond=cond_embedding,
                A_log=self.A_log,
                B = B,
                C = C,
                dt=self.dt_bias,
                D = self.D,
            )
        else:
            A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift = param_update

            A_log = self.A_log + A_shift
            dt_bias = self.dt_bias + dt_shift
            D = self.D + D_shift

            B_scale = B_scale.expand_as(B) if B_scale.shape != B.shape else B_scale
            B_shift = B_shift.expand_as(B) if B_shift.shape != B.shape else B_shift
            C_scale = C_scale.expand_as(C) if C_scale.shape != C.shape else C_scale
            C_shift = C_shift.expand_as(C) if C_shift.shape != C.shape else C_shift

            B = B * B_scale + B_shift
            C = C * C_scale + C_shift


        # dt = controls the discretization for the continuous SSM using ZOH method (Zero-Order Hold) keeping input constant over dt
        dt = torch.cat((dt[:, :, :self.n_heads], torch.flip(dt[:, :, self.n_heads:], (1,))), dim=0)     # Concatenate the upper and lower halves after reversing upper half
        dt = F.softplus(dt + dt_bias)

        A: torch.Tensor = -torch.exp(A_log.float())          # Get the A matrix from the log space

        # Double seq_idx to match the doubled batch (forward || reversed).
        # The cat + flip mirrors how x / BC / dt are doubled above.
        # Keeping int32 is required by the Triton SSD kernel.
        # if seq_idx is not None:
        #     seq_idx_rev = torch.flip(seq_idx, (1,))
        #     seq_idx = torch.cat((seq_idx.to(torch.int32), seq_idx_rev.to(torch.int32)), dim=0)

        # NOTE: if we have different parameters for each sequence within a batch we dont necessarily need to iterate. we can have 1 kernel launch.
        # B, C, and dt can already accept unique parameters for each sequence, except for A.
        # but we can apply the log scale shift to A, via the dt parameter.
        # After the softplus of dt, multiply it by the torch.exp(A_log_shift) to get the exact same effect as shifting A in log space.
        y: torch.Tensor = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", h=self.n_heads),      # Unsqueeze x to distribute among heads
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.n_groups),      # Unsqueeze B matrix to distribute among groups
            rearrange(C, "b l (g n) -> b l g n", g=self.n_groups),      # Unsqueeze C matrix to distribute among groups
            chunk_size=self.chunk_size, 
            D=None,
            z=None,
            seq_idx=None,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
                
        y = rearrange(y, "b l h p -> b l (h p)")    # Squeeze output vector 'y' 
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_lower, y_upper = y[:batch_size], torch.flip(y[batch_size:], (1,))     # Split y into lower and upper halves

        # y(t) = C * x(t) + D * u(t)

        # x(t) = y_lower + y_upper = combine the lower and upper halves of the output
        # D = repeat(F.linear(...)) = calculate the gating/skip vector for each token in each head
        # u(t) = x_og * repeat(F.linear(...)) = scale the residual skip connection

        y = y_lower + y_upper + x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=D), "b l h -> b l (h p)", p=self.d_inner // self.n_heads
        )

        y = self.norm(y, z)     # Apply the z gating mechanism manually to the output y
        logits = self.out_proj(y)  # Map the output to the logits

        return logits