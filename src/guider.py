import torch.nn.functional as F
import torch.nn as nn
import torch

from einops import repeat, rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

class Guider(nn.Module):
    def __init__(
            self,
            d_model=768,    # Token embedding dimension
            d_state=64,     # State matrices dimension
            d_conv=17,      # convolution kernel dimension
            n_groups=1,     # Number of groups for the state matrices
            head_dim=64,    # Dimension of each head
            expand=2,       # expansion factor for d_inner
            dt_min=0.001,   # min. discretization time step
            dt_max=0.1,     # max. discretization time step
            dt_init_floor=1e-4, # initial floor for dt
            dt_limit=(0.0,torch.inf),   # range limit for dt
            learnable_init_states=False,    # whether to learn the initial states
            bias=False,                     # whether to use bias in the linear layers
            activation="swish",             # activation function to use
            conv_bias=True,                 # whether to use bias in the convolution layers 
            conv_init=None,                 # innitialization for the convolution weights
            # Optimization parameters
            chunk_size=256,                 # chunk size for the input sequences
            device=None,                    # Default device initialization
            dtype=None,                     # data type for the model parameters
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype} if device is not None else {'dtype': dtype}

        self.d_model = d_model
        self.d_state = d_state

        self.n_heads = d_model // head_dim
        self.head_dim = head_dim
        self.n_groups = n_groups

        self.expand = expand
        self.d_inner = self.expand * self.d_model

        self.d_conv = d_conv
        self.conv_init = conv_init

        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.bias = bias
        self.chunk_size = chunk_size

        self.logits = None

        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory_kwargs)

        # Declare the convolution dimension and the kernel
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs
        )

        # sample uniformly the initial weights of the convolution layer
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        # set the initialization weights as a parameter to be learnable
        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.n_heads, self.head_dim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.activation = activation.lower()
        self.act = nn.SiLU() if self.activation in ["silu","swish"] else nn.ReLU()
        
        # log uniform sampling of dt and clamp it to the floor
        dt = -torch.exp(
            torch.rand(self.n_heads, **factory_kwargs) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)) + torch.log(torch.tensor(dt_min)))
        )
        dt = torch.clamp(dt, min=dt_init_floor)

        # compute the inverse of dt for numerical stability
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        self.dt_bias._no_weight_decay = True

        # initialize the A matrix as a tensor of ones of size n_heads
        A = torch.ones(
            self.n_heads,
            dtype=torch.float32,
            device=device
        )
        A_log = torch.log(A).to(dtype=dtype)    # Store it as log for numerical stability
        self.A_log = nn.Parameter(A_log)        # declare the A_log as parameter to train on
        self.A_log._no_weight_decay = True

        # D matrix is a learnable parameter, initialized to ones
        self.D = nn.Parameter(
            torch.ones(self.n_heads, device=device)
        )
        self.D._no_weight_decay = True

        # the final output projection layer to get the embedding dimension vector
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias, **factory_kwargs)


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
            self,
            u: torch.Tensor,
            seq_idx: torch.IntTensor=None
    ) -> torch.Tensor:
        """
            input: it will take the unmasked input sequence u from the Hydra model.

            model: it will use the mamba kernel with some addition for binary classification
                    or confidence score prediction.

            output: it will either
            1. return the binary classification of each token (to remask or not)
            2. return the confidence score of each token being remasked

            top_k: applied on the output of the confidence scores to get the remasking vector

            sigmoid 0-1: if thresh >=0.8 1 else 0
        
        """

        # get the batch size, sequence length and dimension of the input tensor shape
        batch_size, _, dim = u.shape

        assert dim == self.d_model, f"Input dimension {dim} does not match model dimension {self.d_model}"

        zxbcdt = self.in_proj(u)    # project the input to the inner dimension
        A = -torch.exp(self.A_log.float())      # get the A matrix from the log space to pass to mamba_chunk_scan_combined
        initial_states = repeat(self.init_states, "... -> b ...", b = batch_size) if self.learnable_init_states else None # repeat the initial states for each batch
        dt_limit_kwargs = {} if self.dt_limit == (0.0, torch.inf) else dict(dt_limit=self.dt_limit)     # the dt_limit kwargs

        # split the zxbcdt tensor into z, xBC, and dt
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.n_groups * self.d_state,
                self.n_heads
            ],
            dim=-1
        )

        # apply the softplus function to dt and add the dt_bias
        dt = F.softplus(dt + self.dt_bias)

        assert torch.isfinite(dt).all() == True, "dt contains non-finite values"
        assert torch.is_neg(dt) == False, "dt contains negative values"

        assert self.activation in ["silu", "swish"]

        # apply the convolution kernel and activation function to the xBC tensor
        self.act(
            self.conv1d(xBC.transpose(1,2)).transpose(1,2)
        )

        # split the xBC tensor into x and BC
        x, BC = torch.split(xBC, [self.d_inner, 2 * self.n_groups * self.d_state], dim=-1)

        # split the BC tensor into B and C
        B, C = torch.split(BC, [self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1)

        # pass the inputs to the mamba_chunk_scan_combined function to get y
        y: torch.Tensor = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", h=self.n_heads),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.n_groups),
            rearrange(C, "b l (g n) -> b l g n", g=self.n_groups),
            D=self.D,
            z=rearrange(z, "b l (h p) -> b l h p", h=self.n_heads),
            chunk_size=self.chunk_size,
            initial_states=initial_states,
            seq_idx=seq_idx,
            **dt_limit_kwargs
        )

        y = rearrange(y, "b l h p -> b l (h p)")    # Unsqueeze the output tensor y

        self.logits: torch.Tensor = self.out_proj(y)        # confidence scores: (batch_size, seq_len, 1)

        return self.logits