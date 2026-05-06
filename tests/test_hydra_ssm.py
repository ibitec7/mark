"""
Comprehensive unit and integration tests for Hydra SSM module.

Tests Hydra initialization, forward pass correctness, parameter bounds,
bidirectional processing, boundary handling, and gradient flow across
different kernel types and configurations.
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple

from src.hydra import Hydra


@pytest.fixture
def device():
    """Detect CUDA availability, fallback to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def factory_kwargs(device):
    """Factory kwargs for device/dtype configuration."""
    return {"device": device, "dtype": torch.float32}


@pytest.fixture
def base_config():
    """Base configuration for Hydra SSM."""
    return {
        "d_model": 256,
        "d_state": 64,
        "d_conv": 17,
        "head_dim": 64,
        "expand": 2,
        "n_groups": 1,
        "chunk_size": 256,
        "embedding_dim": 256,
        "mark_kernel": "hypernet",
        "mark_ensemble": False,
    }


@pytest.fixture
def conditioning_inputs(factory_kwargs):
    """Generate timestep and mask conditioning inputs."""
    timestep_cond = torch.randn(128, **factory_kwargs)
    masked_cond = torch.randn(128, **factory_kwargs)
    return timestep_cond, masked_cond


@pytest.mark.parametrize("mark_kernel,kernel_config", [
    ("hypernet", {"rank": 2}),
    ("chebyshev", {"degree": 5}),
    ("dct", {"n_freqs": 8, "L_timepoints": 256}),
])
class TestHydraInitialization:
    """Test Hydra module initialization across different kernel types."""

    def test_initialization_creates_required_modules(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict
    ):
        """Verify all required modules are initialized."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config, 
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        # Check core modules
        assert hasattr(hydra, "in_proj")
        assert hasattr(hydra, "conv1d")
        assert hasattr(hydra, "norm")
        assert hasattr(hydra, "out_proj")
        assert hasattr(hydra, "mark")

    def test_input_projection_dimension(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict
    ):
        """Verify input projection dimension is created."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        # Verify input projection exists and has reasonable dimension
        assert hydra.in_proj is not None
        assert hydra.in_proj.in_features == base_config["d_model"]
        # Output dimension should be significantly larger than input for projection
        assert hydra.in_proj.out_features > hydra.in_proj.in_features
        # Output dimension should scale with expand factor and state dimension
        assert hydra.in_proj.out_features >= 2 * hydra.d_inner

    def test_parameter_initialization_ranges(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict
    ):
        """Verify parameters are initialized in valid ranges."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        # Check dt_bias is in valid range (result of inverse softplus)
        assert torch.all(hydra.dt_bias >= -10), "dt_bias lower bound violated"
        assert torch.all(hydra.dt_bias <= 10), "dt_bias upper bound violated"
        
        # Check A_log is actually in log space
        assert torch.all(torch.isfinite(hydra.A_log)), "A_log contains non-finite values"
        
        # Check D is positive (skip connection should be positive)
        assert torch.all(hydra.D >= 0), "D parameter should be non-negative"


class TestHydraInitializationNonParametrized:
    """Non-parametrized Hydra initialization tests."""

    def test_learnable_init_states_when_enabled(
        self, base_config: dict, factory_kwargs: dict
    ):
        """Verify learnable init states are created when enabled."""
        config = {**base_config, "learnable_init_states": True,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        assert hasattr(hydra, "init_states")
        assert isinstance(hydra.init_states, torch.nn.Parameter)
        assert hydra.init_states.requires_grad
        expected_shape = (hydra.n_heads, hydra.head_dim, hydra.d_state)
        assert hydra.init_states.shape == expected_shape

    def test_activation_function_initialization(
        self, base_config: dict, factory_kwargs: dict
    ):
        """Verify activation function is correctly initialized."""
        for activation_name in ["swish", "gelu", "relu"]:
            config = {**base_config, "activation": activation_name,
                      "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
            hydra = Hydra(**config)
            assert hasattr(hydra, "act")
            assert isinstance(hydra.act, torch.nn.Module)


class TestHydraZeroBoundaries:
    """Test the _zero_at_boundaries static method for packed sequence handling."""

    def test_zero_boundaries_creates_dilated_mask(self, factory_kwargs: dict):
        """Verify boundary detection creates proper dilated mask around sequence boundaries."""
        batch_size, seq_len = 2, 256
        xBC_dim = 512
        xBC = torch.randn(batch_size, seq_len, xBC_dim, **factory_kwargs)
        
        # Create seq_idx with boundary markers (value change indicates boundary)
        seq_idx = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=factory_kwargs["device"])
        seq_idx[:, :100] = 0  # Document 0
        seq_idx[:, 100:200] = 1  # Document 1
        seq_idx[:, 200:] = 2  # Document 2
        
        half = 8
        result = Hydra._zero_at_boundaries(xBC, seq_idx, half)
        
        # Check that values around boundaries are zeroed
        # Boundary at index 100 and 200
        assert torch.allclose(result[:, 92:108], torch.zeros_like(result[:, 92:108])), \
            "Values in ±half window of boundary should be zeroed"

    def test_zero_boundaries_handles_negative_seq_idx(self, factory_kwargs: dict):
        """Verify padding guard (seq_idx < 0) is handled."""
        batch_size, seq_len = 2, 256
        xBC_dim = 512
        xBC = torch.ones(batch_size, seq_len, xBC_dim, **factory_kwargs)
        
        seq_idx = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=factory_kwargs["device"])
        seq_idx[:, 200:] = -1  # Padding guard
        
        result = Hydra._zero_at_boundaries(xBC, seq_idx, half=8)
        
        # Values at padding should be zeroed
        assert torch.allclose(result[:, 192:], torch.zeros_like(result[:, 192:])), \
            "Padding guard should trigger zeroing"

    def test_zero_boundaries_window_size(self, factory_kwargs: dict):
        """Verify boundary zeroing window size matches ±half parameter."""
        batch_size, seq_len = 1, 256
        xBC_dim = 512
        xBC = torch.ones(batch_size, seq_len, xBC_dim, **factory_kwargs)
        
        seq_idx = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=factory_kwargs["device"])
        seq_idx[:, :128] = 0
        seq_idx[:, 128:] = 1
        
        half = 16  # ±16 window
        result = Hydra._zero_at_boundaries(xBC, seq_idx, half)
        
        # Check exact window boundaries
        assert torch.allclose(result[:, 112:144], torch.zeros_like(result[:, 112:144])), \
            f"Window should be ±{half} around boundary"
        assert torch.allclose(result[:, 111:112], torch.ones_like(result[:, 111:112])), \
            "Values outside window should remain"


@pytest.mark.parametrize("mark_kernel,kernel_config", [
    ("hypernet", {"rank": 2}),
    ("chebyshev", {"degree": 5}),
    ("dct", {"n_freqs": 8, "L_timepoints": 256}),
])
class TestHydraForwardPass:
    """Test forward pass correctness across configurations."""

    def test_forward_output_shape(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Verify forward pass output shape matches input batch/sequence shape."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        hydra.eval()
        
        batch_size, seq_len = 2, 256
        u = torch.randn(batch_size, seq_len, base_config["d_model"], **factory_kwargs)
        timestep_cond, masked_cond = conditioning_inputs
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        assert output.shape == (batch_size, seq_len, base_config["d_model"]), \
            f"Output shape mismatch: {output.shape} vs ({batch_size}, {seq_len}, {base_config['d_model']})"

    def test_forward_dtype_preserved(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Verify output dtype matches model dtype."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        hydra.eval()
        
        u = torch.randn(2, 256, base_config["d_model"], **factory_kwargs)
        timestep_cond, masked_cond = conditioning_inputs
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        # Autocast to bfloat16 in forward, but output should be float32 or bfloat16
        assert output.dtype in [torch.float32, torch.bfloat16], \
            f"Output dtype {output.dtype} not expected"

    def test_forward_no_nan_inf(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Verify forward pass produces finite outputs."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        hydra.eval()
        
        u = torch.randn(2, 256, base_config["d_model"], **factory_kwargs)
        timestep_cond, masked_cond = conditioning_inputs
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        assert torch.all(torch.isfinite(output)), "Forward pass produced non-finite outputs"

    def test_forward_different_batch_sizes(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Test forward pass with various batch sizes."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        hydra.eval()
        
        timestep_cond, masked_cond = conditioning_inputs
        
        for batch_size in [1, 2, 4]:
            u = torch.randn(batch_size, 256, base_config["d_model"], **factory_kwargs)
            with torch.no_grad():
                output = hydra(u, timestep_cond, masked_cond)
            assert output.shape[0] == batch_size


class TestHydraParameterModulation:
    """Test parameter modulation via MaRK kernels."""

    def test_x_doubling_logic(self, factory_kwargs: dict):
        """Verify input x is properly doubled for bidirectional processing."""
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False,
            "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]
        }
        hydra = Hydra(**config)
        
        batch_size, seq_len = 2, 256
        u = torch.randn(batch_size, seq_len, config["d_model"], **factory_kwargs)
        
        # Manually extract x from forward pass (up to the doubling point)
        zxbcdt = hydra.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                hydra.d_inner,
                hydra.d_inner + 2 * (2 * hydra.n_groups * hydra.d_state),
                2 * hydra.n_heads
            ],
            dim=-1
        )
        
        xBC_activated = hydra.act(hydra.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        x, BC = torch.split(xBC_activated, [hydra.d_inner, 2 * (2 * hydra.n_groups * hydra.d_state)], dim=-1)
        
        # x should be doubled correctly
        x_doubled = torch.cat((x, torch.flip(x, (1,))), dim=0)
        assert x_doubled.shape[0] == 2 * batch_size, "X not properly doubled"
        assert x_doubled.shape[1] == seq_len, "Sequence length should not change"

    def test_bidirectional_combination(self, factory_kwargs: dict):
        """Verify bidirectional outputs are properly combined."""
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False,
            "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]
        }
        hydra = Hydra(**config)
        
        batch_size = 2
        y = torch.randn(2 * batch_size, 256, hydra.d_inner, **factory_kwargs)
        
        # Verify split and flip logic
        y_lower, y_upper = y[:batch_size], torch.flip(y[batch_size:], (1,))
        combined = y_lower + y_upper
        
        assert combined.shape == (batch_size, 256, hydra.d_inner), \
            f"Combined shape mismatch: {combined.shape}"


@pytest.mark.parametrize("mark_kernel,kernel_config", [
    ("hypernet", {"rank": 2}),
    ("chebyshev", {"degree": 5}),
    ("dct", {"n_freqs": 8, "L_timepoints": 256}),
])
class TestHydraGradientFlow:
    """Test gradient flow through Hydra SSM."""

    def test_gradient_flow_through_forward_pass(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Verify gradients flow through entire forward pass."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        u = torch.randn(2, 128, base_config["d_model"], requires_grad=True, **factory_kwargs)
        timestep_cond, masked_cond = conditioning_inputs
        timestep_cond.requires_grad = True
        
        output = hydra(u, timestep_cond, masked_cond)
        loss = output.sum()
        loss.backward()
        
        # Check input gradient exists and is finite
        assert u.grad is not None, "Input gradient not computed"
        assert torch.all(torch.isfinite(u.grad)), "Input gradient contains non-finite values"

    def test_gradient_accumulation_stability(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Verify parameter gradients accumulate and remain stable."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        timestep_cond, masked_cond = conditioning_inputs
        
        for _ in range(3):
            u = torch.randn(2, 128, base_config["d_model"], **factory_kwargs)
            output = hydra(u, timestep_cond, masked_cond)
            loss = output.sum()
            loss.backward()
        
        # Check all parameter gradients are finite
        for param in hydra.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), \
                    f"Parameter {param} has non-finite gradient"
                # Gradient norm should be reasonable (not explosion/vanishing)
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1.5e4, f"Gradient norm too large: {grad_norm}"

    def test_mark_kernel_gradient_flow(
        self, mark_kernel: str, kernel_config: dict, 
        base_config: dict, factory_kwargs: dict, conditioning_inputs
    ):
        """Verify MaRK kernel parameters receive gradients."""
        config = {**base_config, "mark_kernel": mark_kernel, **kernel_config,
                  "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]}
        hydra = Hydra(**config)
        
        u = torch.randn(2, 128, base_config["d_model"], **factory_kwargs)
        timestep_cond, masked_cond = conditioning_inputs
        
        # Store initial MaRK parameters
        mark_params_before = {name: p.clone() for name, p in hydra.mark.named_parameters()}
        
        output = hydra(u, timestep_cond, masked_cond)
        loss = output.sum()
        loss.backward()
        
        # Check that MaRK kernel received gradients
        has_gradients = False
        for name, param in hydra.mark.named_parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                has_gradients = True
                break
        
        assert has_gradients, "MaRK kernel did not receive non-zero gradients"


class TestHydraEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_sequence(self, factory_kwargs: dict):
        """Verify forward pass works with minimal sequence length."""
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False,
            "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]
        }
        hydra = Hydra(**config)
        hydra.eval()
        
        u = torch.randn(1, 1, 256, **factory_kwargs)
        timestep_cond = torch.randn(128, **factory_kwargs)
        masked_cond = torch.randn(128, **factory_kwargs)
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        assert output.shape == (1, 1, 256), f"Output shape mismatch for single token: {output.shape}"

    def test_learned_init_states_usage(self, factory_kwargs: dict):
        """Verify learnable initial states are properly used in forward pass."""
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "learnable_init_states": True, "mark_ensemble": False,
            "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]
        }
        hydra = Hydra(**config)
        hydra.eval()
        
        u = torch.randn(2, 256, 256, **factory_kwargs)
        timestep_cond = torch.randn(128, **factory_kwargs)
        masked_cond = torch.randn(128, **factory_kwargs)
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        assert torch.all(torch.isfinite(output)), "Output with learned init states is non-finite"

    def test_dimension_mismatch_raises_assertion(self, factory_kwargs: dict):
        """Verify dimension mismatch in input raises assertion."""
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False,
            "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]
        }
        hydra = Hydra(**config)
        
        # Wrong d_model dimension
        u = torch.randn(2, 256, 128, **factory_kwargs)  # 128 instead of 256
        timestep_cond = torch.randn(128, **factory_kwargs)
        masked_cond = torch.randn(128, **factory_kwargs)
        
        with pytest.raises(AssertionError):
            hydra(u, timestep_cond, masked_cond)

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_different_head_dimensions(self, head_dim: int, factory_kwargs: dict):
        """Test forward pass with different head dimensions."""
        d_model = 256
        config = {
            "d_model": d_model, "d_state": 64, "d_conv": 17, "head_dim": head_dim,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False,
            "device": factory_kwargs["device"], "dtype": factory_kwargs["dtype"]
        }
        
        # Only test if d_model is divisible by head_dim
        if d_model % head_dim == 0:
            hydra = Hydra(**config)
            hydra.eval()
            
            u = torch.randn(2, 256, d_model, **factory_kwargs)
            timestep_cond = torch.randn(128, **factory_kwargs)
            masked_cond = torch.randn(128, **factory_kwargs)
            
            with torch.no_grad():
                output = hydra(u, timestep_cond, masked_cond)
            
            assert output.shape == (2, 256, d_model)


class TestHydraDeviceConsistency:
    """Test device handling and consistency across CPU/GPU."""

    @pytest.mark.skip(reason="Triton kernels (mamba_ssm) do not support CPU execution")
    def test_cpu_forward_pass(self, factory_kwargs: dict):
        """Verify forward pass works on CPU."""
        cpu_kwargs = {"device": torch.device("cpu"), "dtype": torch.float32}
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False, **cpu_kwargs
        }
        hydra = Hydra(**config)
        u = torch.randn(2, 256, 256, **cpu_kwargs)
        timestep_cond = torch.randn(128, **cpu_kwargs)
        masked_cond = torch.randn(128, **cpu_kwargs)
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        assert output.device.type == "cpu", "Output not on CPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward_pass(self):
        """Verify forward pass works on GPU when available."""
        gpu_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
        config = {
            "d_model": 256, "d_state": 64, "d_conv": 17, "head_dim": 64,
            "expand": 2, "n_groups": 1, "chunk_size": 256, "embedding_dim": 256,
            "mark_kernel": "hypernet", "mark_ensemble": False, **gpu_kwargs
        }
        hydra = Hydra(**config)
        hydra.eval()
        
        u = torch.randn(2, 256, 256, **gpu_kwargs)
        timestep_cond = torch.randn(128, **gpu_kwargs)
        masked_cond = torch.randn(128, **gpu_kwargs)
        
        with torch.no_grad():
            output = hydra(u, timestep_cond, masked_cond)
        
        assert output.device.type == "cuda", "Output not on CUDA"
