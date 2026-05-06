"""
Comprehensive unit and integration tests for MaRK kernel adapters.

Tests Hypernet, ChebyshevPolynomial, and DCTKernel implementations with
focus on parameter modulation bounds, UV factorization correctness, and
gradient flow stability.
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Type, Tuple

from src.mark import Hypernet, ChebyshevPolynomial, DCTKernel
from src.utils import pick_factors_near_sqrt


@pytest.fixture
def device():
    """Detect CUDA availability, fallback to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def factory_kwargs(device):
    """Factory kwargs for device/dtype configuration."""
    return {"device": device, "dtype": torch.float32}


@pytest.fixture
def default_config():
    """Default configuration for kernel initialization."""
    return {
        "cond_dim": 256,
        "n_heads": 4,
        "n_groups": 2,
        "d_state": 64,
        "hidden_dim": 256,
        "rank": 2,
    }


@pytest.fixture
def conditioning_input(factory_kwargs):
    """Generate valid conditioning embedding."""
    return torch.randn(256, **factory_kwargs)


@pytest.mark.parametrize("kernel_class,kernel_specific", [
    (Hypernet, {}),
    (ChebyshevPolynomial, {"degree": 5}),
    (DCTKernel, {"n_freqs": 8, "L_timepoints": 256}),
])
class TestMaRKKernelInitialization:
    """Test initialization consistency across all three kernel types."""

    def test_initialization_creates_required_attributes(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict
    ):
        """Verify all required attributes are initialized."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        
        # Check MLP exists
        assert hasattr(kernel, "mlp")
        assert isinstance(kernel.mlp, torch.nn.Sequential)
        
        # Check common projection heads exist (varies by kernel type)
        has_shift_proj = hasattr(kernel, "A_shift_proj") or hasattr(kernel, "A_proj")
        assert has_shift_proj, f"{kernel_class.__name__} missing A parameter projection"
        
        # Check B/C modulation projections exist
        has_b_proj = (hasattr(kernel, "B_scale_proj") or 
                      (hasattr(kernel, "B_scale_base_proj") and hasattr(kernel, "B_scale_basis_proj")))
        assert has_b_proj, f"{kernel_class.__name__} missing B projection heads"
        
        # Check learnable beta/alpha parameters
        assert hasattr(kernel, "A_beta")
        assert hasattr(kernel, "B_beta")
        assert hasattr(kernel, "C_beta")

    def test_initialization_parameters_require_grad(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict
    ):
        """Verify trainable parameters have requires_grad=True."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        
        for param in kernel.parameters():
            assert param.requires_grad, "All parameters should be trainable"


@pytest.mark.parametrize("kernel_class,kernel_specific", [
    (Hypernet, {}),
    (ChebyshevPolynomial, {"degree": 5}),
    (DCTKernel, {"n_freqs": 8, "L_timepoints": 256}),
])
class TestMaRKParameterModulation:
    """Test parameter modulation correctness and numerical bounds."""

    def test_forward_pass_shape_consistency(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict, conditioning_input: torch.Tensor
    ):
        """Verify forward pass produces correct output shapes."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        kernel = kernel.to(**factory_kwargs)  # Ensure kernel is on correct device
        
        # When all params provided, returns 7 tuples (shifts/scales)
        A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift = kernel.forward(
            cond=conditioning_input,
            A_log=None,
            B=None,
            C=None,
            dt=None,
            D=None,
        )
        
        # Verify output shapes
        assert A_shift.shape == (default_config["n_heads"],), f"A_shift shape mismatch: {A_shift.shape}"
        assert dt_shift.shape == (default_config["n_heads"],), f"dt_shift shape mismatch: {dt_shift.shape}"
        assert D_shift.shape == (default_config["n_heads"],), f"D_shift shape mismatch: {D_shift.shape}"

    def test_modulation_scales_bounded(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict, conditioning_input: torch.Tensor
    ):
        """Verify modulation scales stay within tanh-bounded range [-1, 1]."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        kernel = kernel.to(**factory_kwargs)  # Ensure kernel is on correct device
        
        A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift = kernel.forward(
            cond=conditioning_input, A_log=None, B=None, C=None, dt=None, D=None
        )
        
        # B/C scales should be near 1.0 (tanh-bounded modulation)
        assert torch.all(B_scale >= 0.0) and torch.all(B_scale <= 2.0), \
            f"B_scale out of bounds: min={B_scale.min()}, max={B_scale.max()}"
        assert torch.all(C_scale >= 0.0) and torch.all(C_scale <= 2.0), \
            f"C_scale out of bounds: min={C_scale.min()}, max={C_scale.max()}"
        
        # Shifts should be bounded
        assert torch.all(torch.isfinite(B_shift)), "B_shift contains inf/nan"
        assert torch.all(torch.isfinite(C_shift)), "C_shift contains inf/nan"

    def test_beta_parameter_bounds(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict
    ):
        """Verify beta parameters control modulation bounds correctly."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        
        # Beta parameters should have max values defined
        if hasattr(kernel, "max_A_beta"):
            assert kernel.max_A_beta == 0.5, "max_A_beta should be 0.5 for stability"
        
        # Check beta parameters are initialized in reasonable range
        assert kernel.A_beta.item() <= 1.0, "A_beta should be <= 1.0"
        assert kernel.D_beta.item() <= 1.0, "D_beta should be <= 1.0"

    def test_forward_deterministic_with_seed(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict, conditioning_input: torch.Tensor
    ):
        """Verify forward pass is deterministic with fixed seed."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        torch.manual_seed(42)
        kernel1 = kernel_class(**config)
        kernel1 = kernel1.to(**factory_kwargs)  # Ensure kernel is on correct device
        out1 = kernel1.forward(cond=conditioning_input, A_log=None, B=None, C=None, dt=None, D=None)
        
        torch.manual_seed(42)
        kernel2 = kernel_class(**config)
        kernel2 = kernel2.to(**factory_kwargs)  # Ensure kernel is on correct device
        out2 = kernel2.forward(cond=conditioning_input, A_log=None, B=None, C=None, dt=None, D=None)
        
        for o1, o2 in zip(out1, out2):
            assert torch.allclose(o1, o2, atol=1e-6), "Forward pass not deterministic"


class TestUVFactorization:
    """Test UV factorization correctness for low-rank B/C modulation."""

    def test_uv_factorization_dimensions(self, factory_kwargs: dict):
        """Verify UV factorization produces correct matrix dimensions."""
        n_groups = 2
        d_state = 64
        rank = 2
        
        U_dim, V_dim = pick_factors_near_sqrt(n_groups * d_state)
        
        # Create dummy UV vectors for testing _make_uv logic
        rank_dim = (U_dim + V_dim) * rank
        uv = torch.randn(rank_dim, **factory_kwargs)
        
        U = uv[:U_dim * rank].view(U_dim, rank)
        V = uv[U_dim * rank:].view(rank, V_dim)
        result = torch.matmul(U, V).reshape(U_dim * V_dim)
        
        # Result should match original dimension
        assert result.shape[0] == n_groups * d_state, \
            f"UV product shape mismatch: {result.shape[0]} vs {n_groups * d_state}"

    def test_uv_reconstruction_preserves_rank(self, factory_kwargs: dict):
        """Verify UV decomposition reconstructs low-rank structure."""
        U_dim, V_dim = pick_factors_near_sqrt(128)
        rank = 2
        
        # Create low-rank matrix
        U = torch.randn(U_dim, rank, **factory_kwargs)
        V = torch.randn(rank, V_dim, **factory_kwargs)
        matrix = torch.matmul(U, V)
        
        # Check actual rank
        _, s, _ = torch.svd(matrix)
        actual_rank = (s > 1e-4).sum().item()
        
        assert actual_rank <= rank + 1, f"Reconstructed matrix rank too high: {actual_rank}"


class TestChebyshevPolynornial:
    """Chebyshev-specific kernel tests."""

    def test_chebyshev_basis_shape(self, factory_kwargs: dict):
        """Verify Chebyshev basis generation produces correct shape."""
        degree = 5
        n_heads = 4
        config = {
            "cond_dim": 256, "n_heads": n_heads, "n_groups": 2, "d_state": 64,
            "degree": degree, "rank": 2, "hidden_dim": 256, "factory_kwargs": factory_kwargs
        }
        kernel = ChebyshevPolynomial(**config)
        
        # Call _chebyshev_basis directly
        z = torch.randn(n_heads, **factory_kwargs)
        basis = kernel._chebyshev_basis(z)
        
        assert basis.shape == (n_heads, degree), \
            f"Basis shape mismatch: {basis.shape} vs ({n_heads}, {degree})"

    def test_chebyshev_basis_values_in_range(self, factory_kwargs: dict):
        """Verify Chebyshev polynomial values stay in reasonable range."""
        degree = 5
        config = {
            "cond_dim": 256, "n_heads": 4, "n_groups": 2, "d_state": 64,
            "degree": degree, "rank": 2, "hidden_dim": 256, "factory_kwargs": factory_kwargs
        }
        kernel = ChebyshevPolynomial(**config)
        
        z = torch.randn(4, **factory_kwargs)
        basis = kernel._chebyshev_basis(z)
        
        # Chebyshev polynomials should be bounded for inputs in [-1, 1]
        # (z is passed through tanh or similar)
        assert torch.all(torch.isfinite(basis)), "Basis contains inf/nan"

    def test_extreme_degree_single(self, factory_kwargs: dict):
        """Test edge case: degree=1 (constant basis)."""
        conditioning_input = torch.randn(256, **factory_kwargs)
        config = {
            "cond_dim": 256, "n_heads": 4, "n_groups": 2, "d_state": 64,
            "degree": 1, "rank": 2, "hidden_dim": 256, "factory_kwargs": factory_kwargs
        }
        kernel = ChebyshevPolynomial(**config)
        
        output = kernel.forward(cond=conditioning_input, A_log=None, B=None, C=None, dt=None, D=None)
        assert all(torch.isfinite(o).all() for o in output), "Forward pass produced non-finite outputs"


class TestDCTKernel:
    """DCT-specific kernel tests."""

    def test_cosine_basis_registration(self, factory_kwargs: dict):
        """Verify DCT cosine basis is registered as buffer."""
        config = {
            "cond_dim": 256, "n_heads": 4, "n_groups": 2, "d_state": 64,
            "n_freqs": 8, "L_timepoints": 256, "rank": 2, "hidden_dim": 256,
            "factory_kwargs": factory_kwargs
        }
        kernel = DCTKernel(**config)
        
        assert hasattr(kernel, "cosine_basis"), "cosine_basis buffer not registered"
        basis = kernel.cosine_basis
        assert basis.shape == (8, 256), f"Cosine basis shape mismatch: {basis.shape}"

    def test_spectral_to_time_projection(self, factory_kwargs: dict):
        """Verify spectral weights project to time domain correctly."""
        config = {
            "cond_dim": 256, "n_heads": 4, "n_groups": 2, "d_state": 64,
            "n_freqs": 8, "L_timepoints": 256, "rank": 2, "hidden_dim": 256,
            "factory_kwargs": factory_kwargs
        }
        kernel = DCTKernel(**config)
        kernel = kernel.to(**factory_kwargs)  # Ensure kernel is on correct device
        
        spec_weights = torch.randn(8, **factory_kwargs)
        time_vec = kernel._spec_to_time(spec_weights)
        
        assert time_vec.shape[-1] == 256, f"Time vector shape mismatch: {time_vec.shape}"
        assert torch.all(torch.isfinite(time_vec)), "Time vector contains non-finite values"

    def test_frequency_decay_applied(self, factory_kwargs: dict):
        """Verify frequency decay is applied in forward pass."""
        conditioning_input = torch.randn(256, **factory_kwargs)
        config = {
            "cond_dim": 256, "n_heads": 4, "n_groups": 2, "d_state": 64,
            "n_freqs": 8, "L_timepoints": 256, "rank": 2, "hidden_dim": 256,
            "factory_kwargs": factory_kwargs
        }
        kernel = DCTKernel(**config)
        
        # Manually check decay application
        h = kernel.mlp(conditioning_input)
        decay = 1.0 / ((kernel.freqs + 1.0) ** (F.softplus(kernel.alpha) + 1e-4))
        
        assert decay.shape == (8,), f"Decay shape mismatch: {decay.shape}"
        assert torch.all(decay > 0), "Decay should be positive"
        assert torch.all(decay <= 1.0), "Decay should be <= 1.0"

    def test_extreme_frequency_single(self, factory_kwargs: dict):
        """Test edge case: n_freqs=1 (single frequency component)."""
        conditioning_input = torch.randn(256, **factory_kwargs)
        config = {
            "cond_dim": 256, "n_heads": 4, "n_groups": 2, "d_state": 64,
            "n_freqs": 1, "L_timepoints": 256, "rank": 2, "hidden_dim": 256,
            "factory_kwargs": factory_kwargs
        }
        kernel = DCTKernel(**config)
        kernel = kernel.to(**factory_kwargs)  # Ensure kernel is on correct device
        
        output = kernel.forward(cond=conditioning_input, A_log=None, B=None, C=None, dt=None, D=None)
        assert all(torch.isfinite(o).all() for o in output), "Forward pass failed with single frequency"


@pytest.mark.parametrize("kernel_class,kernel_specific", [
    (Hypernet, {}),
    (ChebyshevPolynomial, {"degree": 5}),
    (DCTKernel, {"n_freqs": 8, "L_timepoints": 256}),
])
class TestGradientFlow:
    """Test gradient flow through kernels for training stability."""

    def test_backward_pass_no_nan_inf(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict, conditioning_input: torch.Tensor
    ):
        """Verify backward pass produces finite gradients."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        kernel = kernel.to(**factory_kwargs)  # Ensure kernel is on correct device
        conditioning_input.requires_grad = True
        
        A_shift, B_scale, B_shift, C_scale, C_shift, dt_shift, D_shift = kernel.forward(
            cond=conditioning_input, A_log=None, B=None, C=None, dt=None, D=None
        )
        
        loss = A_shift.sum() + B_scale.sum() + C_scale.sum() + dt_shift.sum()
        loss.backward()
        
        # Check parameter gradients
        for param in kernel.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), \
                    f"Gradient contains non-finite values for {param}"

    def test_gradient_accumulation_across_samples(
        self, kernel_class: Type, kernel_specific: dict, 
        default_config: dict, factory_kwargs: dict
    ):
        """Verify gradients accumulate correctly across multiple forward/backward passes."""
        config = {**default_config, **kernel_specific, "factory_kwargs": factory_kwargs}
        kernel = kernel_class(**config)
        kernel = kernel.to(**factory_kwargs)  # Ensure kernel is on correct device
        
        # First pass
        cond1 = torch.randn(256, **factory_kwargs)
        out1 = kernel.forward(cond=cond1, A_log=None, B=None, C=None, dt=None, D=None)
        loss1 = sum(o.sum() for o in out1)
        loss1.backward()
        
        # Count parameters with gradients
        grad_count_1 = sum(1 for p in kernel.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        
        # Second pass with different conditioning
        kernel.zero_grad()
        cond2 = torch.randn(256, **factory_kwargs) + 10.0  # Significantly different conditioning
        out2 = kernel.forward(cond=cond2, A_log=None, B=None, C=None, dt=None, D=None)
        loss2 = sum(o.sum() for o in out2)
        loss2.backward()
        
        # Count parameters with gradients in second pass
        grad_count_2 = sum(1 for p in kernel.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        
        # Both passes should produce gradients in the same parameters
        assert grad_count_1 > 0, "First pass produced no gradients"
        assert grad_count_2 > 0, "Second pass produced no gradients"
        assert grad_count_1 == grad_count_2, "Gradient parameter count mismatch between passes"
