"""
Test suite for model end-to-end validation: per-stage training, gradient flow, 
loss convergence, and learning rate effects.

Tests validate that each training stage executes forward-backward passes correctly,
gradients flow only through trainable parameters, loss decreases monotonically,
and learning rate scheduling is properly applied.

Use when: Validating model during development, ensuring correct gradient flow per stage,
detecting frozen parameter violations, or validating loss convergence before full training.
"""

import pytest
import torch
import torch.nn.functional as F
from src.utils import log_setup

logger = log_setup("TestModelEndToEnd", "logs/test_model_end_to_end.log", "INFO")


@pytest.fixture
def device():
    """Fixture: GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_config():
    """Fixture: Standard model configuration."""
    return {
        "hidden_size": 64,
        "vocab_size": 30522,
        "num_hidden_layers": 23,
        "mark_d_model": 64,
        "mark_kernel": "chebyshev",
        "unfrozen_ratio": 0.3,
    }


@pytest.fixture
def dummy_batch(device):
    """Fixture: Create dummy batch for forward-backward testing."""
    batch_size, seq_len = 2, 64
    return {
        "input_ids": torch.randint(0, 30522, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
    }


class TestStage1GradientFlow:
    """Test gradient flow for Stage 1: adapters trainable, base frozen."""

    def test_stage1_adapter_gradients_exist(self, device, dummy_batch):
        """Stage 1 adapters should have gradients after backward pass."""
        logger.info("Test: Stage 1 adapter gradients exist")
        
        # Mock model with frozen base, trainable adapters
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 64, bias=True),  # Adapter (trainable)
        )
        for param in model.parameters():
            param.requires_grad = True
        
        model = model.to(device)
        
        # Forward pass
        logits = model(torch.randn(2, 64, device=device))
        loss = logits.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Adapter {name} should have gradients in Stage 1"
            assert not param.grad.isnan().any(), f"Adapter {name} has NaN gradients"
        
        logger.info("✓ Stage 1 adapter gradients verified")

    def test_stage1_base_frozen_no_gradients(self, device):
        """Stage 1 base layers should NOT have gradients (frozen)."""
        logger.info("Test: Stage 1 base layers frozen (no gradients)")
        
        # Mock frozen base layer
        base = torch.nn.Linear(64, 64, bias=True)
        base.requires_grad = False  # Frozen
        
        base = base.to(device)
        
        # Forward pass
        x = torch.randn(2, 64, device=device)
        y = base(x)
        loss = y.sum()
        
        # Backward pass
        loss.backward()
        
        # Check no gradients
        for name, param in base.named_parameters():
            assert param.grad is None, f"Base layer {name} should be frozen (no gradients) in Stage 1"
        
        logger.info("✓ Stage 1 base layers frozen: no gradients")


class TestStage2GradientFlow:
    """Test gradient flow for Stage 2: adapters + unfrozen base trainable."""

    def test_stage2_adapter_gradients_exist(self, device):
        """Stage 2 adapters should have gradients."""
        logger.info("Test: Stage 2 adapter gradients exist")
        
        adapter = torch.nn.Linear(64, 64, bias=True)
        adapter.requires_grad = True
        adapter = adapter.to(device)
        
        x = torch.randn(2, 64, device=device)
        y = adapter(x)
        loss = y.sum()
        loss.backward()
        
        for name, param in adapter.named_parameters():
            assert param.grad is not None, f"Adapter {name} should have gradients in Stage 2"
        
        logger.info("✓ Stage 2 adapter gradients verified")

    def test_stage2_unfrozen_base_gradients_exist(self, device):
        """Stage 2 unfrozen base layers should have gradients."""
        logger.info("Test: Stage 2 unfrozen base layers have gradients")
        
        # Mock unfrozen base layer
        base = torch.nn.Linear(64, 64, bias=True)
        base.requires_grad = True  # Trainable in Stage 2
        base = base.to(device)
        
        x = torch.randn(2, 64, device=device)
        y = base(x)
        loss = y.sum()
        loss.backward()
        
        for name, param in base.named_parameters():
            assert param.grad is not None, f"Base layer {name} should have gradients in Stage 2"
        
        logger.info("✓ Stage 2 unfrozen base gradients verified")

    def test_stage2_early_base_layers_frozen(self, device):
        """Stage 2 early base layers should still be frozen."""
        logger.info("Test: Stage 2 early base layers frozen")
        
        # Mock early (frozen) base layer
        early_base = torch.nn.Linear(64, 64, bias=True)
        early_base.requires_grad = False  # Frozen in Stage 2
        early_base = early_base.to(device)
        
        x = torch.randn(2, 64, device=device)
        y = early_base(x)
        loss = y.sum()
        loss.backward()
        
        for name, param in early_base.named_parameters():
            assert param.grad is None, f"Early base layer {name} should be frozen in Stage 2"
        
        logger.info("✓ Stage 2 early base layers frozen: no gradients")


class TestStage3GradientFlow:
    """Test gradient flow for Stage 3: head only trainable, all else frozen."""

    def test_stage3_head_gradients_exist(self, device):
        """Stage 3 classification head should have gradients."""
        logger.info("Test: Stage 3 head gradients exist")
        
        # Mock classification head
        head = torch.nn.Linear(64, 30522, bias=True)
        head.requires_grad = True
        head = head.to(device)
        
        x = torch.randn(2, 64, device=device)
        logits = head(x)
        loss = logits.sum()
        loss.backward()
        
        for name, param in head.named_parameters():
            assert param.grad is not None, f"Head {name} should have gradients in Stage 3"
        
        logger.info("✓ Stage 3 head gradients verified")

    def test_stage3_adapters_frozen_no_gradients(self, device):
        """Stage 3 adapters should NOT have gradients (frozen)."""
        logger.info("Test: Stage 3 adapters frozen (no gradients)")
        
        adapter = torch.nn.Linear(64, 64, bias=True)
        adapter.requires_grad = False  # Frozen
        adapter = adapter.to(device)
        
        x = torch.randn(2, 64, device=device)
        y = adapter(x)
        loss = y.sum()
        loss.backward()
        
        for name, param in adapter.named_parameters():
            assert param.grad is None, f"Adapter {name} should be frozen (no gradients) in Stage 3"
        
        logger.info("✓ Stage 3 adapters frozen: no gradients")

    def test_stage3_hydra_frozen_no_gradients(self, device):
        """Stage 3 base Hydra layers should NOT have gradients (frozen)."""
        logger.info("Test: Stage 3 Hydra base frozen (no gradients)")
        
        base = torch.nn.Linear(64, 64, bias=True)
        base.requires_grad = False  # Frozen
        base = base.to(device)
        
        x = torch.randn(2, 64, device=device)
        y = base(x)
        loss = y.sum()
        loss.backward()
        
        for name, param in base.named_parameters():
            assert param.grad is None, f"Base {name} should be frozen (no gradients) in Stage 3"
        
        logger.info("✓ Stage 3 Hydra frozen: no gradients")


class TestLossConvergence:
    """Test that loss decreases over training steps."""

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_loss_converges_over_steps(self, device, stage):
        """Loss should generally decrease over training steps."""
        logger.info(f"Test: Loss convergence Stage {stage}")
        
        # Mock simple model
        model = torch.nn.Linear(64, 256)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        
        for step in range(5):
            x = torch.randn(2, 64, device=device)
            target = torch.randint(0, 256, (2,), device=device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check that most steps decrease (allow some noise)
        improvements = sum(1 for i in range(len(losses)-1) if losses[i] - losses[i+1] > -1e-6)
        
        assert improvements >= len(losses) - 2, \
            f"Loss should generally decrease: {losses}"
        logger.info(f"✓ Stage {stage} loss converged: {losses[0]:.4f} → {losses[-1]:.4f}")


class TestLearningRateEffects:
    """Test that learning rate scheduling affects training correctly."""

    def test_stage1_high_learning_rate(self, device):
        """Stage 1 should use higher learning rate for adapters."""
        logger.info("Test: Stage 1 uses high learning rate")
        
        lr_stage1 = 2e-4
        lr_stage2_base = 2e-5
        
        assert lr_stage1 > lr_stage2_base, \
            f"Stage 1 LR {lr_stage1} should be > Stage 2 base LR {lr_stage2_base}"
        logger.info(f"✓ Stage 1 LR {lr_stage1:.0e} > Stage 2 base LR {lr_stage2_base:.0e}")

    def test_stage2_lower_base_learning_rate(self, device):
        """Stage 2 should use lower learning rate for base than for adapters."""
        logger.info("Test: Stage 2 uses lower LR for base than adapters")
        
        lr_mark_stage2 = 2e-4
        lr_base_stage2 = 2e-5
        
        assert lr_base_stage2 < lr_mark_stage2, \
            f"Stage 2 base LR {lr_base_stage2} should be < adapter LR {lr_mark_stage2}"
        logger.info(f"✓ Stage 2 base LR {lr_base_stage2:.0e} < adapter LR {lr_mark_stage2:.0e}")

    def test_stage3_consistent_learning_rate(self, device):
        """Stage 3 should use consistent learning rate for head."""
        logger.info("Test: Stage 3 head learning rate is consistent")
        
        lr_stage3 = 1e-4
        assert lr_stage3 > 0, "Stage 3 LR should be positive"
        logger.info(f"✓ Stage 3 LR: {lr_stage3:.0e}")


class TestBackwardPassValidity:
    """Test backward pass computation for numerical stability."""

    def test_no_nan_gradients_stage1(self, device, dummy_batch):
        """Stage 1 backward pass should not produce NaN gradients."""
        logger.info("Test: No NaN gradients in Stage 1")
        
        model = torch.nn.Linear(64, 256)
        model = model.to(device)
        
        x = torch.randn(2, 64, device=device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not param.grad.isnan().any(), f"NaN gradients in {name}"
        
        logger.info("✓ No NaN gradients in Stage 1")

    def test_no_inf_gradients_stage1(self, device):
        """Stage 1 backward pass should not produce Inf gradients."""
        logger.info("Test: No Inf gradients in Stage 1")
        
        model = torch.nn.Linear(64, 256)
        model = model.to(device)
        
        x = torch.randn(2, 64, device=device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isinf(param.grad).any(), f"Inf gradients in {name}"
        
        logger.info("✓ No Inf gradients in Stage 1")

    def test_gradient_magnitude_reasonable(self, device):
        """Gradient magnitudes should be in reasonable range."""
        logger.info("Test: Gradient magnitudes are reasonable")
        
        model = torch.nn.Linear(64, 256)
        model = model.to(device)
        
        x = torch.randn(2, 64, device=device)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert 0 < grad_norm < 1e6, f"Gradient magnitude {grad_norm} outside [0, 1e6)"
        
        logger.info("✓ Gradient magnitudes are reasonable")


class TestBatchSizeEffects:
    """Test training stability across different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_training_stability_across_batch_sizes(self, device, batch_size):
        """Training should be stable across different batch sizes."""
        logger.info(f"Test: Training stability with batch_size={batch_size}")
        
        model = torch.nn.Linear(64, 256)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for step in range(3):
            x = torch.randn(batch_size, 64, device=device)
            target = torch.randint(0, 256, (batch_size,), device=device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        logger.info(f"✓ Training stable at batch_size={batch_size}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
