import pytest
import torch

from src.scheduler import NoiseScheduler

def test_embeddings(vocab: int=32000, d_model: int=512):

    """Test the embedding layers of the NoiseScheduler.

    Args:
        vocab (int, optional): Size of the vocabulary. Defaults to 32000.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.

    Asserts:
        - The step embedding shape matches (1, d_model).
        - The diffusion embedding shape matches (1, d_model).
        - The input embedding shape matches (1, seq_len, d_model).
        - The confidence embedding shape matches (1, seq_len, d_model).
    """

    scheduler: NoiseScheduler = NoiseScheduler(vocab_size=vocab, d_model=d_model, hidden_layers=3, max_steps=1000)

    dummy_total_steps: torch.Tensor = torch.randint(0, 1000, (1,), device="cuda")
    dummy_current_step: torch.Tensor = torch.randint(0, dummy_total_steps, (1,), device="cuda")

    dummy_input_ids: torch.Tensor = torch.randint(0, vocab, (1,10), device="cuda")

    dummy_confidence_scores: torch.Tensor = torch.rand((1, 10, 1), device="cuda")

    step_embedding: torch.Tensor = scheduler.step_embedding(dummy_current_step)
    diffusion_embedding: torch.Tensor = scheduler.diffusion_embedding(dummy_total_steps)
    input_embedding: torch.Tensor = scheduler.token_embedding(dummy_input_ids)
    conf_embedding: torch.Tensor = scheduler.conf_in_proj(dummy_confidence_scores)

    assert step_embedding.shape == (1, d_model), f"Expected step embedding shape (1, {d_model}), got {step_embedding.shape}"
    assert diffusion_embedding.shape == (1, d_model), f"Expected diffusion embedding shape (1, {d_model}), got {diffusion_embedding.shape}"
    assert input_embedding.shape == (1, 10, d_model), f"Expected input embedding shape (1, 10, {d_model}), got {input_embedding.shape}"
    assert conf_embedding.shape == (1, 10, d_model), f"Expected confidence embedding shape (1, 10, {d_model}), got {conf_embedding.shape}"

def test_gru(vocab: int=32000, d_model: int=512):

    """Test the GRU layer of the NoiseScheduler.

    Args:
        vocab (int, optional): Size of the vocabulary. Defaults to 32000.
        d_model (int, optional): Dimensionality of the model. Defaults to 512

    Asserts:
        - The output shape of the GRU matches (batch_size, d_model * 2).
    """

    scheduler: NoiseScheduler = NoiseScheduler(vocab_size=vocab, d_model=d_model, hidden_layers=3, max_steps=1000)

    dummy_input_ids: torch.Tensor = torch.randint(0, vocab, (2, 5), device="cuda")
    dummy_confidence_scores: torch.Tensor = torch.rand((2, 5, 1), device="cuda")
    dummy_timestep: torch.Tensor = torch.randint(0, 1000, (2,), device="cuda")
    dummy_total_steps: torch.Tensor = torch.randint(0, 1000, (2,), device="cuda")
    dummy_confidence_scores: torch.Tensor = torch.rand((2, 5, 1), device="cuda")

    batch_size, seq_len = dummy_input_ids.shape

    input_embedding: torch.Tensor = scheduler.token_embedding(dummy_input_ids)
    step_embedding: torch.Tensor = scheduler.step_embedding(dummy_timestep).unsqueeze(1).expand(batch_size, seq_len, -1)
    diffusion_embedding: torch.Tensor = scheduler.diffusion_embedding(dummy_total_steps).unsqueeze(1).expand(batch_size, seq_len, -1)
    conf_embedding: torch.Tensor = scheduler.conf_in_proj(dummy_confidence_scores)

    features: torch.Tensor = [input_embedding, step_embedding + diffusion_embedding, conf_embedding]

    features: torch.Tensor = torch.cat(features, dim=-1)

    h_n: torch.Tensor = scheduler.gru(features)[1]

    assert h_n.shape == (2, 2, d_model), f"Expected output shape (2, 2, {d_model}), got {h_n.shape}"

def test_mlp_output_shape(vocab: int=32000, d_model: int=512, batch_size: int=2, seq_len: int=2):

    """
    Test the output shape of the MLP in the NoiseScheduler.

    Args:
        vocab (int, optional): Size of the vocabulary. Defaults to 32000.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.
        batch_size (int, optional): Batch size. Defaults to 2.
        seq_len (int, optional): Sequence length. Defaults to 2.

    Asserts:
        - The output shape of the MLP matches (batch_size, 1).
    """
    
    scheduler: NoiseScheduler = NoiseScheduler(vocab_size=vocab, d_model=d_model, hidden_layers=3, max_steps=1000)

    dummy_h_n: torch.Tensor = torch.randn(batch_size, seq_len, d_model, device="cuda")\
        .transpose(0, 1).reshape(batch_size, -1)

    mlp_out: torch.Tensor = scheduler.mlp(dummy_h_n)

    assert mlp_out.shape == (batch_size, 1), f"Expected MLP output shape ({batch_size}, 1), got {mlp_out.shape}"
