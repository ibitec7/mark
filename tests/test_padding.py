import pytest
import torch

import src.padding as padding

def test_unpad_input_only():
    # Create a sample hidden states tensor and attention mask
    hidden_states = torch.randn(3, 5, 10)  # (batch_size, seq_len, feature_dim)
    attention_mask = torch.tensor([[1, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0]])

    # Call the unpad_input_only function
    result = padding.unpad_input_only(hidden_states, attention_mask)

    # Check the shape of the result
    assert result.shape == (6, 10)  # Total valid tokens: 6 (3 + 2 + 1)

    # Let's change the attention mask to have different valid tokens
    attention_mask = torch.tensor([[1, 0, 0, 0, 1], [0, 1, 1, 0, 1], [1, 0, 1, 1, 1]])

    # Call the unpad_input_only function again
    result = padding.unpad_input_only(hidden_states, attention_mask)

    assert result.shape == (9, 10)  # Total valid tokens: 9 (2 + 3 + 4)

def test_unpad_input():
    # Create a sample hidden states tensor and attention mask
    hidden_states = torch.randn(3, 5, 10)  # (batch_size, seq_len, feature_dim)
    attention_mask = torch.tensor([[1, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0]])

    # Call the unpad_input function
    result_hidden_states, indices, cu_seqlens, max_seqlen_in_batch = padding.unpad_input(hidden_states, attention_mask)

    # Check the shape of the result
    assert result_hidden_states.shape == (6, 10)  # Total valid tokens: 6 (3 + 2 + 1)
    assert indices.shape == (6,)
    assert cu_seqlens.shape == (4,)  # Cumulative sequence lengths
    assert max_seqlen_in_batch == 3  # The maximum seq_len of valid tokens in the batch is 3

    # Let's change the attention mask to have different valid tokens
    attention_mask = torch.tensor([[1, 0, 0, 0, 1], [0, 1, 1, 0, 1], [1, 0, 1, 1, 1]])

    # Call the unpad_input function again
    result_hidden_states, indices, cu_seqlens, max_seqlen_in_batch = padding.unpad_input(hidden_states, attention_mask)

    assert result_hidden_states.shape == (9, 10)  # Total valid tokens: 9 (2 + 3 + 4)
    assert indices.shape == (9,)                  # Total valid tokens is 9 so there should be 9 indices
    assert cu_seqlens.shape == (4,)               # Cumulative sequence lengths
    assert max_seqlen_in_batch == 4               # The maximum seq_len in the unpadded batch

def test_pad_input():
    # Create a sample hidden states tensor and indices
    hidden_states = torch.randn(6, 10)  # (total_selected, feature_dim)
    indices = torch.tensor([0, 1, 2, 3, 4, 5])  # Indices of valid tokens
    batch_size = 3
    seq_len = 5

    # Call the pad_input function
    padded_hidden_states = padding.pad_input(hidden_states, indices, batch_size, seq_len)

    # Check the shape of the result
    assert padded_hidden_states.shape == (3, 5, 10)  # (batch_size, seq_len, feature_dim)

    # For each index check if the padded hidden states match the original hidden states.
    for i, idx in enumerate(indices):
        batch = idx // seq_len          # Get the batch index
        pos = idx % seq_len             # Get the position in the sequence
        assert torch.allclose(padded_hidden_states[batch, pos], hidden_states[i])

    # Also check that in all positions that were not in indices are padded with zeros.
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for idx in indices:
        batch = idx // seq_len
        pos = idx % seq_len
        mask[batch, pos] = True
    assert torch.all(padded_hidden_states[~mask] == 0)

def test_unpad_pad_cycle():
    batch_size, seq_len, feature_dim = 4, 6, 8
    # create a random hidden‐states and a random attention mask
    hidden = torch.randn(batch_size, seq_len, feature_dim)
    # random bool mask; ensure at least one True so unpad isn’t empty
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)
    mask[0, 0] = True

    # 1) unpad
    unpadded, indices, cu_seqlens, max_seq = padding.unpad_input(hidden, mask)
    # sanity checks
    assert unpadded.shape == (mask.sum().item(), feature_dim)
    assert indices.shape[0] == mask.sum().item()

    # 2) pad back
    reconstructed = padding.pad_input(unpadded, indices, batch_size, seq_len)

    # shape restored
    assert reconstructed.shape == hidden.shape

    # on masked (valid) positions we recover the original
    assert torch.allclose(reconstructed[mask], hidden[mask])

    # on the rest we have zeros
    assert torch.all(reconstructed[~mask] == 0)