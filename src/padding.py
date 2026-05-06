"""
    Helper functions for the padding and unpadding of tokens in a batch.

    These are used for the purpose of efficient computation padding input tokens to the same length.
    When these tokens are used then only unpadded tokens are considered in the loss computation.
"""

### NOTE: Ensure that there is no such thing as not being able to remask a token that was not unmasked in an earlier step like in LlaDA,
## All response tokens can be remasked at any diffusion step even if not remasked earlier as other predictions may make earlier tokens irrelevant.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx: torch.autograd.grad_mode, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        To just get the values of the input that are in the indices tensor, so that they can be flattened
        for efficient computation.

        Used to unpad the input sequences in a batch.

        Args:
            ctx: The context for autograd.

            input {torch.Tensor}: The input tensor from which to gather values. shape (batch_size, ...)

            indices {torch.Tensor}: The indices of the values to gather. shape (batch_size, ...)
        """
        # Save the current indices for the backward pass
        ctx.save_for_backward(indices)
        
        # Sanity checks for the input tensors and the indices tensor
        assert input.dim() >= 2, "Input and indices must have at least 2 dimensions"
        assert indices.dim() == 1, "Indices must be a 1D tensor"

        # Get the dimension of the first axis and other dimensions
        ctx.first_axis_dim, other_dim_shape = input.shape[0], input.shape[1:]

        # Reshape the input tensor for efficient indexing
        input_flat = rearrange(input, "b ... -> b (...)")

        result = input_flat[indices]

        # Reshape the result to the desired output shape
        return result.view(-1, *other_dim_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Get the indices tensor back from the context of the forward pass
        indices, = ctx.saved_tensors

        # Sanity checks for the grad_output tensor
        assert grad_output.dim() >= 2, "grad_output must be a 2D tensor"

        # Collect all other dimensions of the grad_output tensor except the first one
        other_dim_shape = grad_output.shape[1:]

        # Reshape the grad_output tensor to flatten the other dimension
        grad_output = rearrange(grad_output, "b ... -> b (...)")

        # Create a zero tensor to hold the gradient that are from the flattened grad_output
        grad_input: torch.Tensor = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                 device=grad_output.device,
                                 dtype=grad_output.dtype)
        
        # Scatter the grad_output values into the grad_input tensor at the specified indices
        grad_input.scatter_(0,
                            repeat(indices, "b -> b d", d=grad_output.shape[1]),
                            grad_output)
        
        # Reshape the grad tensor the same way as the input tensor
        return grad_input.view(ctx.first_axis_dim, *other_dim_shape), None
    
index_first_axis = IndexFirstAxis.apply

class IndexPutFirstAxis(torch.autograd.Function):
    """This function is used to populate values into the first axis of a tensor at specified indices.
    Used for padding the sequences in a batch to the same length.

    Args:
        ctx: The context for autograd.

        values {torch.Tensor}: The values to be put into the tensor. Should be at least 2D.

        indices {torch.Tensor}: The indices where the values should be placed. Should be 1D.

        first_axis_dim {int}: The size of the first axis of the output tensor.
    """

    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:

        # Save the indices in the gradient context for the backward pass
        ctx.save_for_backward(indices)

        # sanity checks on the input and value tensors
        assert indices.ndim == 1, "Indices must be a 1D tensor"
        assert values.ndim >= 2, "Values must be at least a 2D tensor"

        # initialize the output tensor with zeros the specific first dimension and other dimensions of values.
        output = torch.zeros((first_axis_dim,
                             values.shape[1]),
                             device=values.device,
                             dtype=values.dtype)
        
        # populate the output tensor with the values at the specified indices
        output[indices] = values

        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:

        # Get indices from the saved context of forward pass
        indices, = ctx.saved_tensors

        # Populate the gradient values at the specified indices
        grad_values = grad_output[indices]

        # Return the populated gradient values.
        return grad_values, None, None
    

index_put_first_axis = IndexPutFirstAxis.apply

def unpad_input(
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Removing the padding from the input sequences.

    Args:
        hidden_states {torch.Tensor}: Hidden states of shape (batch_size, seq_len, ...)

        attention_mask {torch.Tensor}: Masking tensor of shape (batch_size, seq_len), int 1 means valid and 0 invalid.

    Returns:
        hidden_states {torch.Tensor}: shape (total_selected, ...) where total_selected is the number of valid tokens in the batch (selected by masked_indices).

        indices {torch.Tensor}: shape (total_selected,) the indices of the valid tokens in the original input sequence.

        cu_seqlens {torch.Tensor}: shape (batch_size + 1,) the cummulative seq_lens to index the hidden states.

        max_seq_len_in_batch {int}: The maximum sequence length in the batch, used for padding purposes.
    """

    # Get the sequence lengths in the batch
    seq_lens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int64)

    # Get the 1D indices from the masked indices where the values are 1 (valid tokens)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # Calculate the maximum sequence length in the batch (would likely be same for all as tokenizer pads to max length)
    max_seqlen_in_batch = int(seq_lens_in_batch.max().item())

    # Calculate the cumulative sequence lengths
    cu_seqlens = F.pad(torch.cumsum(seq_lens_in_batch, dim=0, dtype=torch.int32), pad=(1,0))

    hidden_states = torch.as_tensor(index_first_axis(
        rearrange(hidden_states, "b s ... -> (b s) ..."),
        indices
    ))

    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

def unpad_input_only(
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor,
) -> torch.Tensor:
    """
    Same as unpad_input (meaning removing padding from input) but only returns the hidden states without indices and cu_seqlens.

    This method saves some overhead memory

    Args:
        hidden_states {torch.Tensor}: Hidden states of shape (batch_size, seq_len, ...)

        attention_mask {torch.Tensor}: Masking tensor of shape (batch_size, seq_len), int 1 means valid and 0 invalid.

    Returns:
        torch.Tensor: shape (total_selected, ...) where total_selected is the number of valid tokens in the batch (selected by masked_indices).
    """

    # Get the 1D indices from the masked_indices where the values are 1 (valid tokens)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    rearranged_hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")

    return index_first_axis(rearranged_hidden_states, indices)

def pad_input(
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
) -> torch.Tensor:
    """Add padding to sequences

    Args:
        hidden_states {torch.Tensor}: shape (total_selected, ...) where total_selected is the number of valid tokens in the batch (selected by indices).

        indices {torch.Tensor}: shape (total_selected,) the indices of the valid tokens in the original input sequence.

        batch_size {int}: the number of sequences in the batch.

        seq_len {int}: the maximum sequence length in the batch.

    Returns:
        torch.Tensor: shape (batch_size, seq_len, ...) where the hidden states are padded to the maximum sequence length in the batch.
    """

    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)