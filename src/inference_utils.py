import torch

from transformers import AutoTokenizer
import random

from einops import repeat

from .hydra import Hydra
from .guider import Guider

MASKED_TOKEN = "<|mdm_mask|>"
# EMBEDDING DIM = (seq_len, embedding_dim)
MASKED_ID = 126336
MASKED_TOKEN_EMBEDDING = torch.zeros(1, 768)
MODEL = Hydra(d_model=768,)
GUIDER = Guider(d_model=768, d_state=64, d_conv=16, n_groups=1, head_dim=64, expand=2)
TOKENIZER = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")

def mask_predictor(prompt, masked_input, temp=1.0, noise=0.1, timestep=None):
    """
    Wrapper function for the Hydra model to predict masked tokens.

    Args:
        prompt (torch.Tensor): The prompt tensor of shape (batch_size, seq_len, d_model).
        masked_input (torch.Tensor): The input tensor with masked tokens of shape (batch_size, seq_len, d_model).
        temp (float): Temperature for sampling.
        noise (float): Scale of the noise to be added.
        timestep (torch.float32): Timestep tensor for the diffusion process.

    Returns:
        torch.Tensor: Predicted tokens after applying Gumbel noise. Shape (batch_size, seq_len, d_model).
    """

    input = torch.cat((prompt, masked_input), dim=1)

    # pseudo code to tokenize this combined token ids of "input" variable
    # tokenizer = AutoTokenizer.from_pretrained("LlaDA")
    # input_embeddings = tokenizer(input, return_tensors="pt")

    logits = MODEL.forward(input, timestep)
    predicted_tokens = add_gumbel_noise(logits, temperature=temp, noise_scale=noise)
    return predicted_tokens
    
def binary_mask(confidence_scores: torch.Tensor, top_k_ratio: float) -> torch.Tensor:
    """
    The top-k remasking strategy to choose which tokens to remask based on their confidence scores
    and the K-ratio. It outputs the binary mask to apply to the input tokens.
    
    Args:
        confidence_scores: Tensor of shape (batch_size, seq_len, 1) containing confidence scores
        top_k_ratio: Float between 0 and 1 indicating the proportion of tokens to keep unmasked
    
    Returns:
        Binary mask of shape (batch_size, seq_len, 1) where:
        - 0 indicates tokens to leave unmasked (high confidence)
        - 1 indicates tokens to remask (low confidence)
    """
    batch_size, seq_len, _ = confidence_scores.shape
    
    # Squeeze out the last dimension for easier processing
    confidence_scores = confidence_scores.squeeze(-1)
    
    # Calculate how many tokens to keep unmasked per batch
    k_ratio = max(1, int(seq_len * top_k_ratio))
    
    # Create empty mask tensor (initialized with 1s, meaning all tokens get remasked)
    mask = torch.ones_like(confidence_scores)
    
    # For each item in the batch
    for i in range(batch_size):
        # Get indices of top-k tokens with highest confidence
        _, top_indices = torch.topk(confidence_scores[i], k=k_ratio, dim=0)
        
        mask[i, top_indices] = 0
    
    return mask.unsqueeze(-1)

# A method to add noise for the sampling process depending on the temperature and noise scale.
def add_gumbel_noise(logits: torch.Tensor, temperature=1.0, noise_scale=0.1):
    """
    Add Gumbel noise to logits for sampling.
    
    Args:
        logits (torch.Tensor): Logits to which Gumbel noise will be added.
        temperature (float): Temperature parameter for scaling the noise.
        noise_scale (float): SSampling stepscale of the Gumbel noise.
        
    Returns:
        torch.Tensor: Logits with added Gumbel noise.
    """

    if temperature == 0:
        return logits
    
    # Increase precision as low-precision Gumbel Max improves perplexity but reduce generation quality.
    logits = logits.to(torch.float64)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits))) * noise_scale
    return logits + gumbel_noise / temperature

def get_num_transfer_tokens(mask_index, steps):

    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, dtype=torch.int64, device=mask_index.device) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

# Algorithm 4 of LlaDA paper
def greedy_remask(prompt, ans_len, sampling_steps):

    # Fully masked output sequence 'r_1'
    r_1 = repeat(MASKED_TOKEN_EMBEDDING, "1 d -> n d", n = ans_len)
    r_t = r_1.clone()

    t = 1.0

    for i in range(1, sampling_steps + 1):
        s = (sampling_steps - i - 1) / sampling_steps

        # Mask predictor will be the forward function of the Hydra given a partially masked input and a prompt
        r_0 = mask_predictor(prompt, r_t, out_confidence=False)


        for j in range(ans_len):
            if not torch.allclose(r_t[j], MASKED_TOKEN_EMBEDDING.squeeze(0)):
                r_0[j] = r_t[j]
            
            else:                
                r_0[j] = MASKED_TOKEN_EMBEDDING.squeeze(0) if random.random() < s else r_0[j]

        r_t = r_0.clone()

    return r_0


# This remasking strategy will only be used during inference.
@torch.no_grad()
def guider_remask(prompt, ans_len, sampling_steps, temperature=1.0, noise_scale=0.1, pretrain=False):
    
    if not pretrain:
        masked_response = repeat(MASKED_TOKEN_EMBEDDING, "1 d -> n d", n = ans_len)
    elif pretrain:
        masked_response = repeat(MASKED_TOKEN_EMBEDDING, "1 d -> n d", n = ans_len)
        masked_response = torch.cat((prompt, masked_response), dim=1)

    for timestep in range(1, sampling_steps + 1):
        
        k_ratio = timestep / (sampling_steps + 1)

        output_tokens = mask_predictor(prompt, masked_response, temp=temperature, noise=noise_scale)

        confidence_scores = GUIDER.forward(output_tokens, timestep)

        batch_size, seq_len, _ = confidence_scores.shape

        mask = binary_mask(confidence_scores, k_ratio)

        # Apply the mask to output_tokens - replacing masked positions with their embedding
        # Where mask is 1, we replace with discrete mask embedding; where mask is 0, we keep output_tokens
        d_model = output_tokens.size(-1)
        masked_positions = (mask == 1).expand(-1, -1, d_model)
        masked_embedding = MASKED_TOKEN_EMBEDDING.expand(batch_size, seq_len, d_model)

        remasked_tokens = torch.where(masked_positions, masked_embedding, output_tokens)

        # Update masked_response for the next iteration
        masked_response = remasked_tokens.clone()
            
# while !guider = torch.zeros(ans_len):
# [1, 2, 3, 4, 5] -> [0, 1, 0, 0, 1] -> [0, 0, 0, 0, 0]