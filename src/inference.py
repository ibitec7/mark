from .nemo import NemoForMaskedLM
from .hydra_model import HydraForMaskedLM
from .utils import load_config
from .train import corruption_process, masking_process
import torch
from .utils import get_best_checkpoint

import matplotlib.pyplot as plt

import polars as pl

from transformers import AutoTokenizer
# pip install colorama
from colorama import init, Fore, Style

TRAIN_CONFIG = load_config("configs/training_config.yaml", dict_config=True)
MODEL_CONFIG = load_config("configs/hydra.yaml", pretrained_config=True)

init()  # Initialize colorama

def format_corrupted_tokens_colorama(tokenizer, original_ids, corrupted_ids, target_confidence, color=Fore.RED):
    """Format tokens with red color for corrupted ones using colorama."""
    original_tokens = tokenizer.convert_ids_to_tokens(original_ids[0])
    corrupted_tokens = tokenizer.convert_ids_to_tokens(corrupted_ids[0])
    confidence_mask = target_confidence[0]
    
    formatted_tokens = []
    for orig_token, corr_token, is_masked in zip(original_tokens, corrupted_tokens, confidence_mask):
        if is_masked:
            formatted_tokens.append(f"{color}{corr_token}{Style.RESET_ALL}")
        else:
            formatted_tokens.append(corr_token)
    
    return tokenizer.convert_tokens_to_string(formatted_tokens)

if __name__ == "__main__":

    df = pl.read_parquet("data/wikitext-103-v1/test.parquet")

    my_string = ""

    for i in range(50, 54):
        my_string += df[i, "text"] + "\n"

    model_base = HydraForMaskedLM(config=MODEL_CONFIG)

    chkpt = torch.load("./checkpoints/hydra/best_hydra_model-v4.ckpt", weights_only=False, map_location="cuda")

    # chkpt = torch.load("./checkpoints/best_hydra_model.ckpt", weights_only=False, map_location="cuda")

    model = NemoForMaskedLM(config=TRAIN_CONFIG, model=model_base, inference=True)

    model.load_state_dict(chkpt["state_dict"])
    model.to("cuda")
    # print(chkpt["epoch"])

    for param in model.parameters():
        param.requires_grad = False

    # print(model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # my_string = """
    # Hi I am a university student studing for my degree. I built a new model that predicts tokens iteratively similar to how diffusion models work in computer vision.
    # The model is called Hydra and it uses a novel approach to masked language modeling by incorporating timestep embeddings and a corruption process during training.
    # This allows the model to better understand the context and semantics of the text, leading to improved performance on various NLP tasks.
    # """

    tokens = tokenizer(my_string, return_tensors="pt").convert_to_tensors(tensor_type="pt")

    tensor: torch.Tensor = tokens["input_ids"]
    attn_mask: torch.Tensor = tokens["attention_mask"]

    print("Sequence Length: ", tensor.shape[1])
    
    # tensor = torch.tensor([i for i in range(1, 30)]).unsqueeze(0)

    # current_step, total_step = sample_timestep()

    current_step = 90
    total_step = 100

    mask_ratio = 1.0 - (float(current_step) / float(total_step))

    masked_tensor, masked_indices, p_mask = masking_process(tensor, mask_ratio=mask_ratio)

    corrupt_tensor, target_confidence, _, _ = corruption_process(tensor, ratio=0.2, vocab_size=32000)

    output = model.forward(masked_tensor.cuda(), attention_mask=attn_mask.cuda(), current_timestep=current_step, total_timestep=total_step)

    predicted_tokens = torch.argmax(output.logits, dim=-1)

    predict_out = masked_tensor.clone().cuda()
    predict_out[masked_indices] = predicted_tokens[masked_indices]

    correction_mask: torch.Tensor = (predict_out != tensor.cuda())

    predict_out[correction_mask] = 103  # [MASK] token id in BERT tokenizer

    a = []
    a.append(masked_indices.sum().item())
    a.append(correction_mask.sum().item())

    for i in range((total_step - current_step)):


        current_step = min(current_step + i + 1, total_step)
        output = model.forward(predict_out, attention_mask=attn_mask.cuda(), current_timestep=current_step, total_timestep=total_step)

        predicted_tokens = torch.argmax(output.logits, dim=-1)

        predict_out[correction_mask] = predicted_tokens[correction_mask]

        correction_mask: torch.Tensor = (predict_out != tensor.cuda())
        predict_out[correction_mask] = 103  # [MASK] token id in BERT tokenizer

        a.append(correction_mask.sum().item())

    print("Logits Dimension: ", predicted_tokens.shape)

    print("Masked tokens: ", target_confidence.sum().item())

    # print("Original Tensor: ", tensor)
    # print("---" * 45)
    # print("Corrupted Tensor: ", corrupt_tensor)
    # print("---" * 45)
    # print("Target Confidence: ", target_confidence)
    print("---" * 45)

    original_string = format_corrupted_tokens_colorama(tokenizer, tensor, tensor, masked_indices, color=Fore.BLUE)


    print("Original String: ", original_string)

    print("---" * 45)

    # Format and print the corrupted tokens in color
    formatted_corrupted_string = format_corrupted_tokens_colorama(tokenizer, tensor, masked_tensor, masked_indices, color=Fore.RED)
    print("Formatted Masked String: ", formatted_corrupted_string)

    print("---" * 45)

    formatted_predicted_string = format_corrupted_tokens_colorama(tokenizer, tensor, predict_out, masked_indices, color=Fore.GREEN)
    print("Formatted Predicted String: ", formatted_predicted_string)

    print("---" * 45)

    formatted_full_predicted_string = format_corrupted_tokens_colorama(tokenizer, tensor, predicted_tokens.cpu(), masked_indices, color=Fore.MAGENTA)
    print("Formatted Full Predicted String: ", formatted_full_predicted_string)

    plt.plot(range(1, len(a)+1), a, color="orange", marker='o')
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Tokens Remaining to Correct")
    plt.title(f"Token Correction Progress {mask_ratio*100:.1f}% Masking")
    plt.show()
