import torch
import pytest
import polars as pl
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer
import shutil
import os

from src.train import training_loop, training_loop_guider
from src.train import corrupt_tokens, masking_process

from src.hydra_model import HydraForMaskedLM, GuiderCore
from src.utils import load_config

CONFIG: PretrainedConfig = load_config("./configs/hydra.yaml", pretrained_config=True)
TRAIN_CONFIG: dict = load_config("./configs/training_config.yaml")

shutil.rmtree("src/__pycache__", ignore_errors=True)
shutil.rmtree("tests/__pycache__", ignore_errors=True)

def test_training_loop():

    """
        Test the simple training loop without scheduler or guider. Similar to the LlaDA pre-training.
    """

    path = "./data/wikitext-103-v1/train_tokens.pt"

    input_ids: torch.IntTensor = torch.load("./data/wikitext-103-v1/train_tokens.pt")[:TRAIN_CONFIG["batch_size"]] if os.path.exists(path) else None


    if input_ids is not None and input_ids.shape[-1] == TRAIN_CONFIG["pad_length"]:
        pass

    elif os.path.exists("./data/wikitext-103-v1/train.parquet"):
        df: pl.DataFrame = pl.read_parquet("./data/wikitext-103-v1/train.parquet")
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        input_ids = torch.empty((TRAIN_CONFIG["batch_size"], TRAIN_CONFIG["pad_length"]), dtype=torch.int64)

        for i, row in enumerate(df.iter_rows()):
            if i == TRAIN_CONFIG["batch_size"]:
                break
            text = row[0]
            encoding: torch.IntTensor = tokenizer.encode(text, return_tensors="pt", padding="max_length", max_length=TRAIN_CONFIG["pad_length"], truncation=True).to(dtype=torch.int64)
            input_ids[i] = encoding
    else:
        input_ids = torch.randint(1, 10000, (TRAIN_CONFIG["batch_size"], TRAIN_CONFIG["pad_length"]), dtype=torch.int64)

    weights = torch.load("./models/hydra_23layers.pt")

    assert input_ids.shape == (TRAIN_CONFIG["batch_size"], TRAIN_CONFIG["pad_length"]), f"Input IDs shape mismatch: {input_ids.shape}"

    hydra: HydraForMaskedLM = HydraForMaskedLM(config=CONFIG)
    hydra.load_state_dict(weights)
    optimizer: torch.optim.Optimizer = torch.optim.Adam(hydra.parameters(), lr=TRAIN_CONFIG["learning_rate"])

    # Run training loop
    training_loop(input_ids, hydra, optimizer, epochs=TRAIN_CONFIG["epochs"],\
                   batch_size=TRAIN_CONFIG["batch_size"], checkpoint_dir=TRAIN_CONFIG["checkpoint_dir"])

@pytest.mark.skip(reason="Not working on this feature yet")
def test_training_loop_guider():
    """
    Test the training loop for the Guider model.
    """

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_set = pl.read_parquet("data/wikitext-103-v1/train.parquet")

    input_ids: torch.IntTensor = torch.tensor([], dtype=torch.int32)

    for i, row in enumerate(train_set.iter_rows()):
        if i == TRAIN_CONFIG["batch_size"] + 1:
            break

        text = row[0]
        encoding: torch.IntTensor = tokenizer.encode(text, return_tensors="pt", padding="max_length", max_length=TRAIN_CONFIG["pad_length"], truncation=True).to(dtype=torch.int64)

        assert encoding.shape == (1, TRAIN_CONFIG["pad_length"]), f"Encoding shape mismatch: {encoding.shape}"

        input_ids = torch.cat((input_ids, encoding), dim=0)

    assert input_ids.shape == (TRAIN_CONFIG["batch_size"], TRAIN_CONFIG["pad_length"]), f"Input IDs shape mismatch: {input_ids.shape}"

    # Mock data
    # input_ids: torch.Tensor = torch.randint(0, CONFIG.vocab_size, (16, 128), device="cuda")
    guider: GuiderCore = GuiderCore(config=CONFIG)
    optimizer_guider = torch.optim.Adam(guider.parameters(), lr=0.001)

    # Run training loop with guider
    training_loop_guider(input_ids, guider, optimizer_guider, epochs=TRAIN_CONFIG["epochs"],\
                          batch_size=TRAIN_CONFIG["batch_size"], checkpoint_dir=TRAIN_CONFIG["checkpoint_dir"])

@pytest.mark.skip(reason="Not working on this feature yet")
def test_corrupt_tokens():
    """
        Test the corrupt tokens function to ensure its shape.

        Asserts:
            - The shape of the corrupted tokens matches the input IDs.
            - The shape of the target confidence matches the batch size.
            - The length of the corrupt indices is greater than zero.
    """

    input_ids: torch.Tensor = torch.randint(0, 1000, (32, 128), device="cuda")
    vocab_size: int = 1000
    corrupted_tokens, target_confidence, corrupt_indices, _ = corrupt_tokens(input_ids, vocab_size)

    assert corrupted_tokens.shape == input_ids.shape
    assert target_confidence.shape == input_ids.shape
    assert len(corrupt_indices) > 0

@pytest.mark.skip(reason="Not working on this feature yet")
def test_masking_process():

    """
        Test the masking process function to ensure its shape.

        Asserts:
            - The shape of the corrupt tokens matches the input IDs.
            - The shape of the masked indices matches the input IDs.
            - The shape of the p_mask matches the input IDs.
    """

    input_ids: torch.Tensor = torch.randint(0, 1000, (32, 128), device="cuda")

    corrupt_tokens: torch.Tensor
    masked_indices: torch.Tensor
    p_mask: torch.Tensor
    corrupt_tokens, masked_indices, p_mask = masking_process(input_ids)

    assert corrupt_tokens.shape == input_ids.shape
    assert masked_indices.shape == input_ids.shape
    assert p_mask.shape == input_ids.shape