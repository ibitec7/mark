import pytest
import yaml
import torch

from transformers.configuration_utils import PretrainedConfig

from src.hydra_model import HydraForMaskedLM

config: dict = yaml.safe_load(open("configs/hydra.yaml", "r"))
hydra_config = PretrainedConfig.from_dict(config)

MODEL: HydraForMaskedLM | None = None

def test_hydra_init():
    """
    Test the initialization of the HydraForMaskedLM model.
    """

    global MODEL
    model = HydraForMaskedLM(config=hydra_config)
    assert model is not None, "Model initialization failed."

    MODEL = model


def test_fwd():
    """
    Test the forward pass of the HydraForMaskedLM model.
    """
    global MODEL
    if MODEL is None:
        pytest.skip("Model not initialized. Skipping test.")

    input_ids = torch.randint(0, 32000, (1, 10), device="cuda")
    attention_mask = torch.ones((1, 10), device="cuda")

    outputs = MODEL(input_ids=input_ids, attention_mask=attention_mask)
    
    assert outputs is not None, "Forward pass failed."
    assert hasattr(outputs, 'logits'), "Output does not contain logits."
    assert outputs.logits.shape == (1, 10, 32000), f"Expected logits shape (1, 10, 32000), got {outputs.logits.shape}"