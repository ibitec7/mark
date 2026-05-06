import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from src.nemo import NemoForMaskedLM
from src.hydra_model import HydraForMaskedLM
from src.utils import load_config
from src.train import corrupt_tokens, masking_process

CONFIG = load_config("./configs/training_config.yaml", dict_config=True)

MODEL_CONFIG = load_config("./configs/hydra.yaml", pretrained_config=True)

class DummyHydraForMaskedLM(HydraForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    def forward(self, input_ids, attention_mask=None, current_timestep=None, total_timestep=None, labels=None, p_mask=None):
        # Return a dummy MaskedLMOutput with loss
        class DummyOutput:
            def __init__(self):
                self.loss = torch.tensor(1.23)
        return DummyOutput()

def make_dummy_config():
    return {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "epochs": 1,
        "lr_scheduler": {"type": "none"}
    }

@pytest.fixture
def dummy_model():
    CONFIG = load_config("./configs/training_config.yaml", dict_config=True)

    MODEL_CONFIG = load_config("./configs/hydra.yaml", pretrained_config=True)

    MODEL = HydraForMaskedLM(config=MODEL_CONFIG)
    
    try:
        return NemoForMaskedLM(config=CONFIG, trainer=None, model=MODEL)
    except TypeError as e:
        print(f"{e}")

@pytest.fixture
def dummy_batch():
    # 2 samples, 4 tokens each
    input_ids = torch.randint(0, 100, (2, 4), device="cuda")
    attention_mask = torch.ones_like(input_ids, device="cuda")


    return (input_ids, attention_mask)


def test_forward(dummy_model, dummy_batch):

    items = dummy_model._prepare_batch_inputs(dummy_batch)

    out = dummy_model.forward(
            input_ids=items["corrupt_tokens"],
            attention_mask=items["attention_mask"],
            current_timestep=items["current_step"],
            total_timestep=items["total_steps"],
            labels=items["labels"],
            p_mask=items["p_mask"],
        )
    assert hasattr(out, "loss")
    assert isinstance(out.loss, torch.Tensor)

def test_training_step(dummy_model, dummy_batch):
    loss = dummy_model.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)

def test_validation_step(dummy_model, dummy_batch):
    dummy_model.val_losses.clear()
    dummy_model.validation_step(dummy_batch, batch_idx=0)
    assert len(dummy_model.val_losses) == 1

def test_setup_training_data(dummy_model, dummy_batch):
    input_ids, attention_mask = dummy_batch
    dummy_model.setup_training_data()
    dl = dummy_model.train_dataloader()
    batch = next(iter(dl))

    batch_tokens, batch_mask = batch
    batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
    batch_mask = batch_mask.to(device="cuda", non_blocking=True)

    assert batch_tokens is not None
    assert batch_tokens.shape == (CONFIG["batch_size"], CONFIG["pad_length"])

    assert batch_mask.shape == (CONFIG["batch_size"], CONFIG["pad_length"])
    assert batch_mask is not None
