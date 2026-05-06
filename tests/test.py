from nemo import lightning as nl
from megatron.core.optimizer import OptimizerConfig

import torch
import os
import torch.nn as nn

from src.nemo import NemoForMaskedLM
from src.hydra_model import HydraForMaskedLM
from src.utils import load_config

MODEL_CONFIG = load_config("./configs/hydra.yaml", pretrained_config=True)
TRAIN_CONFIG = load_config("./configs/training_config.yaml", dict_config=True)
CHECKPOINT_DIR = os.path.join("./checkpoints", "hydra")

MODEL_PATH = os.path.join("./models", "hydra_23layers.pt")

if __name__ == "__main__":

    model: HydraForMaskedLM = HydraForMaskedLM(config=MODEL_CONFIG)
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)

    print("optimizer initialised successfully!")

    trainer = nl.Trainer(
        devices=1,
        max_steps=50,
        accelerator="gpu",
        strategy="ddp",
        # plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        limit_val_batches=0,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )

    nemo_model = NemoForMaskedLM(config=TRAIN_CONFIG, trainer=trainer, model=model)

    nemo_model.setup_training_data()

    train_dl = nemo_model.train_dataloader()

    trainer.fit(nemo_model, train_dl)