import torch
import os
import logging
from omegaconf import DictConfig

from lightning.pytorch.strategies import DDPStrategy

import wandb

from transformers import BertTokenizerFast
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoModelForMaskedLM

from .utils import load_config, log_setup, get_best_checkpoint
from .hydra_model import HydraForMaskedLM
from .nemo import NemoForMaskedLM

from nemo import lightning as nl

torch.serialization.add_safe_globals([DictConfig])

torch.manual_seed(42)

TRAIN_CONFIG = load_config("configs/training_config.yaml", dict_config=True)
MODEL_CONFIG = load_config("configs/training_config.yaml", pretrained_config=True)

# os.environ.setdefault("NCCL_IB_DISABLE", "1")
# os.environ.setdefault("NCCL_NET", "Socket")

DATA_DIR="./data/wikitext-103-v1"
LOG_DIR="./logs"

logger=log_setup(log_name="TrainingLogger", log_file=os.path.join(LOG_DIR, "training.log"), level=logging.INFO)

TOKENIZER: BertTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")

MODEL: HydraForMaskedLM = HydraForMaskedLM

TRACKER: wandb.Run | None = None


def initialize_tracker() -> wandb.Run:
    """Lazy initialization of W&B tracker."""
    global TRACKER
    if TRACKER is not None:
        return TRACKER
    
    try:
        TRACKER = wandb.init(
            project=TRAIN_CONFIG["wandb"]["project"],
            config={
                "stage": TRAIN_CONFIG["stage"],
                "learning_rate_hydra": TRAIN_CONFIG["learning_rate_hydra"],
                "learning_rate_mark": TRAIN_CONFIG["learning_rate_mark"],
                "learning_rate_cls": TRAIN_CONFIG["learning_rate_cls"],
                "batch_size": TRAIN_CONFIG["batch_size"],
                "epochs": TRAIN_CONFIG["epochs"],
                "model": TRAIN_CONFIG["wandb"]["model_name"],
            }
        )
        logger.info("W&B initialized successfully")
    except Exception as e:
        logger.warning(f"W&B initialization failed: {e}. Proceeding without W&B tracking.")
        TRACKER = None
    
    return TRACKER


if __name__ == "__main__":
    
    # Initialize W&B tracker
    initialize_tracker()

    weights = torch.load(TRAIN_CONFIG["weights_path"], map_location="cpu")

    model_base: HydraForMaskedLM = HydraForMaskedLM(config=MODEL_CONFIG)

    # model_base: HydraForMaskedLM = AutoModelForMaskedLM.from_pretrained("./models/hydra_chebyshev_mark")
    # print(type(model_base))

    # print("Loaded AutoModelForMaskedLM from Hugging Face Hub with keys:")
    # for key in model_base.state_dict().keys():
    #     print(key)

    # Load weights with strict=False to handle new buffers (e.g., freqs) not in old checkpoints
    # model_base.load_state_dict(weights)
    missing_keys, unexpected_keys = model_base.load_state_dict(weights, strict=False)
    
    if missing_keys:
        logger.info(f"Missing keys when loading checkpoint (will use initialized values): {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

    del weights

    model_base = model_base.to(device="cuda")

    os.makedirs("./checkpoints/hydra_mark", exist_ok=True)

    chkpt_callback = ModelCheckpoint(
        dirpath="./checkpoints/hydra_mark",
        filename="best_hydra_mark",
        monitor="val_loss",
        mode="min",
    )

    if TRAIN_CONFIG["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(TRAIN_CONFIG.get("matmul_precision", "high"))

    if TRAIN_CONFIG["trainer"].get("strategy", None) == "ddp":
        # I can guarantee a static graph when using MaRK adapters, to prevent strict DDP checks from failing the implementation of CUDA streams.
        strategy = DDPStrategy(find_unused_parameters=True, static_graph=False, broadcast_buffers=False)         
        TRAIN_CONFIG["trainer"].pop("strategy", None)

        trainer: nl.Trainer = nl.Trainer(
            **TRAIN_CONFIG["trainer"],
            callbacks=[chkpt_callback],
            strategy=strategy,
            max_epochs=TRAIN_CONFIG["epochs"],
            gradient_clip_val=1.0
        )
    else:
        strategy = None

        trainer: nl.Trainer = nl.Trainer(
            **TRAIN_CONFIG["trainer"],
            callbacks=[chkpt_callback],
            max_epochs=TRAIN_CONFIG["epochs"],
            gradient_clip_val=1.0
        )

    model: NemoForMaskedLM = NemoForMaskedLM(config=TRAIN_CONFIG, tracker=TRACKER, trainer=trainer, model=model_base)
    model.to(device="cuda")


    TRACKER.watch(model, log=TRAIN_CONFIG["wandb"]["watch_log"], log_freq=TRAIN_CONFIG["wandb"]["log_freq"])

    model.setup_training_data()

    train_dl = model.train_dataloader()
    val_dl = model.val_dataloader()

    log_dir: str = "./profiles"

    path: str = TRAIN_CONFIG.get("checkpoint_dir", "./checkpoints/hydra_mark")

    best_chkpt: str = get_best_checkpoint(path)

    if TRAIN_CONFIG["ckpt_weights_only"] == True:
        
        ckpt = torch.load(best_chkpt, weights_only=False)

        # Load the checkpoint weights into the model (strict=False to handle new buffers)
        # model.load_state_dict(ckpt["state_dict"], strict=True)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing_keys:
            logger.info(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        trainer.fit(model, train_dl, val_dl, ckpt_path=None)

    else:
        trainer.fit(model, train_dl, val_dl, ckpt_path=best_chkpt)

