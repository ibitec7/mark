import torch
from torch.utils.data import DataLoader

import logging
import os

from .scheduler import linear_warmup_cosine_decay, linear_warmup_polynomial_decay, inverse_sqrt_warmup
from .hydra_model import GuiderCore
from .utils import log_setup, load_config, TrainingMetrics
import math
import numpy as np

from tqdm import tqdm

DATA_DIR = os.path.join("data", "training")

LOG_FILE = os.path.join("logs", "training.log")
LOG_LEVEL = logging.INFO

CONFIG = load_config("configs/hydra.yaml", pretrained_config=True)
TRAIN_CONFIG = load_config("configs/training_config.yaml")

HYDRA_RECORDER = TrainingMetrics(dir_name=DATA_DIR, filename="hydra_train_metrics.parquet")
GUIDER_RECORDER = TrainingMetrics(dir_name=DATA_DIR, filename="guider_epoch_data.parquet")

logger = log_setup("TrainingLogger", LOG_FILE, LOG_LEVEL)

# Helper function
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch, checkpoint_dir="checkpoints", prefix="checkpoint"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    path = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    logger.info(f"Checkpoint saved at {path}")

# Helper function
def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path, device: str="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    logger.info(f"Checkpoint loaded from {checkpoint_path}")

    return checkpoint["epoch"]

# Helper function
def masking_process(input_ids: torch.Tensor, floor: float=1e-3, mask_ratio: float=None, mask_id: int=103, correction_mask: torch.Tensor=None, attn_mask: torch.Tensor=None):
    b, l = input_ids.shape

    valid_tokens = attn_mask == 1 if attn_mask is not None else (input_ids != 0)

    if mask_ratio is None:
        t = torch.rand(b, device=input_ids.device)
        p_mask = torch.clamp(t, min=floor, max=1.0 - floor)
        p_mask = p_mask[:, None].repeat(1, l)

        p_mask = torch.where(valid_tokens, p_mask, torch.zeros_like(p_mask))
    else:
        mask_ratio = torch.normal(mean=torch.tensor(mask_ratio), std=torch.tensor(0.05)).clamp(0.0, 1.0)        # Add some noise to the masking ratio
        p_mask = torch.full((b, l), mask_ratio, device=input_ids.device)
        p_mask = torch.clamp(p_mask, min=floor, max=1.0 - floor)
        
        p_mask = torch.where(valid_tokens, p_mask, torch.zeros_like(p_mask))

    logger.debug(f"Input shape: {input_ids.shape}, Masking ratio: {p_mask.mean().item():.4f}")
    masked_indices = ((torch.rand((b,l), device=input_ids.device) < p_mask) & (input_ids != 0)) & ((input_ids < 101) | (input_ids > 102))

    masked_indices = masked_indices if correction_mask is None else (masked_indices & correction_mask)

    logger.debug(f"Masked indices: {masked_indices.sum().item()} out of {b * l}")

    corrupt_tokens = torch.where(masked_indices, mask_id, input_ids)
    logger.debug(f"Corrupted tokens shape: {corrupt_tokens.shape}")

    assert masked_indices.shape == (b, l)

    assert corrupt_tokens.shape == (b, l)

    assert p_mask.shape == (b, l)

    return corrupt_tokens, masked_indices, p_mask

# Helper function
def corruption_process(input_ids: torch.Tensor, ratio=None, vocab_size: int=32000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrupts the input tokens based on a given ratio.
    
    Args:
        input_ids (torch.Tensor): The input token IDs.

        ratio (float): The corruption ratio (0 to 1).
    
    Returns:
        corrupt_tokens (torch.Tensor): The corrupted token IDs.

        target_confidence (torch.Tensor): The target confidence scores for each token.

        corrupt_indices (torch.BoolTensor): Boolean tensor indicating which tokens were corrupted.

        corrupt_mask (torch.Tensor): The corruption mask used for each token.
    """
    b, l = input_ids.shape

    # Calculate a random corruption ratio for each batch if None
    if ratio is None:
        t = torch.rand(b, device=input_ids.device)
        corrupt_mask = torch.clamp(t, min=1e-3, max=1.0 - 1e-3)
        corrupt_mask = corrupt_mask[:, None].repeat(1, l)
    else:
        corrupt_mask = torch.full((b,l), ratio, device=input_ids.device)
        corrupt_mask = torch.clamp(corrupt_mask, min=1e-3, max=1.0 - 1e-3)

    corrupt_indices: torch.BoolTensor = (torch.rand((b,l), device=input_ids.device) < corrupt_mask) & (input_ids != 0)

    # Copy input tokens to corrupt
    corrupt_tokens: torch.Tensor = input_ids.clone()

    target_confidence: torch.Tensor = torch.zeros_like(corrupt_tokens, device=input_ids.device, dtype=torch.float32)

    for i, corrupt_index in enumerate(corrupt_indices):
        num_corrupt = corrupt_index.sum().item()
        corrupt_tokens[i, corrupt_index] = torch.randint(0, vocab_size, (num_corrupt,), device=input_ids.device)
        target_confidence[i, corrupt_index] = 1.0

    assert corrupt_indices.shape == (b, l)

    assert corrupt_tokens.shape == (b, l)

    assert target_confidence.shape == (b, l)

    assert corrupt_mask.shape == (b, l)
    
    return corrupt_tokens, target_confidence, corrupt_indices, corrupt_mask

def sample_timestep(min: int=10, max: int=1000, device="cpu") -> tuple[int, int]:
    total_steps: torch.Tensor = torch.randint(min, max, (1,), device=device)
    current_step: torch.Tensor = torch.randint(0, total_steps, (1,), device=device)

    logger.debug(f"Sampled timestep: {current_step.item()} out of {total_steps.item()}")

    return current_step, total_steps

def context_adaptive_reweight(seq_len, distribution="symmetric-geometric", dtype: torch.dtype=torch.float32, device: torch.device | str="cpu", **kwargs):
    position_ids_l = np.arange(seq_len).reshape(-1, 1)
    position_ids_r = np.arange(seq_len).reshape(1, -1)
    distance = position_ids_l - position_ids_r
    distance = torch.from_numpy(distance).to(device=device)

    def geometric_distribution(k, cart_p: torch.Tensor=0.8, **kwargs):
        if not 0 < cart_p <= 1:
            raise ValueError("p must be between 0 and 1")

        res: torch.Tensor = (math.log(cart_p) + (k.abs() - 1) * math.log(1 - cart_p)).exp() * 0.5
        res.masked_fill_(k == 0, 0)  # ignore distance=0
        return res.to(dtype=dtype)

    if distribution == "symmetric-geometric":
        matrix = geometric_distribution(distance, **kwargs)
    else:
        raise ValueError(f"Unknown distribution {distribution}")

    return matrix.to(dtype=dtype)

def build_lr_scheduler(optimizer, train_config, steps_per_epoch):
    sched_cfg = train_config.get("lr_scheduler", {})
    typ = sched_cfg.get("type", "none")
    if typ == "none":
        return None, False
    if typ in ("cosine", "poly", "inverse_sqrt"):
        # Estimate total steps if not provided
        total_steps = sched_cfg.get("total_steps")
        if total_steps is None:
            total_steps = train_config["epochs"] * steps_per_epoch
        warmup_steps = sched_cfg.get("warmup_steps", 0)
        if typ == "cosine":
            return linear_warmup_cosine_decay(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.1)
            ), True
        if typ == "poly":
            poly = sched_cfg.get("polynomial", {})
            return linear_warmup_polynomial_decay(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                end_lr_ratio=poly.get("end_lr_ratio", 0.0),
                power=poly.get("power", 1.0)
            ), True
        if typ == "inverse_sqrt":
            return inverse_sqrt_warmup(optimizer, warmup_steps=warmup_steps), True
    if typ == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        plateau = sched_cfg.get("plateau", {})
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau.get("factor", 0.5),
            patience=plateau.get("patience", 3),
            min_lr=plateau.get("min_lr", 1e-6),
            verbose=True
        )
        return scheduler, False  # epoch-wise step with metric
    raise ValueError(f"Unknown lr scheduler type: {typ}")

def training_loop_guider(
        input_ids: torch.Tensor, guider: GuiderCore, optimizer: torch.optim.Optimizer,
        epochs: int=10, batch_size: int=32, checkpoint_dir: str="./checkpoints/guider", attention_mask: torch.Tensor=None
):
    logger.debug(f"Starting training loop for Guider for {epochs} epochs")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load the data to RAM for faster transfer to GPU
    data = DataLoader(input_ids, batch_size=batch_size, shuffle=True)

    scaler: torch.amp.grad_scaler.GradScaler = torch.amp.grad_scaler.GradScaler(device="cuda")

    batch_idx: int = 1

    for epoch in range(epochs):

        checkpoint_path = os.path.join(checkpoint_dir, f"guider_checkpoint_epoch_{epoch + 1}.pt")

        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(guider, optimizer, checkpoint_path)
            if start_epoch > epoch:
                logger.info(f"Skipping epoch {epoch+1} as it was already completed")
                continue

        guider.train()

        for batch in tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False):
            batch: torch.Tensor = batch.to("cuda", non_blocking=True)

            if attention_mask is not None:
                batch_tokens, batch_mask = batch
                batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
                batch_mask = batch_mask.to(device="cuda", non_blocking=True)
            else:
                (batch_tokens,) = batch
                batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
            

            mini_data = DataLoader(batch, batch_size=(batch_size // 4), shuffle=False)

            for mini_batch in mini_data:

                corrupted_tokens, target_confidence, _, _ = corruption_process(mini_batch, ratio=torch.rand(1).item(), vocab_size=CONFIG.vocab_size)

                padding_mask: torch.BoolTensor = (corrupted_tokens != 0)

                with torch.autocast(device_type="cuda"):
                    _, loss = guider.forward(corrupted_tokens, attention_mask=padding_mask, labels=target_confidence)

                optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

                batch_idx += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(data)}, Loss: {loss.item():.6f}")

        # Save checkpoint after each epoch
        save_checkpoint(guider, optimizer, epoch + 1, checkpoint_dir=checkpoint_dir, prefix="guider_checkpoint")

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")