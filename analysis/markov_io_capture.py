"""Capture real pre-MaRK B and C vectors for the paper dynamics figure.

One forward pass per kernel is enough; `dynamics_analysis.py` composes the
captured vectors offline with the timestep-swept adapter tensors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from transformers import BertTokenizerFast

from src.hydra_model import HydraForMaskedLM, HydraForMaskedLMConfig
from src.hydra import Hydra

logger = logging.getLogger(__name__)

NUM_LAYERS = 23
N_HEADS = 12
N_GROUPS = 1
D_STATE = 64
BC_DIM = N_GROUPS * D_STATE

# Sequence / plot defaults (paper caption)
MARKOV_SEQ_LEN = 512
MARKOV_TOKEN_POS = 256  # middle of padded sequence
T_SNAPSHOTS = [0.0, 0.25, 0.5, 0.75, 1.0]

CAPTURE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Markov-adapted recurrent kernels modulate the state space model "
    "along the diffusion trajectory for masked language modeling."
)

KERNEL_MARKOV_OVERRIDES: dict[str, dict[str, Any]] = {
    "Hypernet": {"mark_kernel": "hypernet"},
    "Chebyshev": {"mark_kernel": "chebyshev", "degree": 5},
    "DCT": {"mark_kernel": "dct", "n_freqs": 8, "L_timepoints": 256},
}


def _strip_lightning_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("inner."):
            out[k[len("inner.") :]] = v
        elif k.startswith("module.inner."):
            out[k[len("module.inner.") :]] = v
    return out


def build_input_ids(
    tokenizer: BertTokenizerFast,
    seq_len: int,
    device: torch.device,
    text: str | None = None,
) -> torch.Tensor:
    raw = text if text is not None else CAPTURE_TEXT
    enc = tokenizer(
        raw,
        max_length=seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return enc["input_ids"].to(device)


def load_hydra_for_markov(ckpt_path: str, kernel_name: str, device: str) -> HydraForMaskedLM:
    """Load HydraForMaskedLM with the same conventions as eval_weights.py / dynamics checkpoints."""
    dev = torch.device(device)
    extra = KERNEL_MARKOV_OVERRIDES[kernel_name]
    config = HydraForMaskedLMConfig(
        mark_ensemble=False,
        use_eff_compute=False,
        embedding_dim=128,
        mark_mlp_dim=256,
        max_position_embeddings=4096,
        device=str(dev),
        **extra,
    )
    model = HydraForMaskedLM(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    state_dict = _strip_lightning_prefix(raw)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("load_state_dict missing (%d keys): %s ...", len(missing), missing[:3])
    if unexpected:
        logger.warning("load_state_dict unexpected (%d keys): %s ...", len(unexpected), unexpected[:3])
    model = model.to(dev).eval()
    return model


def set_markov_capture_token_pos(model: HydraForMaskedLM, token_pos: int | None) -> None:
    """Set or clear capture on every Hydra mixer."""
    for i in range(NUM_LAYERS):
        mixer = model.hydra.encoder.layer[i].layer.mixer
        if isinstance(mixer, Hydra):
            mixer.markov_capture_token_pos = token_pos
            if token_pos is None:
                mixer._pre_mark_B = None
                mixer._pre_mark_C = None


def collect_bc_stack_from_model(model: HydraForMaskedLM) -> tuple[np.ndarray, np.ndarray]:
    """
    After one forward with capture enabled, stack pre-MaRK B, C per layer.

    Returns:
        B_stack, C_stack: float32 arrays of shape (NUM_LAYERS, BC_DIM)
    """
    B_stack = np.zeros((NUM_LAYERS, BC_DIM), dtype=np.float32)
    C_stack = np.zeros((NUM_LAYERS, BC_DIM), dtype=np.float32)
    for i in range(NUM_LAYERS):
        mixer = model.hydra.encoder.layer[i].layer.mixer
        if not isinstance(mixer, Hydra):
            raise RuntimeError(f"Layer {i} mixer is not Hydra")
        if mixer._pre_mark_B is None or mixer._pre_mark_C is None:
            raise RuntimeError(f"Layer {i}: pre-MaRK B/C not captured (token index in range?)")
        B_stack[i] = mixer._pre_mark_B.numpy()
        C_stack[i] = mixer._pre_mark_C.numpy()
    return B_stack, C_stack


@torch.no_grad()
def forward_capture_bc_stack(
    model: HydraForMaskedLM,
    tokenizer: BertTokenizerFast,
    device: torch.device,
    seq_len: int,
    token_pos: int,
    text: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single forward with capture; does not load or unload the model."""
    input_ids = build_input_ids(tokenizer, seq_len, device, text=text)
    set_markov_capture_token_pos(model, token_pos)
    _ = model(
        input_ids=input_ids,
        attention_mask=(input_ids != 0),
        current_timestep=500,
        total_timestep=1000,
    )
    B_stack, C_stack = collect_bc_stack_from_model(model)
    set_markov_capture_token_pos(model, None)
    return B_stack, C_stack


@torch.no_grad()
def capture_pre_mark_bc_stack(
    ckpt_path: str,
    kernel_name: str,
    device: str,
    seq_len: int = MARKOV_SEQ_LEN,
    token_pos: int = MARKOV_TOKEN_POS,
    text: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One full forward per call; returns (B_stack, C_stack) with real conv-derived B, C.
    """
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dev = torch.device(device)
    model = load_hydra_for_markov(ckpt_path, kernel_name, device)
    B_stack, C_stack = forward_capture_bc_stack(
        model, tokenizer, dev, seq_len, token_pos, text=text
    )

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return B_stack, C_stack


def snapshot_indices(timesteps: np.ndarray, snapshots: list[float]) -> list[int]:
    out = []
    for s in snapshots:
        idx = int(np.argmin(np.abs(timesteps - s)))
        out.append(idx)
    return out
