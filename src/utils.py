import itertools
import logging
import os
import subprocess
import yaml
import polars as pl

import torch
import torch.nn.functional as F
import math

from omegaconf import DictConfig
from transformers.configuration_utils import PretrainedConfig
from transformers import BertTokenizerFast

from typing import Tuple

from datasets import Dataset, IterableDataset, Features, Sequence, Value, concatenate_datasets, load_dataset

from torch.utils.data import DataLoader, DistributedSampler

TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")
logger = logging.getLogger(__name__)


def estimate_utf8_bytes_from_input_ids(input_ids: torch.Tensor, pad_token_id: int = 0) -> int:
    """
    Estimate the UTF-8 byte count represented by a batch of token IDs.

    Special tokens are skipped and padding is removed before decoding.

    Args:
        input_ids (torch.Tensor): Token IDs of shape [batch, seq] or [seq].
        pad_token_id (int, optional): Padding token ID. Defaults to 0.

    Returns:
        int: Total decoded UTF-8 byte count across the batch.
    """
    if input_ids.dim() == 1:
        rows = [input_ids]
    else:
        rows = input_ids

    total_bytes = 0
    for row in rows:
        ids = row.detach().cpu().tolist()
        ids = [token_id for token_id in ids if token_id != pad_token_id]
        text = TOKENIZER.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        total_bytes += len(text.encode("utf-8"))

    return max(total_bytes, 1)

def load_config(path: str="configs/hydra.yaml", pretrained_config: bool=False, dict_config: bool=False) -> PretrainedConfig | dict:
    """
    Load the transformers PretrainedConfig from a given YAML configuration file.

    Args:
        path (str, optional): Path to the YAML configuration file. Defaults to "./configs/hydra.yaml".

    Returns:
        PretrainedConfig: The loaded configuration.
    """

    config_raw: dict = yaml.safe_load(open(path, "r"))

    if pretrained_config:
        try:
            config: PretrainedConfig = PretrainedConfig(**config_raw["hydra_config"])
        except Exception as e:
            config: PretrainedConfig = PretrainedConfig(**config_raw)
    elif dict_config:
        config: DictConfig = DictConfig(config_raw)
    else:
        hydra_config = config_raw.get("hydra_config", config_raw)
        from .hydra_model import HydraForMaskedLMConfig
        return HydraForMaskedLMConfig(**hydra_config)


    return config


def load_hydra_state_dict(weights_path: str) -> dict[str, torch.Tensor]:
    """
    Load a flat ``HydraForMaskedLM`` state dict from a ``.pt`` file or Lightning ``.ckpt``.

    ``.pt`` files store keys like ``hydra.*`` / ``cls.*``. Checkpoints from NeMo/Lightning
    wrap the model as ``inner``; those keys are stripped to match ``HydraForMaskedLM``.
    """
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    if str(weights_path).endswith(".ckpt"):
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        elif isinstance(raw, dict) and "state" in raw:
            sd = raw["state"]
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
        else:
            sd = raw
        out: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("inner."):
                out[k[len("inner.") :]] = v
        if not out:
            raise ValueError(f"No 'inner.' keys found in checkpoint: {weights_path}")
        return out
    return raw


def load_compatible_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    """
    Load only the keys whose names and tensor shapes match the target module.

    Args:
        module (torch.nn.Module): Module receiving the weights.
        state_dict (dict[str, torch.Tensor]): Candidate weights to load.

    Returns:
        tuple[list[str], list[str], list[str]]: Missing keys, unexpected keys, and
        mismatched-shape keys that were skipped.
    """
    module_state = module.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    mismatched_keys: list[str] = []

    for key, value in state_dict.items():
        if key not in module_state:
            unexpected_keys.append(key)
            continue

        target_value = module_state[key]
        if tuple(value.shape) != tuple(target_value.shape):
            mismatched_keys.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(target_value.shape)}"
            )
            continue

        compatible_state[key] = value

    incompatible = module.load_state_dict(compatible_state, strict=False)
    return list(incompatible.missing_keys), unexpected_keys + list(incompatible.unexpected_keys), mismatched_keys


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

CONFIG = load_config("configs/training_config.yaml", dict_config=True)

seq_len = CONFIG.get("pad_length", 512)

del CONFIG

def log_setup(log_name: str, log_file: str, level) -> logging.Logger:
    """
    Setup logging configuration for the training process.

    Args:
        log_name (str): The name of the logger.
        log_file (str): The file where logs will be saved.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """

    if not os.path.exists(log_file):
        os.makedirs("logs", exist_ok=True)
        subprocess.run(["touch", log_file])

    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.propagate = False

    return logger

def get_grad_norm(model, norm_type=2.0):
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    return total_norm.item()

def get_seq_idx(input_ids: torch.Tensor, cls_id: int = 101, pad_id: int = 0) -> torch.Tensor:
    """
    Build seq_idx for Mamba chunked scan from packed sequences.

    Args:
        input_ids : (batch, seq_len) — packed token ids
        cls_id : token id marking the start of a new sequence (default 101)
        pad_id : token id used for padding (default 0)

    Returns:
        seq_idx : (batch, seq_len) int32 — 0-based sequence index per position;
                    padding positions are marked with -1
    """

    is_cls = (input_ids == cls_id) # (B, L) bool

    seq_idx = is_cls.cumsum(dim=-1) - 1 # (B, L) 0-indexed now

    is_pad = (input_ids == pad_id)

    pad_start = is_pad.long().argmax(dim=-1) # (B,) index of first pad
    has_pad = is_pad.any(dim=-1) # (B,) rows that actually have padding

    positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0) # (1, L)
    trailing_pad_mask = (positions >= pad_start.unsqueeze(1)) & has_pad.unsqueeze(1) # (B, L)

    seq_idx = seq_idx.masked_fill(trailing_pad_mask, -1)

    return seq_idx.to(torch.int32)

def enable_cudnn_optimizations():
    """Enable CUDA and cuDNN optimizations."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[INFO] Enabled cuDNN optimizations (TF32)")

@torch.no_grad()
def record_batch(
    batch_id: int,
    epoch: int,
    loss: float,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masked_indices: torch.Tensor,
    current_step: int,
    total_steps: int,
    model: torch.nn.Module=None,
    optimizer: torch.optim.Optimizer=None,
    scaler: torch.cuda.amp.GradScaler=None,
) -> dict:
    """
        Logs metrics for one mini-batch.

        Parameters:
            logits : [batch, seq_len, vocab_size] predictions from the model
            labels : [batch, seq_len] ground truth token IDs
            masked_indices : [batch, seq_len] boolean mask of masked tokens
            p_mask : [batch, seq_len] masking probabilities
    """

    vocab_size = logits.shape[-1]

    total_tokens = masked_indices.numel()
    masked_tokens = masked_indices.sum().item()
    masked_ratio = masked_tokens / total_tokens

    pred_flat = logits.view(-1, vocab_size)
    target_flat = labels.view(-1)
    mask_flat = masked_indices.view(-1)

    # Masked metrics
    masked_logits = pred_flat[mask_flat]
    masked_targets = target_flat[mask_flat]
    masked_probs = torch.softmax(masked_logits, dim=-1)
    masked_pred = masked_probs.argmax(dim=-1)

    masked_acc = (masked_pred == masked_targets).float().mean().item()
    masked_nll = F.cross_entropy(masked_logits, masked_targets, reduction="mean").item()
    masked_ppl = math.exp(masked_nll)

    eps = 1e-12
    entropy_masked = (-masked_probs * (masked_probs + eps).log()).sum(dim=-1).mean().item()

    # Full metrics
    full_probs = torch.softmax(pred_flat, dim=-1)
    full_pred = full_probs.argmax(dim=-1)
    full_acc = (full_pred == target_flat).float().mean().item()
    full_nll = F.cross_entropy(pred_flat, target_flat, reduction="mean").item()
    full_ppl = math.exp(full_nll)

    entropy_full = (-full_probs * (full_probs + eps).log()).sum(dim=-1).mean().item()

    if scaler is not None:
        scaler.unscale_(optimizer) if optimizer is not None else None

    grad_norm = get_grad_norm(model) if model is not None else None

    return dict({
        "batch_id": batch_id,
        "epoch": epoch,
        "loss": loss,
        "masked_tokens": masked_tokens,
        "total_tokens": total_tokens,
        "masked_ratio": masked_ratio,
        "current_step": current_step,
        "total_steps": total_steps,
        "masked_acc": masked_acc,
        "masked_nll": masked_nll,
        "masked_ppl": masked_ppl,
        "entropy_masked": entropy_masked,
        "full_acc": full_acc,
        "full_nll": full_nll,
        "full_ppl": full_ppl,
        "entropy_full": entropy_full,
        "grad_norm": grad_norm,
    })

def get_best_checkpoint(checkpoint_dir: str) -> str:
    """Find the checkpoint with the best (lowest) validation loss."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    
    # If using save_last=True, 'last.ckpt' is always the most recent
    if 'last.ckpt' in checkpoints:
        return os.path.join(checkpoint_dir, 'last.ckpt')
    
    # Otherwise, sort by modification time
    latest = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest)

class TrainingMetrics():
    def __init__(self, dir_name: str="./data/training", filename: str="hydra_batch_data.parquet"):
        self.records = []
        self.save_dir = dir_name
        self.file_name = filename

    @staticmethod
    def _get_grad_norm(model, norm_type=2.0):
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0
        norm_type = float(norm_type)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
        return total_norm.item()

    @torch.no_grad()
    def record_batch(
        self,
        *,
        batch_id: int,
        epoch: int,
        loss: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masked_indices: torch.Tensor,
        current_step: int,
        total_steps: int,
        model: torch.nn.Module=None,
        optimizer: torch.optim.Optimizer=None,
        scaler: torch.cuda.amp.GradScaler=None,
    ):
        """
            Logs metrics for one mini-batch.

            Parameters:
                logits : [batch, seq_len, vocab_size] predictions from the model
                labels : [batch, seq_len] ground truth token IDs
                masked_indices : [batch, seq_len] boolean mask of masked tokens
                p_mask : [batch, seq_len] masking probabilities
        """

        _, _, vocab_size = logits.shape

        total_tokens = masked_indices.numel()
        masked_tokens = masked_indices.sum().item()
        masked_ratio = masked_tokens / total_tokens

        pred_flat = logits.view(-1, vocab_size)
        target_flat = labels.view(-1)
        mask_flat = masked_indices.view(-1)

        # Masked metrics
        masked_logits = pred_flat[mask_flat]
        masked_targets = target_flat[mask_flat]
        masked_probs = torch.softmax(masked_logits, dim=-1)
        masked_pred = masked_probs.argmax(dim=-1)

        masked_acc = (masked_pred == masked_targets).float().mean().item()
        masked_nll = F.cross_entropy(masked_logits, masked_targets, reduction="mean").item()
        masked_ppl = math.exp(masked_nll)

        eps = 1e-12
        entropy_masked = (-masked_probs * (masked_probs + eps).log()).sum(dim=-1).mean().item()

        # Full metrics
        full_probs = torch.softmax(pred_flat, dim=-1)
        full_pred = full_probs.argmax(dim=-1)
        full_acc = (full_pred == target_flat).float().mean().item()
        full_nll = F.cross_entropy(pred_flat, target_flat, reduction="mean").item()
        full_ppl = math.exp(full_nll)

        entropy_full = (-full_probs * (full_probs + eps).log()).sum(dim=-1).mean().item()

        if scaler is not None:
            scaler.unscale_(optimizer) if optimizer is not None else None

        grad_norm = self._get_grad_norm(model) if model is not None else None
        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None

        self.records.append({
            "batch_id": batch_id,
            "epoch": epoch,
            "loss": loss,
            "masked_tokens": masked_tokens,
            "total_tokens": total_tokens,
            "masked_ratio": masked_ratio,
            "current_step": current_step,
            "total_steps": total_steps,
            "masked_acc": masked_acc,
            "masked_nll": masked_nll,
            "masked_ppl": masked_ppl,
            "entropy_masked": entropy_masked,
            "full_acc": full_acc,
            "full_nll": full_nll,
            "full_ppl": full_ppl,
            "entropy_full": entropy_full,
            "grad_norm": grad_norm,
            "lr": lr,
        })

    def save(self):
        df = pl.DataFrame(self.records)
        path = os.path.join(self.save_dir, self.file_name)
        os.makedirs(self.save_dir, exist_ok=True)
        df.write_parquet(path, compression="lz4")

    def reset(self):
        self.records.clear()


def collect_batch_data(df: dict, loss: float, mini_batch_size: int, mini_batch: torch.Tensor,\
        masked_indices: torch.Tensor, current_step: int, total_steps: int, i: int, j: int):
    
    df["batch_id"].append(i // mini_batch_size * j)
    df["loss"].append(loss)
    df["masked_tokens"].append(masked_indices.sum().item())
    df["total_tokens"].append(mini_batch.numel())
    df["masked_ratio"].append(masked_indices.float().mean().item())
    df["current_step"].append(current_step)
    df["total_steps"].append(total_steps)

def tokenize_fast(row):
    """Tokenizes the dataset to Pytorch Tensors."""

    tokens = TOKENIZER(row["text"], truncation=True, padding=False, max_length=4096, return_attention_mask=False, return_token_type_ids=False)
    row["input_ids"] = tokens["input_ids"]
    # row["attention_mask"] = tokens["attention_mask"]

    del row["text"]
    
    return row

def packing(batch, max_length=4096):
        """
        Packs sequences from the batch greedily: concatenates sequences until adding the next would exceed max_length.
        Sorts sequences by length ascending for better packing. Returns packed sequences <= max_length, no truncation/loss.
        Only packs 'input_ids'; no attention_mask handling.
        """
        sequences = batch["input_ids"]
        # Sort by sequence length ascending (greedy packing)
        sequences.sort(key=len)
        
        packed_input_ids = []
        current_input = []
        current_len = 0
        
        for inp in sequences:
            seq_len = len(inp)
            if current_len + seq_len <= max_length:
                # Add to current pack
                current_input.extend(inp)
                current_len += seq_len
            else:
                # Finalize current pack and start new one
                if current_input:
                    packed_input_ids.append(current_input)
                current_input = inp[:]
                current_len = seq_len

        # Add the last pack
        if current_input:
            packed_input_ids.append(current_input)

        # Pad each packed sequence to max_length with 0s
        for i in range(len(packed_input_ids)):
            seq_len = len(packed_input_ids[i])
            if seq_len < max_length:
                packed_input_ids[i] += [0] * (max_length - seq_len)

        assert len(packed_input_ids) == 4096 or all(len(seq) == max_length for seq in packed_input_ids), "All packed sequences must be of max_length"

        return {"input_ids": packed_input_ids}

def download_dataset(max_length: int = 4096):
    from datasets import Dataset
    import glob
    import shutil

    datasets = [
        "common-pile/stackexchange_filtered",
        "common-pile/libretexts_filtered",
        "common-pile/youtube_filtered",
        "common-pile/pubmed_filtered",
        "common-pile/cccc_filtered",
        "common-pile/project_gutenberg_filtered",
        "common-pile/arxiv_papers_filtered",
        "common-pile/news_filtered",
        "common-pile/doab_filtered",
        "common-pile/pressbooks_filtered",
        "iohadrubin/wikitext-103-raw-v1",
        "common-pile/data_provenance_initiative_filtered",
        "common-pile/wikimedia_filtered",
        "common-pile/arxiv_abstracts_filtered"
    ]

    actual_ratios = {
        "common-pile/stackexchange_filtered": float(12.0/ 120.0),
        "common-pile/libretexts_filtered": float(2.0/120.0),
        "common-pile/youtube_filtered": float(27.0/120.0),
        "common-pile/pubmed_filtered": float(9.0/120.0),
        "common-pile/cccc_filtered": float(12.0/120.0),
        "common-pile/project_gutenberg_filtered": float(6.0/120.0),
        "common-pile/arxiv_papers_filtered": float(9.0/120.0),
        "common-pile/news_filtered": float(2.0/120.0),
        "common-pile/doab_filtered": float(18.0/120.0),
        "common-pile/pressbooks_filtered": float(2.0/120.0),
        "iohadrubin/wikitext-103-raw-v1": float(3.0/120.0),
        "common-pile/data_provenance_initiative_filtered": float(10.0/120.0),
        "common-pile/wikimedia_filtered": float(10.0/120.0),
        "common-pile/arxiv_abstracts_filtered": float(4.0/120.0)
    }

    features = Features({"input_ids": Sequence(feature=Value("int16"))})

    ratios = [0.1, float(5.0/300.0), 0.225, 0.075, 0.1, 0.05, 0.075, float(3.0/300.0), 0.15, float(3.0/300.0), 0.025, float(10.0/120.0), float(10.0/120.0), float(4.0/120.0)]
    rows = [int(1_200_000 * r) for r in ratios]

    data_dir = "./data/train_shards"
    shard_size = 10_000
    os.makedirs(data_dir, exist_ok=True)

    # Check for existing shards to resume globally
    existing_shards = glob.glob(os.path.join(data_dir, "shard_*.parquet"))
    if existing_shards:
        shard_indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_shards]
        shard_idx = max(shard_indices) + 1
        print(f"Resuming from global shard {shard_idx}")
    else:
        shard_idx = 0

    # continuing from shard idx 111.
    datasets = datasets
    rows = rows

    # Process one dataset at a time
    for i, ds_name in enumerate(datasets):
        print(f"Processing dataset {i}: {ds_name}")
        
        ds_iter: IterableDataset = load_dataset(ds_name, split="train", streaming=True)

        ds_iter = ds_iter.select_columns(["text"]).shuffle(buffer_size=10_000, seed=42)

        ds = Dataset.from_generator(lambda: (yield from ds_iter), features=Features({"text": Value("string")}), num_proc=8)

        del ds_iter

        features = Features({"input_ids": Sequence(feature=Value("int16"))})

        ds = ds.map(tokenize_fast, batched=False, num_proc=8, keep_in_memory=False, features=features)

        features = Features({"input_ids": Sequence(feature=Value("int16"))})

        ds = ds.map(packing, fn_kwargs={"max_length": max_length}, batched=True, batch_size=5000, num_proc=8, keep_in_memory=False, features=features)

        try:
            ds = ds.shuffle(seed=42).take(rows[i])
        except Exception as e:
            ds = ds.shuffle(seed=42)

        # Convert to IterableDataset and shard directly
        ds_iterable = ds.to_iterable_dataset()
        ds_iter = iter(ds_iterable)

        # Shard this dataset's processed data
        while True:
            shard = list(itertools.islice(ds_iter, shard_size))
            if not shard:
                break
            
            shard_ds = Dataset.from_list(shard)
            shard_path = os.path.join(data_dir, f"shard_{shard_idx}.parquet")
            shard_ds.to_parquet(shard_path, compression="lz4")
            
            print(f"Saved shard {shard_idx} to {shard_path}")
            shard_idx += 1

        # Clear memory
        del ds, ds_iterable, ds_iter

        # Delete cache for this dataset to free disk space
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        # dataset_cache_name = ds_name.replace("/", "___")
        for dataset_cache_name in os.listdir(cache_dir):
            shutil.rmtree(os.path.join(cache_dir, dataset_cache_name), ignore_errors=True)
        print(f"Deleted cache for {ds_name}")

    # After all datasets, handle val/test splitting
    print(f"Total shards saved: {shard_idx}")

    # ds = load_dataset("parquet", data_files=os.path.join(data_dir, "shard_*.parquet"), split="train")
    # ds = ds.shuffle(seed=42)

    # for i in range(shard_idx):
        

    test_dir = "./data/test_shards"
    val_dir = "./data/val_shards"

    shard_ids = torch.tensor([i for i in range(shard_idx)])
    train_shards_mask = torch.where(torch.rand(shard_idx) < 0.7, True, False)

    shard_ids = shard_ids[~train_shards_mask].tolist()

    val_shards_mask = torch.where(torch.rand(len(shard_ids)) < 0.5, True, False)

    shard_ids = torch.tensor(shard_ids)

    val_shards = shard_ids[val_shards_mask].tolist()
    test_shards = shard_ids[~val_shards_mask].tolist()

    os.makedirs(val_dir, exist_ok=True)

    for idx in val_shards:
        src_path = os.path.join(data_dir, f"shard_{idx}.parquet")
        dst_path = os.path.join(val_dir, f"shard_{idx}.parquet")
        os.rename(src_path, dst_path)

    os.makedirs(test_dir, exist_ok=True)

    for idx in test_shards:
        src_path = os.path.join(data_dir, f"shard_{idx}.parquet")
        dst_path = os.path.join(test_dir, f"shard_{idx}.parquet")
        os.rename(src_path, dst_path)

def arrow_dataloader(
        data_dir: str,
        split: str = "train",
        batch_size: int = 8,
        num_workers: int = 4,
        keep_in_memory: bool = True,
        **dl_kwargs
):
    import glob

    def load_parquet_dataset_with_metadata_fallback(parquet_paths: list[str]) -> Dataset:
        import pyarrow.parquet as pq

        tables = []
        for parquet_path in parquet_paths:
            table = pq.read_table(parquet_path)
            metadata = table.schema.metadata or {}
            if b"huggingface" in metadata:
                metadata = {key: value for key, value in metadata.items() if key != b"huggingface"}
                table = table.replace_schema_metadata(metadata or None)
            tables.append(Dataset(table))

        ds = tables[0] if len(tables) == 1 else concatenate_datasets(tables)

        if "input_ids" not in ds.column_names:
            raise KeyError(f"Expected an 'input_ids' column in benchmark parquet files under {data_dir}")

        extra_columns = [column for column in ds.column_names if column != "input_ids"]
        if extra_columns:
            ds = ds.remove_columns(extra_columns)

        return ds.cast(Features({"input_ids": Sequence(feature=Value("int16"))}))

    # Support both flat and nested directory structures
    # ds = load_dataset("parquet", data_files=os.path.join(data_dir, f"*.parquet"), split="train", streaming=False, keep_in_memory=keep_in_memory)
    parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)
    if not parquet_files:
        # Fallback to non-recursive search
        parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    try:
        ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=False, keep_in_memory=keep_in_memory)
    except TypeError as exc:
        if "must be called with a dataclass type or instance" not in str(exc):
            raise
        logger.warning(
            "Falling back to manual parquet loading for %s because embedded Hugging Face metadata is incompatible: %s",
            data_dir,
            exc,
        )
        ds = load_parquet_dataset_with_metadata_fallback(parquet_files)

    ds = ds.with_format("torch")

    # Check if distributed training is initialized before using DistributedSampler
    # use_distributed = torch.cuda.device_count() > 1
    use_distributed = (
        torch.distributed.is_available() 
        and torch.distributed.is_initialized() 
        and torch.cuda.device_count() > 1
    )
    
    sampler = None
    if split == "train":
        sampler = DistributedSampler(ds, shuffle=True, drop_last=False) if use_distributed else None

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train") and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            **dl_kwargs
        )
    else:
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if use_distributed else None

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train") and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            **dl_kwargs
        )

    return loader

def pick_factors_near_sqrt(L: int) -> Tuple[int,int]:
    # find integer p,q such that p*q == L and |p-q| minimal
    # if exact factorization impossible, pick nearest by searching divisors
    best = None
    for p in range(int(math.sqrt(L)), 0, -1):
        if L % p == 0:
            q = L // p
            best = (p, q)
            break
    if best is None:
        # fallback: pick p = floor(sqrt(L)), q = ceil(L/p) (allow non-exact reshape by padding/trimming)
        p = int(math.sqrt(L))
        q = math.ceil(L / p)
        best = (p, q)
    return best