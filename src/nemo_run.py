from dataclasses import dataclass

import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset
from nemo.collections import llm
from nemo.collections.llm import BaseMambaConfig130M
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
try:
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
except ModuleNotFoundError:
    from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
import nemo_run as run
import torch
import os


def dataset_steps(data_prefix: str, seq_length: int, global_batch_size: int) -> int:
    """Return the number of optimizer steps to consume every token in the dataset once."""
    ds = IndexedDataset(data_prefix)
    total_tokens = int(np.sum(ds.index.sequence_lengths))
    return total_tokens // seq_length // global_batch_size

torch.manual_seed(42)

@dataclass
class BertMambaConfig130M(BaseMambaConfig130M):
    """BaseMambaConfig130M with BERT tokenizer (vocab 30522 → padded to 30528).

    Also enables full activation recomputation to fit on small GPUs (≤ 4 GiB).
    """
    tokenizer_library: str = "huggingface"
    tokenizer_name: str = "bert-base-uncased"
    make_vocab_size_divisible_by: int = 16
    seq_length: int = 4096
    # Activation recompute: trades compute for memory; recommended for large seq_length
    # recompute_granularity: str = "full"
    # recompute_method: str = "uniform"
    # recompute_num_layers: int = 24


if __name__ == "__main__":
    os.makedirs("/workspace/checkpoints/guider", exist_ok=True)

    SEQ_LENGTH = 4096
    GLOBAL_BATCH_SIZE = 8
    TRAIN_PREFIX = "/workspace/data/megatron_train_data/megatron_data"
    VAL_PREFIX   = "/workspace/data/megatron_val_data/megatron_data"

    train_steps = dataset_steps(TRAIN_PREFIX, SEQ_LENGTH, GLOBAL_BATCH_SIZE)
    val_steps   = dataset_steps(VAL_PREFIX,   SEQ_LENGTH, GLOBAL_BATCH_SIZE)

    recipe = llm.mamba2_130m.finetune_recipe(
        resume_path="/workspace/models/mamba2_130m_nemo",
        dir="/workspace/checkpoints/guider",
        name="guider_finetuning_run",
        num_nodes=1,
        num_gpus_per_node=1 if torch.cuda.is_available() else 0,
    )

    if not torch.cuda.is_available():
        recipe.trainer.accelerator = "cpu"
        recipe.trainer.devices = 1

    # Override model to use BERT tokenizer matching the converted checkpoint vocab (30528)
    recipe.model.config = run.Config(BertMambaConfig130M)
    recipe.model.tokenizer = run.Config(
        get_nmt_tokenizer,
        library="huggingface",
        model_name="bert-base-uncased",
        use_fast=True,
    )

    recipe.trainer.max_steps = train_steps
    recipe.trainer.max_epochs = 1
    recipe.trainer.val_check_interval = train_steps
    recipe.trainer.limit_val_batches = val_steps

    recipe.data = run.Config(
        PreTrainingDataModule,
        paths={
            "train": [TRAIN_PREFIX],
            "validation": [VAL_PREFIX],
            "test": [VAL_PREFIX],
        },
        seq_length=SEQ_LENGTH,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=1,
        num_workers=8,
    )

    run.run(recipe, executor=run.LocalExecutor())