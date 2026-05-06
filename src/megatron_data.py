import os
import torch
import polars as pl
from megatron.core.datasets import indexed_dataset
import numpy as np
import tqdm

parquet_dir = "./data/val_shards_unpacked"
data_dir = "./data/megatron_val_unpacked"
output_prefix = "megatron_data"
vocab_size = 30522

os.makedirs(data_dir, exist_ok=True)

files = os.listdir(parquet_dir)
parquet_files = [f for f in files if f.endswith(".parquet")]

builder = indexed_dataset.IndexedDatasetBuilder(
    bin_path=os.path.join(data_dir, f"{output_prefix}.bin"),
    dtype=np.int16,
)

for file in tqdm.tqdm(parquet_files, desc="Processing parquet files", unit="file"):
    df = pl.read_parquet(os.path.join(parquet_dir, file))
    for tokens in df["input_ids"]:
        builder.add_item(torch.tensor(tokens, dtype=torch.int16))
        builder.end_document()

builder.finalize(os.path.join(data_dir, f"{output_prefix}.idx"))
