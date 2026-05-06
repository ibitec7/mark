import datasets
from datasets import Features, Sequence, Value
from utils import packing, tokenize_fast
from transformers import BertTokenizerFast
import polars as pl
import os

def tokenizer_txt(file_path: str, prefix: str, seq_length: int = 4096) -> None:
    """Tokenize a raw text file and chunk into fixed-length `input_ids` blocks."""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    with open(file_path, "r") as file:
        content = file.read()

    encoded = tokenizer(
        content,
        truncation=False,  # tokenize everything, then chunk
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    chunks: list[list[int]] = []
    for i in range(0, len(input_ids), seq_length):
        chunk = input_ids[i : i + seq_length]
        chunk[0] = 101 # CLS token
        chunk[-1] = 102 # SEP token
        if len(chunk) < seq_length:
            chunk = chunk + [0] * (seq_length - len(chunk))  # pad to fixed length
        chunks.append(chunk)

    out_path = os.path.join(
        os.path.dirname(file_path),
        f"{prefix}_{os.path.basename(file_path).split('.')[0]}.parquet",
    )
    pl.DataFrame({"input_ids": chunks}).write_parquet(out_path, compression="lz4")

if __name__ == "__main__":
    prefix = "packed"

    directories = [ "data/benchmarks/arxiv", "data/benchmarks/pubmed"]

    for directory in directories:

        if directory == "data/benchmarks/ptb":
            tokenizer_txt(os.path.join(directory, "ptb.test.txt"), prefix=prefix)
            continue
        
        for file in os.listdir(directory):
            # Skip previously packed outputs; they don't have a `text` column.
            if file.startswith(f"{prefix}_"):
                continue
            if not file.endswith(".parquet"):
                continue
            
            shard_path = os.path.join(directory, file)
            ds = datasets.load_dataset("parquet", data_files=shard_path, split="train")

            # Tokenize: `tokenize_fast` expects a `text` column and creates `input_ids`.
            # Don't force an output schema here; datasets may contain extra columns.
            ds = ds.map(tokenize_fast, batched=False, num_proc=8, keep_in_memory=False)

            # Packing only needs `input_ids`. Dropping everything else avoids schema mismatches.
            drop_cols = [c for c in ds.column_names if c != "input_ids"]
            if drop_cols:
                ds = ds.remove_columns(drop_cols)

            out_features = Features({"input_ids": Sequence(feature=Value("int16"))})
            ds = ds.map(
                packing,
                fn_kwargs={"max_length": 4096},
                batched=True,
                batch_size=5000,
                num_proc=8,
                keep_in_memory=False,
                features=out_features,
            )

            ds.to_parquet(os.path.join(directory, f"{prefix}_{os.path.basename(file).split('.')[0]}.parquet"), compression="lz4")