import polars as pl
from huggingface_hub import login
import os
import json

with open("token.json", "r") as f:
    TOKEN = json.load(f)["token"]


login(token=TOKEN)

splits = {'test': 'wikitext-2-v1/test-00000-of-00001.parquet', 'train': 'wikitext-2-v1/train-*.parquet', 'validation': 'wikitext-2-v1/validation-00000-of-00001.parquet'}

train_set = pl.scan_parquet('hf://datasets/Salesforce/wikitext/' + splits['train'])
test_set = pl.scan_parquet('hf://datasets/Salesforce/wikitext/' + splits['test'])
validation_set = pl.scan_parquet('hf://datasets/Salesforce/wikitext/' + splits['validation'])

os.makedirs("./data/wikitext-2-v1", exist_ok=True)

train_set.collect().write_parquet("./data/wikitext-2-v1/train.parquet")
test_set.collect().write_parquet("./data/wikitext-2-v1/test.parquet")
validation_set.collect().write_parquet("./data/wikitext-2-v1/validation.parquet")

splits = {'test': 'wikitext-103-v1/test-00000-of-00001.parquet', 'train': 'wikitext-103-v1/train-*.parquet', 'validation': 'wikitext-103-v1/validation-00000-of-00001.parquet'}

train_set = pl.scan_parquet('hf://datasets/Salesforce/wikitext/' + splits['train'])
test_set = pl.scan_parquet('hf://datasets/Salesforce/wikitext/' + splits['test'])
validation_set = pl.scan_parquet('hf://datasets/Salesforce/wikitext/' + splits['validation'])

os.makedirs("./data/wikitext-103-v1", exist_ok=True)

train_set.collect().write_parquet("./data/wikitext-103-v1/train.parquet")
test_set.collect().write_parquet("./data/wikitext-103-v1/test.parquet")
validation_set.collect().write_parquet("./data/wikitext-103-v1/validation.parquet")

# Cleaning done here:
path = "../data"

dirs = ["wikitext-2-v1", "wikitext-103-v1"]

for dir in dirs:

    for file in os.listdir(os.path.join(path, dir)):
        file_path = os.path.join(path, dir, file)

        df: pl.DataFrame = pl.read_parquet(file_path)

        df: pl.DataFrame = df.filter(
            pl.col("text").str.contains(r"\w+")
        )

        df.write_parquet(file_path, compression="lz4")