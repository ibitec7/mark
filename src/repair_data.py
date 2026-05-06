import os
import glob
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value

def repair_shards(data_dir: str):
    """
    Reads existing parquet shards, removes incompatible Hugging Face metadata,
    casts them to the correct schema, and rewrites them with fresh metadata.
    """
    print(f"\nScanning directory: {data_dir}")
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if not files:
        print(f"  > No parquet files found.")
        return

    features = Features({"input_ids": Sequence(Value("int16"))})

    for f in files:
        filename = os.path.basename(f)
        try:
            table = pq.read_table(f)
            
            if table.schema.metadata and b'huggingface' in table.schema.metadata:
                new_metadata = {k: v for k, v in table.schema.metadata.items() if k != b'huggingface'}
                table = table.replace_schema_metadata(new_metadata)
            
            ds = Dataset(table)
            
            ds = ds.cast(features)
            
            temp_path = f + ".tmp"
            ds.to_parquet(temp_path, compression="lz4")
            
            os.replace(temp_path, f)
            print(f"  [OK] Repaired {filename}")
            
        except Exception as e:
            print(f"  [FAIL] Could not repair {filename}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dirs_to_fix = [
        "./data/train_shards",
        "./data/val_shards",
        "./data/test_shards"
    ]

    print("Starting metadata repair...")
    for d in dirs_to_fix:
        if os.path.exists(d):
            repair_shards(d)
        else:
            print(f"\nSkipping {d} (directory does not exist)")
    
    print("\nRepair complete.")