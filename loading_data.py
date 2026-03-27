import os
import pandas as pd
import pyarrow.parquet as pq

def load_parquet(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    table = pq.read_table(file_path, use_pandas_metadata=False)
    return table.to_pandas()


def reverse_mapping(mapping: dict):
    return {v: k for k, v in mapping.items()}