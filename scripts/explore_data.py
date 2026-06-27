"""
Data exploration script.
Run with: poetry run python scripts/explore_data.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")

def explore_file(path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"FILE: {path.name}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KB")
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn names:\n{list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nBasic statistics:\n{df.describe()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")

    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn names:\n{list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nBasic statistics:\n{df.describe()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")

    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        print(f"Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")

    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        print(f"Keys: {list(data.keys())}")
        for key in data.keys():
            arr = data[key]
            print(f"  [{key}] shape={arr.shape}, dtype={arr.dtype}")

    elif suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else data[0]}")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
    else:
        print("Unknown format — open manually")

def main() -> None:
    files = list(DATA_DIR.iterdir())
    if not files:
        print(f"No files found in {DATA_DIR}. Copy your data there first.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) in data/raw/")
    for f in sorted(files):
        if f.name != ".gitkeep":
            explore_file(f)

    print(f"\n{'='*60}")
    print("DONE — share this output so we can build the data loader.")

if __name__ == "__main__":
    main()