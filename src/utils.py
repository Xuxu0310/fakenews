"""
Utility functions for the fake news detection project.
"""
import random
import numpy as np
import torch
import json
import os


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dataframe(df, title: str, max_rows: int = 20):
    """Print a pandas DataFrame with a title."""
    print_section(title)
    if df is None or len(df) == 0:
        print("Empty DataFrame")
        return
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"\nShowing first {max_rows} rows of {len(df)} total rows.")
    else:
        print(df.to_string(index=False))


def print_dict_pretty(d: dict, title: str):
    """Print a dictionary as JSON."""
    print_section(title)
    print(json.dumps(d, indent=2, ensure_ascii=False))


def ensure_dir(path: str):
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)
