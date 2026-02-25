"""Data loading utilities for the Flood Detection System."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_dataset(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a tabular dataset from CSV or Excel.

    Args:
        file_path: Path to .csv, .xlsx, or .xls file.
        sheet_name: Excel sheet name (optional).

    Returns:
        Loaded pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        # sep=None lets pandas infer comma/tab/semicolon-delimited text.
        df = pd.read_csv(path, sep=None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df
