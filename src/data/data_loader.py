import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from a given path."""
    return pd.read_csv(filepath)
