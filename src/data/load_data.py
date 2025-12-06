import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    print(f"📥 Loading: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    return df