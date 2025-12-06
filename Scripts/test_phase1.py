# test_pipeline_phase1.py
import os
import pandas as pd
from pathlib import Path

# Make sure Python can find your src package (project-root/src)
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data.load_data import load_data
from data.validate_data import validate_telco_data

# === CONFIG ===
DATA_PATH = str(Path(__file__).resolve().parents[1] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET_COL = "Churn"

def main():
    print("=== Testing Phase 1: Load → Validate ===")

    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head(3))

    # 2. Validate Data
    print("\n[2] Validating data...")
    validate_telco_data(df)

    print("\n✅ Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()
