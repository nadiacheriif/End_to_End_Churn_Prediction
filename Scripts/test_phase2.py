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
from data.preprocess import preprocess_data
from data.feature_engineering import build_features

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

    #3. Preprocess Data
    print("\n[3] Preprocessing data...")
    df_preprocessed = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data preprocessed. Shape: {df_preprocessed.shape}")
    print(df_preprocessed.head(3))
    #4. Build Features
    print("\n[4] Building features...")
    df_features = build_features(df_preprocessed, target_col=TARGET_COL)
    print(f"Features built. Shape: {df_features.shape}")
    print(df_features.head(3))
if __name__ == "__main__":
    main()

