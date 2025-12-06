import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Run a lightweight validation of the Telco Churn dataset using pandas.

    This function intentionally avoids a dependency on Great Expectations
    (which may have different APIs across versions) and performs a set of
    pragmatic checks that match the project's Phase 1 expectations.
    """
    print("🔍 Starting data validation...")

    failed: List[str] = []

    # === Schema validation ===
    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    for col in required_columns:
        if col not in df.columns:
            failed.append(f"missing_column:{col}")
        else:
            if df[col].isnull().any():
                failed.append(f"nulls_in_column:{col}")

    # === Business logic / categorical checks ===
    def check_in_set(col: str, allowed: List[str]):
        if col not in df.columns:
            return
        invalid = ~df[col].isin(allowed)
        if invalid.any():
            pct = float(invalid.mean())
            failed.append(f"invalid_values_in_{col}:{pct:.2f}")

    check_in_set("gender", ["Male", "Female"])
    check_in_set("Partner", ["Yes", "No"])
    check_in_set("Dependents", ["Yes", "No"])
    check_in_set("PhoneService", ["Yes", "No"])
    check_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    check_in_set("InternetService", ["DSL", "Fiber optic", "No"])

    # === Numeric checks ===
    def to_numeric(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    if "tenure" in df.columns:
        tenure = to_numeric("tenure")
        if tenure.isnull().any():
            failed.append("non_numeric:tenure")
        else:
            if (tenure < 0).any() or (tenure > 120).any():
                failed.append("tenure_out_of_bounds")

    if "MonthlyCharges" in df.columns:
        monthly = to_numeric("MonthlyCharges")
        if monthly.isnull().any():
            failed.append("non_numeric:MonthlyCharges")
        else:
            if (monthly < 0).any() or (monthly > 200).any():
                failed.append("MonthlyCharges_out_of_bounds")

    if "TotalCharges" in df.columns:
        total = to_numeric("TotalCharges")
        if total.isnull().any():
            # The original dataset has some blank TotalCharges entries; warn but don't fail
            print("⚠️  Warning: TotalCharges has non-numeric values; will be handled in preprocessing")
        else:
            if (total < 0).any():
                failed.append("TotalCharges_negative")

    # === Pairwise logical check ===
    if set(["TotalCharges", "MonthlyCharges"]).issubset(df.columns):
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        valid_mask = total.notnull() & monthly.notnull()
        if valid_mask.any():
            proportion = float((total[valid_mask] >= monthly[valid_mask]).mean())
            if proportion < 0.9:
                failed.append(f"total_vs_monthly_ratio:{proportion:.2f}")

    success = len(failed) == 0

    if success:
        print("✅ Data validation PASSED")
    else:
        print("❌ Data validation FAILED")
        print("Failed checks:", failed)

    return success, failed


