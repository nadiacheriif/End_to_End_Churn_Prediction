import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

print("=== Phase 2: Modeling with XGBoost ===")

df = pd.read_csv("data/processed/telco_churn_processed.csv")

TARGET_COL = "Churn"
FEATURE_COLS = [col for col in df.columns if col != TARGET_COL]
X = df[FEATURE_COLS]
y = df[TARGET_COL]
print(f"Data loaded. Shape: {df.shape}")
print(f"Features: {FEATURE_COLS}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    return accuracy