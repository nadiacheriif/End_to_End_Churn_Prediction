import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score

print("=== Phase 2: Modeling with XGBoost ===")

df = pd.read_csv("data/processed/telco_churn_processed.csv")

# target must be numeric 0/1
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

# Convert all object dtype columns to numeric using pd.get_dummies
# This ensures XGBoost can work with the data
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
if object_cols:
    print(f"Converting {len(object_cols)} object columns to numeric...")
    df = pd.get_dummies(df, columns=object_cols, drop_first=True, dtype=int)
    print(f"Shape after encoding: {df.shape}")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.4

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, pos_label=1)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)
mlflow.log_params(study.best_params)
best_model = XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)    
mlflow.sklearn.log_model(best_model, "xgb_model")
proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (proba >= THRESHOLD).astype(int)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=4))
mlflow.log_metric("test_recall", recall_score(y_test, y_pred, pos_label=1))
mlflow.log_metric("test_precision", precision_score(y_test, y_pred, pos_label=1))
#print path to the logged model
model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgb_model"
print(f"Model logged in run {mlflow.active_run().info.run_id} at {model_uri}")      
print("✅ Model training and logging complete.")