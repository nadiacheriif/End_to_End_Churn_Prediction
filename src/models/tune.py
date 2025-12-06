import optuna
import mlflow
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

def tune_model(X: pd.DataFrame, y: pd.Series, n_trials=20):
    """
    Tunes an XGBoost model using Optuna and logs the results with MLflow.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        n_trials (int): Number of trials for Optuna optimization.
    """
    def objective(trial):
        # Define the hyperparameters to tune
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
        
        # Create the XGBoost model
        model = XGBClassifier(**params)
        
        # Evaluate using cross-validation
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        mean_score = scores.mean()
        
        # Log the hyperparameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("mean_recall", mean_score)
        
        return mean_score

    # Start an MLflow run to track the experiment
    with mlflow.start_run(run_name="xgb_optuna_tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value
        
        # Log the best parameters and score to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_recall", best_score)
        
        print(f"🏆 Best Params: {best_params}")
        print(f"🏆 Best Recall: {best_score}")
    
    return study.best_params