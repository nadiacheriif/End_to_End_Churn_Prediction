import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import pandas as pd

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probas)
    }

    with mlflow.start_run(run_name="xgb_churn_evaluation") as run:
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # Confusion matrix visualization
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

    print("📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics
