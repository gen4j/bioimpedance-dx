"""
scripts/train.py

Training pipeline for the osteomyelitis bioimpedance classifier.

Every run of this script is a fully tracked MLflow experiment.
Nothing is trained without being logged. This is the foundation
of reproducibility — a regulatory requirement, not a nice-to-have.

Usage:
    poetry run python scripts/train.py
"""

import os
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from bioimpedance_dx.data import (
    FEATURE_COLUMNS,
    INT_TO_LABEL,
    SEVERITY_ORDER,
    load_normalized_dataset,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.getenv("DATA_PATH", "data/raw/Normalized_NEw_data.xlsx")
MLFLOW_URI = os.getenv("MLFLOW_URI", "sqlite:///mlflow.db")
RANDOM_SEED = 42
TEST_SIZE = 0.20

HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

SHORT_NAMES = ["Normal", "Type1", "Type2", "Type3", "Type4"]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute sensitivity and specificity for each class.

    Sensitivity = TP / (TP + FN)
        The most critical metric for a diagnostic device.
        A missed bone infection (false negative) is the worst outcome.

    Specificity = TN / (TN + FP)
        Correctly clearing healthy patients.
        A false positive causes unnecessary treatment.

    FDA reviewers will ask for both, per class, with confidence intervals.
    This function computes the point estimates. CIs come later.
    """
    cm = confusion_matrix(y_true, y_pred)
    metrics: dict[str, dict[str, float]] = {}

    for i, name in enumerate(SHORT_NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metrics[name] = {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    return metrics


def compute_macro_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> float:
    """Macro-averaged One-vs-Rest ROC AUC for multiclass problems."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train() -> None:
    print("=" * 60)
    print("Osteomyelitis Bioimpedance Classifier — Training")
    print("=" * 60)

    # 1. Load validated data
    print(f"\n[1/6] Loading data from: {DATA_PATH}")
    X, y = load_normalized_dataset(DATA_PATH)
    n_features = X.shape[1]
    n_classes = len(SEVERITY_ORDER)

    # 2. Stratified train/test split
    # Stratified means class proportions are preserved in both sets.
    # Without this, random chance could give you a test set with
    # fewer Type4 samples, making your metrics look better than they are.
    print(f"\n[2/6] Splitting: {int((1-TEST_SIZE)*100)}% train / "
          f"{int(TEST_SIZE*100)}% test (stratified, seed={RANDOM_SEED})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    for i, name in enumerate(SHORT_NAMES):
        print(f"  [{name}] train={(y_train==i).sum()}, test={(y_test==i).sum()}")

    # 3. Configure MLflow
    print(f"\n[3/6] Configuring MLflow at: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("osteomyelitis-bioimpedance-classifier")

    # 4. Train inside MLflow run
    print(f"\n[4/6] Training RandomForest...")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")

        # Log everything about this run so it can be reproduced exactly
        mlflow.log_params(HYPERPARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("feature_columns", str(FEATURE_COLUMNS))
        mlflow.log_param("data_path", DATA_PATH)

        start = time.time()
        model = RandomForestClassifier(**HYPERPARAMS)
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")

        # 5. Evaluate on held-out test set
        print(f"\n[5/6] Evaluating...")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        accuracy = float((y_pred == y_test).mean())
        auc = compute_macro_auc(y_test, y_prob, n_classes)
        per_class = compute_per_class_metrics(y_test, y_pred)

        macro_sens = float(np.mean([per_class[n]["sensitivity"] for n in SHORT_NAMES]))
        macro_spec = float(np.mean([per_class[n]["specificity"] for n in SHORT_NAMES]))

        print(f"\n  Accuracy:          {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  Macro AUC:         {auc:.4f}")
        print(f"  Macro Sensitivity: {macro_sens:.4f}")
        print(f"  Macro Specificity: {macro_spec:.4f}")

        print(f"\n  {'Class':<10} {'Sens':>8} {'Spec':>8} "
              f"{'PPV':>8} {'NPV':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"  {'-'*60}")
        for name in SHORT_NAMES:
            m = per_class[name]
            print(f"  {name:<10} {m['sensitivity']:>8.4f} {m['specificity']:>8.4f} "
                  f"{m['ppv']:>8.4f} {m['npv']:>8.4f} "
                  f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

        print(f"\n{classification_report(y_test, y_pred, target_names=SHORT_NAMES)}")

        # Log all metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_auc", auc)
        mlflow.log_metric("macro_sensitivity", macro_sens)
        mlflow.log_metric("macro_specificity", macro_spec)
        mlflow.log_metric("training_time_s", elapsed)

        for name in SHORT_NAMES:
            m = per_class[name]
            mlflow.log_metric(f"sensitivity_{name}", m["sensitivity"])
            mlflow.log_metric(f"specificity_{name}", m["specificity"])
            mlflow.log_metric(f"ppv_{name}", m["ppv"])
            mlflow.log_metric(f"npv_{name}", m["npv"])

        # 6. Save model
        print(f"[6/6] Saving model...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="osteomyelitis-bioimpedance-classifier",
            input_example=X_test[:3],
        )

        Path("models").mkdir(exist_ok=True)
        model_path = Path("models") / f"classifier_{run_id[:8]}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(str(model_path))

        print(f"\n{'='*60}")
        print(f"Complete. Run ID: {run_id}")
        print(f"View UI: poetry run mlflow ui  →  http://localhost:5000")
        print(f"{'='*60}")


if __name__ == "__main__":
    train()
