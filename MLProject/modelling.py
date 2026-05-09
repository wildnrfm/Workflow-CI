"""
modelling.py - untuk MLflow Project
mlflow run akan inject MLFLOW_RUN_ID dan tracking URI otomatis.
Script ini tidak perlu set tracking URI atau experiment sendiri.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    classification_report
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH   = os.path.join(BASE_DIR, "diabetes_preprocessing", "train.csv")
TEST_PATH    = os.path.join(BASE_DIR, "diabetes_preprocessing", "test.csv")
TARGET_COL   = "Outcome"
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

PARAM_GRID = {
    "n_estimators"    : [50, 100, 200],
    "max_depth"       : [4, 6, 8, None],
    "min_samples_split": [2, 5],
}


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train = train.drop(TARGET_COL, axis=1)
    y_train = train[TARGET_COL]
    X_test  = test.drop(TARGET_COL, axis=1)
    y_test  = test[TARGET_COL]
    print(f"[data] train={X_train.shape} | test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Tidak Diabetes", "Diabetes"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def save_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#4C9BE8")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def save_roc_curve(model, X_test, y_test, path):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C9BE8", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#ccc")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def train():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    print("[tuning] Mulai GridSearchCV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        PARAM_GRID, cv=cv, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model  = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_f1  = grid.best_score_

    print(f"[tuning] Best params : {best_params}")
    print(f"[tuning] Best CV F1  : {best_cv_f1:.4f}")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy : {acc:.4f} | Precision : {prec:.4f}")
    print(f"Recall   : {rec:.4f} | F1        : {f1:.4f} | AUC : {auc:.4f}")

    # Simpan artefak
    cm_path     = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    fi_path     = os.path.join(ARTIFACT_DIR, "feature_importance.png")
    roc_path    = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")

    save_confusion_matrix(y_test, y_pred, cm_path)
    save_feature_importance(best_model, list(X_train.columns), fi_path)
    save_roc_curve(best_model, X_test, y_test, roc_path)

    with open(report_path, "w") as f:
        f.write(f"Best Params: {best_params}\n\n")
        f.write(classification_report(y_test, y_pred,
                target_names=["Tidak Diabetes", "Diabetes"]))

    # MLflow logging — tidak set tracking URI, biarkan mlflow run yang handle
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy",        acc)
        mlflow.log_metric("precision",       prec)
        mlflow.log_metric("recall",          rec)
        mlflow.log_metric("f1_score",        f1)
        mlflow.log_metric("roc_auc",         auc)
        mlflow.log_metric("best_cv_f1",      best_cv_f1)
        mlflow.log_metric("n_train_samples", len(y_train))
        mlflow.log_metric("n_test_samples",  len(y_test))

        mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_artifact(cm_path,     artifact_path="plots")
        mlflow.log_artifact(fi_path,     artifact_path="plots")
        mlflow.log_artifact(roc_path,    artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="reports")

        print(f"\n[mlflow] Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("=" * 50)
    print("  MODELLING — Workflow CI")
    print("=" * 50)
    train()
    print("Selesai!")
