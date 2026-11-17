import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (average_precision_score, f1_score,
                              confusion_matrix, classification_report,
                              precision_recall_curve)
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Threads for XGBoost/LightGBM. -1 = all cores. Shared hosts (e.g. Streamlit Cloud)
# report many cores but only allow ~2, and oversubscribing them makes training crawl.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))


class FraudEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_weight=0.55, lgb_weight=0.45, random_state=42):
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.random_state = random_state
        self.xgb_model = None
        self.lgb_model = None
        self.feature_names_ = None
        self.threshold_ = 0.5

    def fit(self, X_train, y_train, feature_names=None):
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        n_neg, n_pos = int((y_train==0).sum()), int((y_train==1).sum())
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg/max(n_pos,1),
            eval_metric="aucpr", random_state=self.random_state,
            n_jobs=N_JOBS, verbosity=0,
        )
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=self.random_state,
            n_jobs=N_JOBS, verbose=-1,
        )
        self.xgb_model.fit(X_train, y_train)
        self.lgb_model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        p = (self.xgb_weight * self.xgb_model.predict_proba(X)[:,1] +
             self.lgb_weight * self.lgb_model.predict_proba(X)[:,1])
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= self.threshold_).astype(int)

    def tune_threshold(self, X_val, y_val, beta=1.0):
        probs = self.predict_proba(X_val)[:,1]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where((precision+recall)>0,
                         (1+beta**2)*precision*recall/((beta**2*precision)+recall), 0)
        self.threshold_ = float(thresholds[np.argmax(f[:-1])])
        return self.threshold_


def evaluate(model, X_test, y_test, out_dir: Path = None):
    probs = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)
    auc_pr = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"\nAUC-PR: {auc_pr:.4f} | F1: {f1:.4f} @ threshold={model.threshold_:.2f}")
    print(classification_report(y_test, preds, target_names=["Legit","Fraud"]))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_pr(y_test, probs, auc_pr, out_dir/"pr_curve.png")
        _plot_cm(cm, out_dir/"confusion_matrix.png")
    return {"auc_pr": auc_pr, "f1": f1, "threshold": model.threshold_}


def _plot_pr(y_test, probs, auc_pr, path):
    p, r, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(7,5))
    plt.plot(r, p, lw=2, color="#E63946", label=f"AUC={auc_pr:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Fraud Detection")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _plot_cm(cm, path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_model(model, scaler, feature_names, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir/"ensemble_model.joblib")
    joblib.dump(scaler, out_dir/"scaler.joblib")
    joblib.dump(feature_names, out_dir/"feature_names.joblib")


def load_model(model_dir: Path):
    return (joblib.load(model_dir/"ensemble_model.joblib"),
            joblib.load(model_dir/"scaler.joblib"),
            joblib.load(model_dir/"feature_names.joblib"))
