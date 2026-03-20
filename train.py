"""
Heart Disease Classifier — Training Script
Dataset: UCI Heart Disease (Cleveland) — heart_disease.csv
Run:     python train.py
Outputs: model.pkl, features.json, evaluation_plots.png
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)

# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading dataset from heart_disease.csv ...")
df = pd.read_csv("heart_disease.csv")
print(f"Shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}\n")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

df = df.dropna(subset=FEATURES + ["target"])
X = df[FEATURES].astype(float)
y = df["target"]

# ── 2. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

# ── 3. Pipeline: scaler + model ───────────────────────────────────────────────

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# ── 4. Cross-validation ───────────────────────────────────────────────────────

print("Running 5-fold stratified cross-validation ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
print(f"CV F1 scores : {cv_f1.round(3)}")
print(f"Mean CV F1   : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}\n")

# ── 5. Final fit ──────────────────────────────────────────────────────────────

pipeline.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy" : round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall"   : round(recall_score(y_test, y_pred), 4),
    "f1"       : round(f1_score(y_test, y_pred), 4),
    "roc_auc"  : round(roc_auc_score(y_test, y_proba), 4),
    "cv_f1_mean": round(float(cv_f1.mean()), 4),
    "cv_f1_std" : round(float(cv_f1.std()), 4),
}

print("── Test-set metrics ────────────────────────────────")
for k, v in metrics.items():
    print(f"  {k:<14}: {v}")
print()
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ── 7. Plots ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
axes[0].set_title("Confusion matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150)
print("Saved evaluation_plots.png")

# Feature importance
importances = pipeline.named_steps["clf"].feature_importances_
feat_df = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(feat_df.round(4).to_string())

# ── 8. Save artefacts ─────────────────────────────────────────────────────────

joblib.dump(pipeline, "model.pkl")
print("\nModel saved → model.pkl")

with open("features.json", "w") as f:
    json.dump({"features": FEATURES, "metrics": metrics}, f, indent=2)
print("Metadata saved → features.json")
