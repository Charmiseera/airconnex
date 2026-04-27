"""
ml/train.py
===========
Phase 3 — ML Model Training for Smart Connection Feasibility Predictor

Models trained:
  1. Random Forest Classifier  (sklearn)
  2. XGBoost Classifier         (xgboost)

Steps:
  1. Load preprocessing1.csv
  2. Separate features (X) and target (y) without dropping columns
  3. Stratified 80/20 train/test split
  4. MinMax scale features (fit on train only)
  5. Train both models
  6. 5-fold cross-validation
  7. Evaluate on held-out test set (accuracy, precision, recall, F1)
  8. Feature importance
  9. Save best model as ml/model.pkl + scaler as ml/scaler.pkl
  10. Save evaluation report as ml/evaluation_report.txt

Target classes:
  0 = Risky
  1 = Tight
  2 = Safe
"""

import os, sys, warnings, json

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing    import MinMaxScaler
from sklearn.metrics          import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed — only Random Forest will run.")

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # airconnex/
ML_DIR  = os.path.join(BASE, "ml")
DATA    = os.path.join(BASE, "preprocessing1.csv")
REPORT  = os.path.join(ML_DIR, "evaluation_report.txt")
MODEL_F = os.path.join(ML_DIR, "model.pkl")
SCALER_F= os.path.join(ML_DIR, "scaler.pkl")
FEAT_F  = os.path.join(ML_DIR, "feature_names.json")

SEP = "=" * 65
CLASS_NAMES = ["Risky", "Tight", "Safe"]

# ── Helper: pretty section header ─────────────────────────────────────────────
def header(title):
    print(f"\n{SEP}")
    print(title)
    print(SEP)

# ── STEP 1 — Load Data ────────────────────────────────────────────────────────
header("STEP 1 — LOAD PREPROCESSED DATA")

df = pd.read_csv(DATA)
print(f"Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Target distribution:\n{df['risk_label'].map({0:'Risky',1:'Tight',2:'Safe'}).value_counts().to_string()}")

# ── STEP 2 — Feature Selection ────────────────────────────────────────────────
header("STEP 2 — FEATURE SELECTION")

# Keep all features from preprocessing1.csv, only drop the target label
FEATURE_COLS = [c for c in df.columns if c != "risk_label"]
print(f"\nFeatures used ({len(FEATURE_COLS)}):")
for i, c in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {c}")

X = df[FEATURE_COLS].values
y = df["risk_label"].values

# ── STEP 3 — Train / Test Split ───────────────────────────────────────────────
header("STEP 3 — STRATIFIED 80/20 SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train : {X_train.shape[0]:,} rows")
print(f"Test  : {X_test.shape[0]:,}  rows")
print(f"Features: {X_train.shape[1]}")

# ── STEP 4 — Scaling ──────────────────────────────────────────────────────────
header("STEP 4 — MINMAX SCALING (fit on train only)")

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("Scaled to [0, 1] -- no data leakage OK")

# ── STEP 5 — Define Models ────────────────────────────────────────────────────
header("STEP 5 — MODEL CONFIGURATION")

models = {}

models["Random Forest (Unscaled)"] = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = None,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features   = "sqrt",
    class_weight   = "balanced",
    random_state   = 42,
    n_jobs         = -1
)

models["Random Forest (Scaled)"] = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = None,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features   = "sqrt",
    class_weight   = "balanced",
    random_state   = 42,
    n_jobs         = -1
)

if XGBOOST_AVAILABLE:
    models["XGBoost (Scaled)"] = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "mlogloss",
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0
    )

for name, m in models.items():
    print(f"\n  {name}:")
    print(f"    {m.get_params()}")

# ── STEP 6 — Train & Cross-Validate ───────────────────────────────────────────
header("STEP 6 — TRAINING + 5-FOLD CROSS-VALIDATION")

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    print(f"\n  Training {name} ...")
    
    # Choose unscaled or scaled data based on model name
    X_tr = X_train if "Unscaled" in name else X_train_sc
    
    cv_result = cross_validate(
        model, X_tr, y_train,
        cv      = cv,
        scoring = ["accuracy", "f1_macro", "f1_weighted"],
        n_jobs  = -1,
        verbose = 0
    )
    cv_scores[name] = cv_result

    acc_mean  = cv_result["test_accuracy"].mean()
    acc_std   = cv_result["test_accuracy"].std()
    f1_mean   = cv_result["test_f1_macro"].mean()
    f1_std    = cv_result["test_f1_macro"].std()

    print(f"    CV Accuracy  : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"    CV F1-Macro  : {f1_mean:.4f} ± {f1_std:.4f}")

    # Final fit on full training set
    model.fit(X_tr, y_train)
    print(f"    Final fit on {X_tr.shape[0]:,} rows ✓")

# ── STEP 7 — Test Set Evaluation ──────────────────────────────────────────────
header("STEP 7 — HELD-OUT TEST SET EVALUATION")

test_scores = {}

for name, model in models.items():
    # Choose unscaled or scaled test data based on model name
    X_te = X_test if "Unscaled" in name else X_test_sc
    y_pred = model.predict(X_te)

    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro")
    f1_wt   = f1_score(y_test, y_pred, average="weighted")
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)

    test_scores[name] = {
        "accuracy"   : acc,
        "f1_macro"   : f1_mac,
        "f1_weighted": f1_wt,
        "report"     : report,
        "cm"         : cm.tolist(),
    }

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1-Macro     : {f1_mac:.4f}")
    print(f"  F1-Weighted  : {f1_wt:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':10} {'Risky':>8} {'Tight':>8} {'Safe':>8}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    print(f"\n  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")

# ── STEP 8 — Feature Importance ───────────────────────────────────────────────
header("STEP 8 — FEATURE IMPORTANCE")

importance_data = {}

for name, model in models.items():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature"   : FEATURE_COLS,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance_data[name] = fi_df.to_dict(orient="records")

        print(f"\n  {name} — Top 15 Features:")
        print(f"  {'Rank':>4}  {'Feature':<35}  {'Importance':>12}")
        print(f"  {'─'*4}  {'─'*35}  {'─'*12}")
        for rank, row in fi_df.head(15).iterrows():
            print(f"  {rank+1:>4}  {row['feature']:<35}  {row['importance']:>12.4f}")

        # Save importance CSV
        fi_path = os.path.join(ML_DIR, f"feature_importance_{name.replace(' ', '_').lower()}.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"\n  Saved: {fi_path}")

# ── STEP 9 — Pick Best Model ──────────────────────────────────────────────────
header("STEP 9 — BEST MODEL SELECTION")

best_name = max(test_scores, key=lambda n: test_scores[n]["f1_macro"])
best_model = models[best_name]
best_score = test_scores[best_name]

print(f"\n  Comparison (F1-Macro on test set):")
for name, sc in test_scores.items():
    marker = " <-- BEST" if name == best_name else ""
    print(f"    {name:<20}  F1-Macro={sc['f1_macro']:.4f}  Accuracy={sc['accuracy']:.4f}{marker}")

print(f"\n  DONE — Best model: {best_name}")

# ── STEP 10 — Save Artefacts ──────────────────────────────────────────────────
header("STEP 10 — SAVING ARTEFACTS")

joblib.dump(best_model, MODEL_F)
print(f"  model.pkl   → {MODEL_F}")

joblib.dump(scaler, SCALER_F)
print(f"  scaler.pkl  → {SCALER_F}")

with open(FEAT_F, "w") as fp:
    json.dump(FEATURE_COLS, fp, indent=2)
print(f"  feature_names.json → {FEAT_F}")

# ── STEP 11 — Write Evaluation Report ─────────────────────────────────────────
header("STEP 11 — WRITING EVALUATION REPORT")

lines = []
lines.append("=" * 65)
lines.append("SMART CONNECTION FEASIBILITY PREDICTOR — ML EVALUATION REPORT")
lines.append("=" * 65)
lines.append(f"\nDataset : {DATA}")
lines.append(f"Rows    : {df.shape[0]:,}")
lines.append(f"Features: {len(FEATURE_COLS)}")
lines.append(f"\nTarget classes: 0=Risky | 1=Tight | 2=Safe")
lines.append("")

for name in models:
    sc  = test_scores[name]
    cv_ = cv_scores[name]
    lines.append("─" * 55)
    lines.append(f"MODEL: {name}")
    lines.append("─" * 55)
    lines.append(f"  CV Accuracy  : {cv_['test_accuracy'].mean():.4f} ± {cv_['test_accuracy'].std():.4f}")
    lines.append(f"  CV F1-Macro  : {cv_['test_f1_macro'].mean():.4f} ± {cv_['test_f1_macro'].std():.4f}")
    lines.append(f"  Test Accuracy: {sc['accuracy']:.4f}")
    lines.append(f"  Test F1-Macro: {sc['f1_macro']:.4f}")
    lines.append(f"  Test F1-Wtd  : {sc['f1_weighted']:.4f}")
    lines.append("\n  Confusion Matrix:")
    lines.append(f"  {'':10} {'Risky':>8} {'Tight':>8} {'Safe':>8}")
    for i, row in enumerate(sc["cm"]):
        lines.append(f"  {CLASS_NAMES[i]:10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    lines.append("\n  Classification Report:")
    for l in sc["report"].splitlines():
        lines.append(f"    {l}")
    lines.append("")

lines.append("=" * 65)
lines.append(f"BEST MODEL: {best_name}")
lines.append(f"  Accuracy : {best_score['accuracy']:.4f}")
lines.append(f"  F1-Macro : {best_score['f1_macro']:.4f}")
lines.append(f"  F1-Wtd   : {best_score['f1_weighted']:.4f}")
lines.append(f"\nSaved artefacts:")
lines.append(f"  {MODEL_F}")
lines.append(f"  {SCALER_F}")
lines.append(f"  {FEAT_F}")
lines.append("=" * 65)

report_text = "\n".join(lines)

with open(REPORT, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"  Evaluation report -> {REPORT}")

print(f"\n{SEP}")
print("PHASE 3 -- ML TRAINING COMPLETE [DONE]")
print(f"  Best model : {best_name}")
print(f"  Accuracy   : {best_score['accuracy']:.4f}")
print(f"  F1-Macro   : {best_score['f1_macro']:.4f}")
print(f"{SEP}\n")
