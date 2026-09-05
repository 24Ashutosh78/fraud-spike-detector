"""
Fraud-Spike Detector v2 - Layer 1: Transaction-level classifier
==================================================================
Now built on the real Fraud Detection Handbook simulator data (183 days,
Apr-Sep 2018, 1.75M transactions, real CUSTOMER_ID/TERMINAL_ID) instead of
the anonymized ULB dataset used in the first pass. This unlocks genuine
per-entity velocity/risk features (see feature_engineering.py).

Split: calendar-based, time-forward, no shuffling.
  Train: Apr 1 - Jul 31  (4 months)
  Val:   Aug 1 - Aug 31  (1 month, early stopping / threshold tuning)
  Test:  Sep 1 - Sep 30  (1 month, held out, never touched until scoring)
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RNG = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet("data/featured_transactions.parquet")
df = df.sort_values("TX_DATETIME").reset_index(drop=True)
print(f"Loaded {len(df):,} transactions, {df['TX_FRAUD'].sum():,} fraud "
      f"({df['TX_FRAUD'].mean()*100:.3f}%)")
print(f"Date range: {df['TX_DATETIME'].min()} -> {df['TX_DATETIME'].max()}")

train_df = df[df["TX_DATETIME"] < "2018-08-01"]
val_df = df[(df["TX_DATETIME"] >= "2018-08-01") & (df["TX_DATETIME"] < "2018-09-01")]
test_df = df[df["TX_DATETIME"] >= "2018-09-01"]

feature_cols = [c for c in df.columns if c.startswith(("CUST_", "TERM_"))] + \
    ["log_amount", "hour_of_day", "day_of_week", "is_weekend"]

X_train, y_train = train_df[feature_cols], train_df["TX_FRAUD"]
X_val, y_val = val_df[feature_cols], val_df["TX_FRAUD"]
X_test, y_test = test_df[feature_cols], test_df["TX_FRAUD"]

print(f"Train: {len(train_df):,} rows ({y_train.sum():,} fraud) | "
      f"Val: {len(val_df):,} rows ({y_val.sum():,} fraud) | "
      f"Test: {len(test_df):,} rows ({y_test.sum():,} fraud)")

# ---------------------------------------------------------------------
# Train LightGBM (class-weighted for the imbalance)
# ---------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RNG,
    verbosity=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True)],
)

# ---------------------------------------------------------------------
# Evaluate on held-out September test set
# ---------------------------------------------------------------------
test_scores = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_scores)
pr_auc = average_precision_score(y_test, test_scores)
print(f"\nHeld-out test set (September, never seen in training):")
print(f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

precisions, recalls, thresholds = precision_recall_curve(y_test, test_scores)

# ---------------------------------------------------------------------
# Cost-weighted operating point
#   FN cost = actual $ amount of the missed fraudulent transaction
#   FP cost = fixed manual-review / customer-friction cost (assumption)
# ---------------------------------------------------------------------
FP_COST = 4.0  # USD, assumption - tune to real ops cost
total_fraud_value = test_df.loc[test_df["TX_FRAUD"] == 1, "TX_AMOUNT"].sum()

candidate_thresholds = np.linspace(0.005, 0.995, 199)
cost_curve = []
y_test_arr = y_test.values
amount_arr = test_df["TX_AMOUNT"].values
for t in candidate_thresholds:
    preds = (test_scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_arr, preds, labels=[0, 1]).ravel()
    missed_mask = (y_test_arr == 1) & (preds == 0)
    fn_dollar_cost = amount_arr[missed_mask].sum() if missed_mask.any() else 0.0
    fp_dollar_cost = fp * FP_COST
    total_cost = fn_dollar_cost + fp_dollar_cost
    cost_curve.append((t, tp, fp, fn, tn, fn_dollar_cost, fp_dollar_cost, total_cost))

cost_df = pd.DataFrame(cost_curve, columns=[
    "threshold", "tp", "fp", "fn", "tn", "fn_dollar_cost", "fp_dollar_cost", "total_cost"
])
best_row = cost_df.loc[cost_df["total_cost"].idxmin()]
best_t = best_row["threshold"]
best_preds = (test_scores >= best_t).astype(int)
best_precision = precision_score(y_test, best_preds, zero_division=0)
best_recall = recall_score(y_test, best_preds, zero_division=0)
best_f1 = f1_score(y_test, best_preds, zero_division=0)

def metrics_at_threshold(t):
    preds = (test_scores >= t).astype(int)
    return {
        "threshold": float(t),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
    }

idx_90r = int(np.argmin(np.abs(recalls[:-1] - 0.90))) if len(recalls) > 1 else 0
t_90recall = float(thresholds[idx_90r]) if len(thresholds) else 0.5

# feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

summary = {
    "dataset": "Fraud Detection Handbook simulator, 183 days Apr-Sep 2018 (real CUSTOMER_ID/TERMINAL_ID)",
    "n_transactions": int(len(df)),
    "n_fraud_total": int(df["TX_FRAUD"].sum()),
    "fraud_rate_pct": float(df["TX_FRAUD"].mean() * 100),
    "split": {
        "train_rows": int(len(train_df)), "train_fraud": int(y_train.sum()),
        "train_range": "2018-04-01 to 2018-07-31",
        "val_rows": int(len(val_df)), "val_fraud": int(y_val.sum()),
        "val_range": "2018-08-01 to 2018-08-31",
        "test_rows": int(len(test_df)), "test_fraud": int(y_test.sum()),
        "test_range": "2018-09-01 to 2018-09-30",
    },
    "roc_auc_test": float(roc_auc),
    "pr_auc_test": float(pr_auc),
    "cost_model_assumptions": {
        "false_positive_cost_usd": FP_COST,
        "false_negative_cost": "actual $ amount of the missed fraudulent transaction",
    },
    "operating_points": {
        "cost_minimizing": {
            "threshold": float(best_t),
            "precision": float(best_precision),
            "recall": float(best_recall),
            "f1": float(best_f1),
            "total_cost_usd": float(best_row["total_cost"]),
            "fp_count": int(best_row["fp"]),
            "fn_count": int(best_row["fn"]),
            "fn_dollar_cost_usd": float(best_row["fn_dollar_cost"]),
            "fp_dollar_cost_usd": float(best_row["fp_dollar_cost"]),
        },
        "fixed_0.5": metrics_at_threshold(0.5),
        "recall_target_90pct": metrics_at_threshold(t_90recall),
    },
    "naive_baseline_cost_usd_no_detector": float(total_fraud_value),
    "top_10_features": importances.head(10).to_dict(),
}

with open(f"{OUT_DIR}/metrics_summary_v2.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary["operating_points"]["cost_minimizing"], indent=2))
print(f"\nBaseline (no detector) cost on September test set: ${total_fraud_value:,.2f}")
print(f"Cost-minimizing operating point: precision={best_precision:.3f}, "
      f"recall={best_recall:.3f}, remaining cost=${best_row['total_cost']:,.2f} "
      f"({(1 - best_row['total_cost']/total_fraud_value)*100:.1f}% cost reduction)")
print("\nTop 10 features by importance:")
print(importances.head(10))

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(recalls, precisions, color="#c0392b")
axes[0].scatter([best_recall], [best_precision], color="black", zorder=5,
                label=f"cost-min @ t={best_t:.3f}")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title(f"Precision-Recall (PR-AUC={pr_auc:.3f})")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(cost_df["threshold"], cost_df["total_cost"], color="#2980b9")
axes[1].axvline(best_t, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("Decision threshold"); axes[1].set_ylabel("Total cost on test set (USD)")
axes[1].set_title("Cost-weighted threshold selection"); axes[1].grid(alpha=0.3)

top_feats = importances.head(10)[::-1]
axes[2].barh(top_feats.index, top_feats.values, color="#27ae60")
axes[2].set_title("Top 10 feature importances")
axes[2].set_xlabel("LightGBM importance (split count)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/transaction_classifier_metrics_v2.png", dpi=150)
plt.close()

model.booster_.save_model(f"{OUT_DIR}/lgbm_fraud_classifier_v2.txt")
print(f"\nSaved outputs to {OUT_DIR}/")
