"""
Feature engineering on the real per-customer / per-terminal transaction
stream — v2, FIXED.

v1 used pandas' groupby().rolling(), whose output is ordered by group
(all of terminal A's rows, then all of terminal B's rows, ...) rather than
original chronological row order. Concatenating that back positionally
silently scrambled every engineered feature (verified: every feature had
~0.50 AUC against the label, including TERM_RISK, which by construction
should be near-perfect during scenario-2 compromised-terminal windows).

v2 uses an explicit two-pointer rolling computation per group that writes
results directly into an array at the ORIGINAL row position, so there is
no reordering step to get wrong.

Same two feature families as before:
1. CUSTOMER SPENDING BEHAVIOR (trailing 1/7/30 days, causal, excludes the
   current transaction) — count & average amount. Targets scenario 3
   (compromised customer: frequency + amount jump for ~14 days).
2. TERMINAL RISK SCORE with a 7-day feedback delay (fraud confirmations
   arrive late in reality, so risk is computed only from transactions
   confirmed >= 7 days ago). Targets scenario 2 (compromised terminal:
   100% fraud for a 28-day window).
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/all_transactions.parquet")
df = df.sort_values("TX_DATETIME").reset_index(drop=True)
print(f"Loaded {len(df):,} transactions")

TIMES = df["TX_DATETIME"].values.astype("datetime64[s]").astype(np.int64)
DAY = 86400


def rolling_causal(group_col, value_col, window_seconds):
    """
    For each row, compute (count, sum) of `value_col` over all rows of the
    same `group_col` whose TX_DATETIME is in (t - window_seconds, t) —
    i.e. strictly before the current row's time, causal, no leakage.
    Returns two np.arrays (count, sum) aligned to the ORIGINAL df row order.
    """
    n = len(df)
    count_arr = np.zeros(n, dtype=np.float64)
    sum_arr = np.zeros(n, dtype=np.float64)
    vals = df[value_col].values.astype(np.float64)
    groups = df.groupby(group_col).indices  # key -> array of original positions, time-sorted
    for _, idx in groups.items():
        idx = np.asarray(idx)
        t = TIMES[idx]
        left = 0
        run_sum = 0.0
        run_cnt = 0
        for i in range(len(idx)):
            while left < i and t[left] <= t[i] - window_seconds:
                run_cnt -= 1
                run_sum -= vals[idx[left]]
                left += 1
            pos = idx[i]
            count_arr[pos] = run_cnt
            sum_arr[pos] = run_sum
            run_cnt += 1
            run_sum += vals[idx[i]]
    return count_arr, sum_arr


# ---------------------------------------------------------------------
# 1. Customer spending-behavior features
# ---------------------------------------------------------------------
print("Building customer velocity features (1/7/30 day windows)...")
for w in (1, 7, 30):
    cnt, amt_sum = rolling_causal("CUSTOMER_ID", "TX_AMOUNT", w * DAY)
    df[f"CUST_NBTX_{w}D"] = cnt
    df[f"CUST_AVGAMT_{w}D"] = np.where(cnt > 0, amt_sum / np.maximum(cnt, 1), 0.0)

# ---------------------------------------------------------------------
# 2. Terminal risk score with feedback delay
# ---------------------------------------------------------------------
print("Building terminal risk features (delay=7 days, 1/7/30 day windows)...")
DELAY = 7
nbtx_delay, nbfraud_delay = rolling_causal("TERMINAL_ID", "TX_FRAUD", DELAY * DAY)
for w in (1, 7, 30):
    nbtx_dw, nbfraud_dw = rolling_causal("TERMINAL_ID", "TX_FRAUD", (DELAY + w) * DAY)
    nbtx_window = nbtx_dw - nbtx_delay
    nbfraud_window = nbfraud_dw - nbfraud_delay
    risk = np.where(nbtx_window > 0, nbfraud_window / np.maximum(nbtx_window, 1), 0.0)
    df[f"TERM_RISK_{w}D_DELAY{DELAY}D"] = risk
    df[f"TERM_NBTX_{w}D_DELAY{DELAY}D"] = nbtx_window

# ---------------------------------------------------------------------
# 3. Calendar features
# ---------------------------------------------------------------------
df["hour_of_day"] = df["TX_DATETIME"].dt.hour
df["day_of_week"] = df["TX_DATETIME"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["log_amount"] = np.log1p(df["TX_AMOUNT"])

feature_cols = [c for c in df.columns if c.startswith(("CUST_", "TERM_"))]
print(f"\nFinal feature set ({len(feature_cols)} engineered features): {feature_cols}")

# quick sanity check: individual-feature AUC against label (should now show
# real signal, not ~0.50 across the board)
from sklearn.metrics import roc_auc_score
print("\nSanity check — single-feature AUC vs TX_FRAUD:")
for c in feature_cols + ["log_amount"]:
    print(f"  {c:30s} {roc_auc_score(df['TX_FRAUD'], df[c]):.4f}")

df.to_parquet("data/featured_transactions.parquet", index=False)
print(f"\nSaved data/featured_transactions.parquet  shape={df.shape}")
