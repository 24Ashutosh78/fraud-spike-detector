"""
Consolidate the 183 daily .pkl files (Fraud Detection Handbook simulator
format: TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT,
TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD, TX_FRAUD_SCENARIO) into a single
chronological parquet file for downstream feature engineering.

The files were pickled with an old pandas version that used Int64Index
(removed in pandas 2.0+). We register a small compatibility shim so they
unpickle cleanly in the current environment.
"""
import glob
import sys
import types
import pandas as pd


mod = types.ModuleType("pandas.core.indexes.numeric")
class Int64Index(pd.Index): pass
class Float64Index(pd.Index): pass
class UInt64Index(pd.Index): pass
mod.Int64Index = Int64Index
mod.Float64Index = Float64Index
mod.UInt64Index = UInt64Index
sys.modules["pandas.core.indexes.numeric"] = mod


files = sorted(glob.glob("data/*.pkl"))
print(f"Found {len(files)} daily files: {files[0]} .. {files[-1]}")

frames = []
bad_files = []
for fp in files:
    try:
        df = pd.read_pickle(fp)
        frames.append(df)
    except Exception as e:
        bad_files.append((fp, str(e)))

print(f"Loaded {len(frames)} files OK, {len(bad_files)} failed")
for fp, err in bad_files[:10]:
    print("FAILED:", fp, err)

full = pd.concat(frames, ignore_index=True)

# dtype cleanup - some files may have TX_TIME_SECONDS/DAYS as object
for col in ("TX_TIME_SECONDS", "TX_TIME_DAYS", "TRANSACTION_ID", "TX_FRAUD", "TX_FRAUD_SCENARIO"):
    full[col] = pd.to_numeric(full[col], errors="coerce").astype("int64")
full["TX_AMOUNT"] = pd.to_numeric(full["TX_AMOUNT"], errors="coerce")
full["TX_DATETIME"] = pd.to_datetime(full["TX_DATETIME"])
full["CUSTOMER_ID"] = full["CUSTOMER_ID"].astype(str)
full["TERMINAL_ID"] = full["TERMINAL_ID"].astype(str)

full = full.sort_values("TX_DATETIME").reset_index(drop=True)

print(f"\nCombined: {len(full):,} transactions")
print(f"Date range: {full['TX_DATETIME'].min()} -> {full['TX_DATETIME'].max()}")
print(f"Fraud: {full['TX_FRAUD'].sum():,} ({full['TX_FRAUD'].mean()*100:.3f}%)")
print(f"Customers: {full['CUSTOMER_ID'].nunique():,}  Terminals: {full['TERMINAL_ID'].nunique():,}")
print(full['TX_FRAUD_SCENARIO'].value_counts())

full.to_parquet("data/all_transactions.parquet", index=False)
print("\nSaved data/all_transactions.parquet")
