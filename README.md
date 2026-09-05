# Fraud-Spike Detector — AI Risk Manager Track 

**Loss class:** Card-transaction fraud (defense-only)
**Deliverable:** A working two-layer detector with measured precision/recall and cost-weighted analysis on a held-out, time-forward, calendar-based test set.

This is a rebuild of the first pass on the anonymized public ULB dataset — that version is kept in `v1_ulb_dataset/` for reference. This version uses the user's own 183 daily files, a much better fit for this track because it has real customer/terminal entity IDs.

---

## 1. Dataset

**Fraud Detection Handbook simulator** (Le Borgne, Siblini, Lebichot, Bontempi) — 183 daily pickle files, **April 1 – September 30, 2018**:

- **1,754,155 transactions**, **14,681 confirmed fraud (0.837%)**
- **4,990 customers, 10,000 terminals** — real entity IDs (this is the key upgrade over the ULB dataset, which was anonymized with no per-card/per-merchant ID)
- Three injected fraud scenarios: (1) random high-value transactions (>$220), (2) **compromised terminal** — a terminal goes 100% fraudulent for a 28-day window, (3) **compromised customer** — ~1/3 of a customer's transactions are replaced with fraud, amounts inflated ~5x, for a 14-day window

Scenarios 2 and 3 are genuine "spike" patterns tied to real entities — exactly what this track is asking for, and exactly what the ULB dataset couldn't provide.

**Data-loading note:** the files were pickled with an old pandas version (`Int64Index`, removed in pandas 2.0+). `consolidate_data.py` includes a small compatibility shim to load them cleanly and merges all 183 days into one chronological file.

---

## 2. Method — two complementary layers

### Layer 1: Transaction-level classifier

LightGBM binary classifier using:
- `log(amount)`, hour-of-day, day-of-week, weekend flag
- **Causal per-customer velocity** (trailing 1/7/30 days, excludes the current transaction): transaction count, average amount — targets scenario 3
- **Causal per-terminal risk score with a 7-day feedback delay**: fraud rate in a trailing window that ends 7 days ago, not today — because in reality, confirmed-fraud labels (chargebacks/disputes) arrive days late, so a real-time model can't see "was this terminal fraudulent in the last 3 days." Targets scenario 2.

**Split:** calendar-based, time-forward, no shuffling — train on **April–July**, validate on **August**, test on **September** (never touched until final scoring).

### Layer 2: Entity-level spike detector (unsupervised)

A rolling z-score control chart per entity — never looks at the fraud label. Two iterations are documented below because the first one taught us something about the data:

- **Daily granularity, count+amount combined → weak** (terminal precision 4.2%, recall 31%). Cause: average terminal traffic is under 1 transaction/day, so daily counts are mostly Poisson noise.
- **Weekly granularity, amount-only → much better.** Two fixes: (a) weekly buckets average out the sparse-count noise; (b) amount, not count, turns out to be the dominant signal — mechanically sensible, since a "compromised terminal" doesn't necessarily see *more* transactions, it just makes existing ones fraudulent, while a "compromised customer" inflates amounts ~5x. Amount z-score alone beat count z-score at every threshold tested.

Final design: weekly total $ volume per entity, z-score against a trailing 8-week causal baseline, flagged if z > threshold.

### Defense-only scope
Both layers only score/flag transactions or entity-weeks for review. Neither layer, nor any output of this system, provides a way to construct, launder, or evade detection of a fraudulent transaction.

---

## 3. Results

### Layer 1 — Transaction classifier (held-out test: September 2018, 287,873 transactions, 2,547 fraud)

| Metric | Value |
|---|---|
| ROC-AUC | **0.902** |
| PR-AUC (average precision) | **0.720** |

**Cost model** (FN cost = actual $ lost per missed fraud; FP cost = **$4** flat assumption for manual-review/friction — tune to your real ops cost):

| Operating point | Threshold | Precision | Recall | FP | FN | Total cost (Sept) |
|---|---|---|---|---|---|---|
| **Cost-minimizing** | 0.81 | **55.2%** | **75.5%** | 1,562 | 623 | **$40,826** |
| Fixed 0.5 | 0.50 | 24.8% | 79.0% | 6,119 | 534 | higher (more FP cost) |
| 90%-recall target | 0.164 | 1.7% | 90.0% | 131,782 | 255 | very high (FP-dominated) |
| **No detector** | — | — | 0% | 0 | 2,547 | **$329,881** |

At the cost-minimizing threshold: **87.6% reduction in fraud-related cost** ($329,881 → $40,826) on a full unseen month, at 55% precision and 76% recall — sending ~0.6% of legitimate September transactions to review.

**Top features:** customer average-amount (7D/30D) and log-amount lead, followed closely by the terminal risk/volume features — confirming both engineered signal families earn their place.

*(Charts: `outputs/transaction_classifier_metrics_v2.png`)*

### Layer 2 — Entity-level spike detector (test weeks: Aug 25 – Sep 30, 2018)

| Entity | Threshold | Precision | Recall | Flagged |
|---|---|---|---|---|
| Terminal | z > 2 | 11.3% | 28.9% | 3,450 |
| Terminal | z > 3 | 15.4% | 15.2% | 1,330 |
| Terminal | z > 4 | 20.9% | 8.2% | 526 |
| Customer | z > 2 | 15.3% | 12.4% | 1,647 |
| Customer | z > 3 | 20.0% | 6.1% | 615 |
| Customer | z > 4 | 30.3% | 4.0% | 267 |

Baseline fraud rate among entity-weeks is ~0.6–1.5%, so even the loosest threshold (z>2) gives a genuine 10-20x lift over random flagging — while operating on **zero label information**, purely from transaction volume behavior. This is deliberately a complementary early-warning layer to Layer 1, not a replacement: it would keep working even before enough labeled fraud accumulates to train a classifier, and it's a plausible product surface on its own ("this terminal's weekly volume just tripled — review before more damage").

*(Charts: `outputs/spike_entity_chart.png` — shows a real compromised terminal example with its weekly $ volume spike aligned to actual fraud weeks)*

---

## 4. Why report it this way (honest-metrics bar)

- **Calendar time-forward split**, not random — the model only ever sees the past when scoring the future, mirroring real deployment.
- **Cost-weighted threshold selection**, not accuracy/F1 — a merchant cares about dollars, not F-scores. The $4 FP-cost assumption is stated explicitly and is the one number to swap for your real ops cost.
- **Two dead ends kept visible, not edited out**: the early-stopping bug (LightGBM was tracking `binary_logloss` instead of `auc` and quitting after 1 tree — silently capping ROC-AUC at 0.65 instead of 0.90) and the daily-granularity spike detector (Poisson noise, 4% precision) are both documented in the code comments and this README rather than deleted, because the fixes and the reasoning behind them are as informative as the final numbers.
- **Layer 2 reported at multiple thresholds**, not just the best-looking one, so the precision/recall trade-off is visible rather than cherry-picked.

---

## 5. Files

This package is self-contained — the data is included, so it re-runs end to end with no external upload needed.

- `data/daily_raw.zip` — the original 183 daily `.pkl` files, zipped (the raw source data)
- `data/all_transactions.parquet` — those 183 files already merged into one chronological table (output of `consolidate_data.py` — included so you can skip straight to feature engineering if you don't want to re-run the merge)
- `consolidate_data.py` — loads all 183 daily `.pkl` files (with pandas-compat shim for the old pandas version they were pickled with), merges into `data/all_transactions.parquet`
- `data/featured_transactions.parquet` — output of `feature_engineering.py`: the transaction table with the customer-velocity and terminal-risk features already added (included so `pipeline_v2.py` can be run directly without rebuilding features)
- `feature_engineering.py` — causal customer-velocity and delayed terminal-risk features → `data/featured_transactions.parquet` (already included above, but this script regenerates it in ~1-2 min from `data/all_transactions.parquet` if you want to change anything)
- `pipeline_v.py` — Layer 1: LightGBM training, calendar split, cost-weighted evaluation
- `spike_detector_v.py` — Layer 2: weekly amount-based entity spike detector
- `outputs/metrics_summary_v.json` — full Layer 1 metrics and cost breakdown
- `outputs/spike_entity_metrics_v.json` — full Layer 2 metrics
- `outputs/transaction_classifier_metrics_v.png`, `outputs/spike_entity_chart.png` — plots
- `outputs/lgbm_fraud_classifier_v.txt` — saved trained model


Every stage's output is included, so nothing has to be rebuilt to inspect or re-score — but every script that produced them is included too, so any stage can be re-run and reproduced from scratch.

## 6. How to reproduce

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib pyarrow

# data/all_transactions.parquet and data/featured_transactions.parquet are
# already included, so the two steps below are optional - only needed if
# you want to rebuild from the raw daily files or change the features:
unzip -o data/daily_raw.zip -d . (Not required I unzip the file and data is present in data folder)
python consolidate_data.py
python feature_engineering.py

# required - these produce the outputs/ folder (already included, but this
# regenerates it):
python pipeline_v2.py
python spike_detector_v2.py
```

## 7. Honest next steps (scoped out for time)

- Layer 2 precision is still modest in absolute terms (11-30%) — a Poisson-based control chart (rather than a Gaussian z-score) would likely handle the sparse per-entity counts more rigorously than the current approach
- Combine Layer 1's score with Layer 2's entity-level flag as a joint feature/ensemble, rather than reporting them separately
- SHAP explanations per flagged transaction for a reviewer-facing "why was this flagged" workflow
- Calibrate the $4 FP cost against real review-team throughput
- The 90%-recall operating point shows how fast precision collapses (1.7%) — worth stress-testing whether a two-stage review queue (cheap auto-decline above one threshold, human review in a band below it) beats a single threshold
