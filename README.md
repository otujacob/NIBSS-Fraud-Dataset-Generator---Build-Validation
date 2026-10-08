# NIBSS-Calibrated Synthetic Inter-Bank Fraud Dataset

Generator, validation code and benchmark for a synthetic, transaction-level dataset of Nigerian inter-bank payments with fraud. The fraud channel mix, monthly profile, technique mix and mean loss per case follow the **2023 NIBSS Annual Fraud Landscape**. Every other parameter is declared as an assumption in the paper (Table "provenance") and in the code (`[ASSUMED]` comments in `generate_dataset.py`).

> **Read this first.** This is synthetic data. We did not have access to real NIBSS transaction records, so nothing here shows that a model trained on this data will work on real transactions. Channel, monthly and technique profiles match NIBSS *by construction*. Fraud prevalence is set to 0.1%, about 118x the real national rate by volume, so that the data are usable for benchmarking.

Paper: *A NIBSS-Calibrated Synthetic Dataset for Nigerian Inter-Bank Fraud Detection* (Otu Jacob, Colin Kuka, Adam Knowles; Wrexham University). Citation details will be added on publication.

## What you get

| Property | Value |
|---|---|
| Rows | 2,102,211 transactions, 1 Jan to 31 Dec 2023 |
| Customers | 25,000 (fixed age group, location, bank, preferred channel, spending level) |
| Fraud | 2,094 (0.0996%) |
| Channels | ATM, E-commerce, Internet banking, Mobile, POS, Web |
| Format | Parquet (about 160 MB) |

Customer behaviour has heterogeneity (log-normal spend and activity), diurnal and weekly rhythm, bursts, and account-takeover fraud bursts. Behavioural features are **strictly causal**: each uses only a customer's earlier transactions, never the current one. Time windows are exact (24 h = 86,400 s; 7 d = 604,800 s). A 90-day burn-in is simulated before 1 January and discarded.

## Quick start

```bash
pip install -r requirements.txt
python generate_dataset.py --customers 25000 --prevalence 0.001 --seed 42 --out data
```

This writes `data/nibss_synthetic_fraud.parquet` (about 30 seconds). Same seed gives the same file.

```python
import generate_dataset as G
df = G.generate(n_customers=25000, prevalence=0.001, seed=42)
```

Or open `demo.ipynb` (also runs in Colab). It is a thin wrapper around `generate_dataset.py` that generates the data, checks it against NIBSS, brute-force verifies causality, plots distributions and runs a small benchmark.

## Reproducing the paper's results

Run from the repository root, in this order:

```bash
python generate_dataset.py --out data     # the dataset
python compare.py                         # LR / RF / XGBoost benchmark, bootstrap CIs -> results/models.json
python validate.py                        # tests and figures -> results/validation.json, figures/
python seeds.py                           # stability over 5 seeds -> results/seeds.json
```

`validate.py` needs the output of `compare.py`, so run them in that order.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | The generator (all logic lives here) |
| `compare.py` | Chronological split; class-weighted logistic regression, random forest, XGBoost; PR-AUC, recall at fixed false-positive rate, precision in top 1%, bootstrap intervals |
| `validate.py` | Comparison with NIBSS, real-data properties, statistical tests, causality brute force, figures |
| `seeds.py` | Repeats generation across seeds to check stability |
| `demo.ipynb` | Walk-through notebook |
| `figures/`, `results/` | Outputs reported in the paper |

## Columns

| Group | Columns |
|---|---|
| Identifiers | `transaction_id`, `customer_id`, `timestamp` |
| Transaction | `amount`, `channel`, `bank`, `merchant_category`, `location`, `age_group` |
| Labels | `is_fraud`; `fraud_technique` (fraud rows only, **do not use as a feature**) |
| Amount | `amount_log`, `amount_vs_mean_ratio` |
| Customer history (prior transactions only) | `amount_sum_24h`, `amount_mean_7d`, `amount_std_7d`, `tx_count_24h`, `amount_mean_total`, `amount_std_total`, `tx_count_total`, `channel_diversity`, `online_channel_ratio`, `velocity_score` |
| Context | `merchant_risk_score`, `is_peak_hour`, `hour_sin/cos`, `day_sin/cos`, `month_sin/cos` |

First-ever transaction of a customer: ratio = 1, velocity = 0, mean = amount.

## Using it for evaluation: please note

- **Split by time, not at random.** The paper uses train to 7 Aug, validation to 18 Oct, test to 31 Dec.
- **Do not report accuracy or a fixed 0.5 threshold.** At 0.1% prevalence they are uninformative. Use PR-AUC, recall at a fixed false-positive rate, or precision in the top 1% of scores, and class weighting.
- **Exclude `fraud_technique`** (it exists only for fraud rows). Consider also excluding **`tx_count_total`**: history length grows through the year (median prior transactions about 56 in training vs 125 in test), so it carries no fraud signal (ROC-AUC 0.49) but differs between splits. `compare.py` keeps it to match the paper's table; `demo.ipynb` drops it.

## Known limitations

- No validation against real transactions; many parameters (customer mix, diurnal and weekly shape, burst parameters, merchant factors) are assumptions.
- Prevalence is about 118x enriched. Within-day timing of fraud is not modelled.
- Residual history drift described above.
- Single-feature separability is modest (maximum ROC-AUC about 0.67) and benchmark scores are low (PR-AUC 0.03 to 0.07 against 0.0007 for random guessing); confidence intervals between models overlap.

See the paper's Discussion section for the full list.

## Dataset deposit

The Parquet file is too large for GitHub. The archived copy and DOI will be added here once deposited. Until then, regenerate it with the command above.

## Licence

To be confirmed with the authors before release (suggested: MIT for code, CC BY 4.0 for data).
