# Sample dataset

`WA_Fn-UseC_-Accounts-Receivable.csv` is the public IBM accounts-receivable sample distributed through https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories . It contains 2,466 rows and 12 original columns. Treat it as sample/synthetic company data, not evidence of production accuracy.

The same original CSV is included beside the academic notebook so it can run without downloading data. The app does not read outcome columns to predict an uploaded invoice. `scripts/build_models.py` uses this file only for explicit offline rebuilding and evaluation.
