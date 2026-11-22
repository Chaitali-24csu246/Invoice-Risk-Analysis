# Invoice Lens — Invoice Risk Analysis

A company-specific academic prototype that reads an invoice, lets the user confirm its details, and predicts **late payment** and **days overdue**. It does not predict permanent non-payment or fraud.

## What is included

- Original demo, PDF/image upload, OCR, manual entry, review and session-history workflow.
- Five selectable classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and XGBoost.
- Three selectable regressors: Ridge, Random Forest Regressor and Gradient Boosting Regressor.
- XGBoost and Random Forest Regressor highlighted and preselected because they won the notebook's validation comparisons.
- Compare models page: validation/test tables, metrics, charts, confusion matrices, fitted parameters and plain-language explanations. Dummy and median baselines are comparison references.
- Same-invoice predictions from every model; history records the selected model names and threshold.
- Eight fitted pipelines included in `models/invoice_models.joblib`. No model download, training or API key is needed to use the app.

## Run locally

Use Python 3.12. From this folder:

```bash
python3 start.py
```

The launcher creates `.venv`, installs requirements and opens the app. Alternatively:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows activate with `.venv\Scripts\activate`. The included Mac/Windows launchers are also available.

For image OCR install Tesseract separately: `brew install tesseract` on Mac, or `sudo apt-get install tesseract-ocr tesseract-ocr-eng` on Ubuntu. On Windows install Tesseract and add its executable to PATH. Text PDFs, prepared demo PDFs and manual entry work without OCR installation.

## Quick demonstration

1. Open **Analyse invoice** and select a prepared demo.
2. Review the fields, keep the recommended models, tick confirmation and analyse.
3. Show the risk score, its cutoff and predicted delay. Reveal the recorded demo outcome.
4. Choose another classifier/regressor and analyse again. Expand **Compare all models on this invoice**.
5. Save both results, then open **Session history** to compare or export them.
6. Open **Compare models**. Explain validation selection, test results and the baseline limitation.

## Model provenance and honest interpretation

The model build reproduces the revised `Invoice_Risk_EndTerm.ipynb` configurations. Hyperparameters were chosen by GridSearchCV in the notebook using time-aware training folds. The build script refits those selected configurations; it does not repeat the search or train on the test set.

Data: 2,466 sample/synthetic invoice records, 12 original columns; 1,390 training, 409 validation and 500 test invoices. Another 167 rows are excluded because outcomes were not known at the relevant split cutoff. Validation begins 2013-02-26; test begins 2013-07-12. All test customers were seen in training.

Preprocessing: categorical imputation and one-hot encoding; numerical median imputation and standardisation; zero-variance feature removal. All transformations fit only on training rows. Five raw candidates become 109 encoded features, of which 108 are retained. Constant 30-day payment terms are removed. Optional app dates are used to derive terms, but the removed feature cannot influence these models.

XGBoost was selected by validation AP (~0.8006). Each classifier has its own validation-F1-maximising threshold; XGBoost's is ~0.2881. Test AP is ~0.7176, recall ~0.7746 and F1 ~0.6832. Scores are not calibrated real-world probabilities.

Random Forest Regressor won validation MAE (~2.7255 days), narrowly beating Ridge. Its test MAE is ~2.3527 days. Ridge (~2.2875) and the zero-day median baseline (~2.2840) have lower overall test MAE. The baseline performs poorly on late invoices. The app displays these results without changing selection based on the test set.

The test period was inspected in previous project iterations. It is a held-out chronological period, not a fresh independent evaluation. Results do not establish performance for other companies, currencies, new customers or different payment terms. Class imbalance does not by itself prove overfitting.

## Reproduce and test

```bash
python scripts/build_models.py
python -m unittest discover -s tests -v
```

The build verifies test AP for all five classifiers and test MAE for all three regressors against the executed notebook's rounded results. `models/metadata.json` records exact package versions and fitted parameters. Evaluation predictions and tables are saved alongside the model bundle. The complete notebook and its separate environment requirements are in `notebooks/`.

## GitHub and later deployment

This folder is ready to replace/update the existing `Chaitali-24csu246/Invoice-Risk-Analysis` repository. Keep `models/`, `demo_invoices/`, `.streamlit/`, `core.py`, `comparison.py`, `app.py`, `requirements.txt` and `packages.txt` together. The model bundle is small enough for ordinary Git; do not exclude it.

Nothing has been pushed or deployed as part of this update. For later Streamlit deployment, the entry point is `app.py`, Python 3.12, and system OCR packages are in `packages.txt`. The optional Dockerfile uses port 7860 for a later Docker/Hugging Face deployment. Docker is not required for local or Streamlit hosting.

Never commit `.venv`, credentials, `.streamlit/secrets.toml` or private uploaded invoices. Only the bundled public sample data/demo documents belong in the repository. Only load the project's trusted model file; pickle/joblib files can execute code.

## Data source

Public IBM accounts-receivable sample distributed at https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories . Source CSV: `WA_Fn-UseC_-Accounts-Receivable.csv`. Demo PDFs reconstruct held-out records; they are not original business invoices. See the notebook for the full preprocessing, search, visualisations and interpretation.
