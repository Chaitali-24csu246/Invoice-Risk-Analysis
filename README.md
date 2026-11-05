---
title: Invoice Lens
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Invoice Lens

A company-specific invoice payment-risk demo using the supplied Logistic Regression and Ridge models. Upload a PDF/photo, review the fields, run both models, and save results for the session. Three prepared invoices make the classroom demonstration reproducible.

**Start with [START_HERE.md](START_HERE.md).**

## Run locally

Use Python 3.13 (recommended). Python 3.12 was used for the supplied verification with matching model package versions. Run `python start.py`. On Windows, double-click `START_WINDOWS.bat`. On macOS, use `python3 start.py` from Terminal in this folder.

The launcher installs dependencies once and opens the app. Text PDFs and the built-in demo work without Tesseract. Install Tesseract for photo/scanned-PDF OCR, or use the included Docker image, which installs it automatically.

## Hugging Face Spaces

Create a new Docker Space and upload **the contents of this folder**, preserving the `models/`, `demo_invoices/` and `.streamlit/` directories. Do not upload the ZIP as the only file. Keep this README and Dockerfile at the root. Choose an available CPU tier; a GPU is unnecessary. The repository build starts the app on port 7860. No token belongs in the source code. See START_HERE.md for browser and CLI upload instructions.

## Model inputs

`customerID`, `countryCode`, `PaperlessBill`, `InvoiceAmount`.

Unknown categorical values and invalid amounts are blocked. Dates and invoice numbers are review fields only. Billing method is confirmed from the invoice/record, not inferred from the uploaded file format. No customer history or outcomes are fabricated or fetched during prediction. The score is uncalibrated; the exact saved threshold is used. Both models run independently and negative regression predictions are clipped at zero, matching the notebook.

## Data and limits

All training and recorded evaluation originated in the user's synthetic/sample-data project. Test invoices are from known customers; these results do not validate new companies or arbitrary receipts. Data currency is unspecified. Do not relabel the units as rupees/dollars or claim cross-currency reliability. Prediction concerns lateness, not permanent non-payment or fraud. Scanning is label-based extraction and needs user review, especially on unfamiliar layouts.

The original model files and evaluation files are unchanged from the user's export. XGBoost was a comparison candidate only; it is not needed to load the selected scikit-learn models.

The three demo PDFs/PNGs were reconstructed from held-out invoice IDs using the public mirror below and cross-checked against the exported predictions. They are newly formatted sample documents, not authentic company invoices. One case intentionally illustrates a false alert. Outcomes are withheld from document text and prediction inputs and revealed separately after scoring.

Data mirror used to recover the original invoice fields:
https://github.com/JVGITHUBANALYTICS/MLOps-E2E/blob/master/Lab43/WA_Fn-UseC_-Accounts-Receivable.csv

The source dataset listing supplied by the user did not establish a redistribution license. This educational package contains only the three reconstructed demonstration records and the user's supplied evaluation artifacts; it does not claim a license for the source dataset. Use your own authorised company data for a real deployment.

## Session behaviour

Uploaded bytes and history are stored in the active Streamlit session, not a database or shared upload directory. Download CSV results before ending the session. Refresh/reconnect or a hosting restart can lose session state. The model resource is shared read-only; uploaded content is not globally cached. No external OCR/LLM API, API key, or paid inference service is required.

## Verification

Run `python -m unittest discover -s tests -v` after installing requirements. See VERIFICATION.md for results and environment limitations.
