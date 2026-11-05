# Verification

Checked on 4 November 2026.

## Passed

- All 500 exported test-set classification scores and regression predictions reproduced from the matching invoice inputs, within 1e-10 tolerance.
- 12 automated checks passed: demo prediction parity, text-PDF extraction, PNG OCR, unknown-customer rejection, invalid amount rejection, ambiguous total handling, ambiguous date handling, invalid file rejection, safe CSV export, full demo/save/history/performance flow, required review confirmation and missing manual-input validation.
- Streamlit startup, HTTP health endpoint and main HTML response passed.
- All three demo invoice renders were visually inspected.
- The supplied model and metadata files were preserved without retraining or modification.

## Environment and limits

Local verification used Python 3.12 with the exported NumPy, pandas, SciPy, scikit-learn and joblib versions. The supplied training metadata records Python 3.13; the Dockerfile uses Python 3.13 to match that major/minor version. The selected models require no XGBoost runtime.

Streamlit AppTest exercised the UI behaviour; no browser-based visual review of the web app was performed. PDF layout was inspected. The Docker image was prepared but not built in this environment, and no Hugging Face account deployment was performed. Windows/macOS launcher execution was not available for verification here.

Parsing is intentionally conservative and does not support every invoice layout. User review is required. Text PDFs and the demo work without Tesseract; photo/scanned-PDF OCR needs the system Tesseract executable, installed in the Dockerfile.
