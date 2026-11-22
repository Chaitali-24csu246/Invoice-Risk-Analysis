# Start here

1. Extract the whole project, not just `app.py`.
2. Install Python 3.12 and run `python3 start.py` in the project folder. On Windows use `python start.py`.
3. Start with **Demo invoice**. Check the fields, choose models, tick confirmation and analyse.
4. **Compare models** contains all validation/test results. **Session history** stores results until the browser session ends; download CSV to keep them.

All eight trained models are included. No export, retraining, login or API key is needed. Internet is required for first-time package installation.

If model loading fails, recreate the project's `.venv` using the pinned requirements. Old model files from the previous app are incompatible with this new app; keep this package's `models/` folder intact.

Image scans require Tesseract. If it is unavailable, use a prepared demo PDF, a PDF with selectable text, or manual entry. Poor scans and ambiguous totals/dates should be corrected during review. Unknown customers cannot be scored; do not replace one with an unrelated known customer.

Classifiers and regressors are independent. A likely-on-time result can have a positive estimated delay. The recommended models were chosen on validation data, not because they win every metric. This is a sample-data academic demonstration, not a verified financial decision system.

See README.md for setup, results, reproduction and later deployment details.
