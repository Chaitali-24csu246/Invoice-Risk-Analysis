# Start here

## 1. Unzip the project

Extract the ZIP fully. Open the `invoice-risk-app` folder. Leave the `models` and `demo_invoices` folders in place.

## 2. Start on your computer

Install Python 3.13 if needed.

- **Windows:** double-click `START_WINDOWS.bat`.
- **Mac:** open Terminal in the extracted folder and run `python3 start.py`.
- **Linux:** run `python3 start.py`.

The first launch downloads dependencies and may take several minutes. Subsequent launches reuse them. Keep the terminal window open while using the app. If the browser does not open, copy the local URL printed in the terminal. Ctrl+C stops the app.

Alternative, if you already manage a Python environment:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Photo scanning

Text PDFs and the built-in demo PDFs work immediately. Photos and scanned PDFs also need the Tesseract program:

- macOS with Homebrew: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows: install Tesseract using the options linked in https://tesseract-ocr.github.io/tessdoc/Installation.html and add its installation folder to PATH. Reopen the terminal afterwards.

If OCR is unavailable, use a demo PDF, a text PDF or Enter manually. Hugging Face's Docker build includes Tesseract.

## 3. Give the classroom demo

1. Open **Analyse invoice → Demo invoice → Example 1**.
2. Point out the invoice on the left and editable extracted fields on the right.
3. Confirm the details and click **Analyse payment risk**.
4. Explain the payment class, uncalibrated risk score and estimated days overdue.
5. Open **Reveal the recorded demo outcome**.
6. Save the result. Repeat with Example 2 and Example 3. The third demonstrates a false alert.
7. Open **Session history** and download a CSV.
8. Open **Model performance** to explain test recall, precision and regression errors.
9. To demonstrate actual file upload, download a demo PDF and choose Upload invoice. Use a matching PNG from `demo_invoices` to demonstrate OCR. Outcome reveal is available in Demo invoice mode only.

Demo documents use the historical test dates on purpose. They simulate prediction at invoice issue. The system does not interpret an already overdue historical invoice as a new live invoice today.

## 4. Host temporarily on Hugging Face

This project is prepared for hosting but has not been published to your account.

1. Sign in at https://huggingface.co and select **New Space**.
2. Choose a name, such as `invoice-lens`, and **Docker** as the SDK. Use a blank Docker template if offered.
3. Choose visibility deliberately. Public Spaces expose their repository, including these synthetic demo/evaluation files. Use private visibility if you do not want others to see them.
4. Select an available CPU tier. You do not need a GPU.
5. In the Files tab, upload all project contents and preserve subfolders. The root must contain `README.md`, `Dockerfile`, `app.py`, `core.py` and `requirements.txt`. Upload `models` and `demo_invoices` too.
6. Commit the upload and wait for the build. Open the App tab once running.
7. Share the Space URL when ready. If using private visibility, other people may not have access.

If browser folder upload is inconvenient, use this alternative from the project folder:

```bash
python -m pip install huggingface_hub
hf auth login
hf upload YOUR_USERNAME/invoice-lens . . --repo-type space --exclude '.venv/**' '__pycache__/**' '*.zip'
```

Create the Space first and replace `YOUR_USERNAME/invoice-lens` with its actual ID. Authenticate locally; do not paste a token into the app or send it in chat. The command requires internet and access to that account.

Official hosting reference: https://huggingface.co/docs/hub/spaces-sdks-docker

## What is included

- `app.py`: three-page Streamlit interface.
- `core.py`: document reading, parsing, validation and model scoring.
- `models/`: your unchanged trained models and evaluation exports.
- `demo_invoices/`: three PDF invoices, three matching images and separate outcomes.
- `tests/`: model, parsing and application flow checks.
- `Dockerfile`: Python + OCR hosting environment.
- `requirements.txt`: pinned direct dependencies.

No retraining, accounts, live ledger connection, email sending or permanent database is included in this lightweight version.
