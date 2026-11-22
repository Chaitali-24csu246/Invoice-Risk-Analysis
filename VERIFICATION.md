# Verification — 22 November 2026

Passed locally on Python 3.12.14 using requirements.txt:

- Refit all eight pipelines using the revised notebook's CV-selected configurations and original chronological training rows.
- Reproduced all five test average-precision scores and all three regression test MAEs to the notebook's four-decimal rounding tolerance.
- Reloaded the exported bundle and matched all 500 test predictions for each of the eight models against saved evaluation predictions (classification scores within 1e-7, regression within 1e-10; class decisions identical).
- 16 automated tests passed. Coverage includes all 15 model combinations across three demo records, default predictions, unknown models/customers, invalid amounts/files, conservative ambiguous total/date parsing, safe CSV export, text PDF parsing and real PNG OCR.
- Streamlit AppTest exercised demo analysis, model switching, selected model names in history, comparison tables for validation and test, review confirmation and manual-input validation.
- Streamlit started successfully; the HTTP health endpoint and main HTML returned status 200.

The eight fitted models are included; the app does not train during use. Tests can be rerun with `python -m unittest discover -s tests -v`.

Limitations of verification: no browser-based visual review, no Docker build, no Windows/macOS launcher execution, and no cloud deployment were performed. AppTest verifies Streamlit behaviour but is not a browser-rendering test. The GitHub remote has not been changed. The old app's document-parsing implementation and visual styling were retained.
