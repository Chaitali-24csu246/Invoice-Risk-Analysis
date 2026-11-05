"""Prediction and document parsing. No retraining or outcome lookup during scoring."""
from pathlib import Path
from datetime import datetime
import io
import json
import math
import re

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pymupdf

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 5


def load_assets():
    # Load only the project's trusted local model; never accept uploaded pickle files.
    bundle = joblib.load(ROOT / 'models/invoice_models.joblib')
    metadata = json.loads((ROOT / 'models/metadata.json').read_text())
    encoder = bundle['classifier'].named_steps['preprocess'].named_transformers_['categories']
    categories = dict(zip(bundle['categorical_features'], encoder.categories_))
    return bundle, metadata, {k: list(map(str, v)) for k, v in categories.items()}


def score_invoice(fields, bundle, categories):
    clean = {}
    for key in bundle['categorical_features']:
        value = str(fields.get(key, '')).strip()
        if value not in categories[key]:
            labels = {'customerID': 'customer', 'countryCode': 'country code', 'PaperlessBill': 'billing method'}
            raise ValueError(f"Choose a known {labels[key]}. This model does not support unknown values.")
        clean[key] = value
    try:
        amount = float(fields.get('InvoiceAmount'))
    except (TypeError, ValueError):
        raise ValueError('Enter a valid invoice amount.') from None
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError('Invoice amount must be a finite number greater than zero.')
    clean['InvoiceAmount'] = amount
    issued, due = fields.get('InvoiceDate'), fields.get('DueDate')
    if issued and due and due < issued:
        raise ValueError('Due date cannot be earlier than the invoice date.')
    x = pd.DataFrame([clean], columns=bundle['features'])
    classes = list(bundle['classifier'].classes_)
    score = float(bundle['classifier'].predict_proba(x)[0, classes.index(1)])
    days = max(float(bundle['regressor'].predict(x)[0]), bundle.get('regression_clip_min', 0))
    return {
        'late_score': score,
        'predicted_late': int(score >= bundle['classification_threshold']),
        'predicted_days_overdue': days,
        'threshold': float(bundle['classification_threshold']),
    }


def ocr_image(image):
    import pytesseract
    try:
        return pytesseract.image_to_string(ImageOps.exif_transpose(image).convert('RGB'), config='--psm 6', timeout=25)
    except pytesseract.TesseractNotFoundError:
        raise ValueError('Image scanning needs Tesseract. Use a text PDF, a demo invoice, or manual entry. See START_HERE.md for installation.') from None
    except RuntimeError:
        raise ValueError('Scanning took too long. Try a clearer, smaller image or enter details manually.') from None


def extract_document(content, filename):
    if not content or len(content) > MAX_BYTES:
        raise ValueError('Upload a non-empty file smaller than 10 MB.')
    suffix = Path(filename).suffix.lower()
    if suffix == '.pdf':
        with pymupdf.open(stream=content, filetype='pdf') as doc:
            if doc.needs_pass:
                raise ValueError('Please use an unencrypted PDF.')
            if not 1 <= len(doc) <= MAX_PAGES:
                raise ValueError('Use one invoice with no more than five pages.')
            texts = []
            preview = None
            for page in doc:
                # Bound rendered image dimensions for unusually large pages.
                scale = min(2, 2400 / max(page.rect.width, page.rect.height))
                text = page.get_text('text').strip()
                if len(text) < 40 or preview is None:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                    if preview is None:
                        preview = pix.tobytes('png')
                    if len(text) < 40:
                        text = ocr_image(Image.open(io.BytesIO(pix.tobytes('png'))))
                texts.append(text)
            text = '\n'.join(texts)
    elif suffix in {'.png', '.jpg', '.jpeg'}:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError('Image is too large. Use an image below 20 megapixels.')
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((2400, 2400))
            text = ocr_image(image)
            output = io.BytesIO()
            image.save(output, format='PNG')
            preview = output.getvalue()
    else:
        raise ValueError('Supported files: PDF, PNG and JPG.')
    if not text.strip():
        raise ValueError('No readable text found. Try a clearer document or manual entry.')
    fields, warnings = parse_fields(text)
    return {'text': text, 'fields': fields, 'warnings': warnings, 'preview': preview}


def parse_fields(text):
    fields, warnings = {}, []
    patterns = {
        'invoiceNumber': r'(?:invoice\s*(?:number|no\.?|#))\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9/-]*)',
        'customerID': r'(?:customer\s*(?:id|code))\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9-]*)',
        'countryCode': r'country\s*code\s*[:#-]?\s*(\d+)',
        'PaperlessBill': r'(?:billing\s*(?:method|type)|paperless\s*bill)\s*[:#-]?\s*(Electronic|Paper)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            fields[key] = match.group(1).strip()
    if 'PaperlessBill' in fields:
        fields['PaperlessBill'] = fields['PaperlessBill'].capitalize()
    # Multiple different totals must be resolved by the reviewer, never guessed.
    candidates = re.findall(r'^\s*(?:invoice\s+amount|grand\s+total|total\s+due|amount\s+due|total)\s*:?\s*(?:[A-Z]{3}|[$£€₹])?\s*([\d][\d,]*(?:\.\d{1,2})?)\s*(?:[A-Z]{3})?\s*$', text, re.I | re.M)
    totals = {float(c.replace(',', '')) for c in candidates}
    if len(totals) == 1:
        fields['InvoiceAmount'] = totals.pop()
    elif len(totals) > 1:
        warnings.append('Multiple different totals detected. Enter the correct invoice amount.')
    for key, label in [('InvoiceDate', 'invoice'), ('DueDate', 'due')]:
        match = re.search(rf'{label}\s*date\s*:?\s*(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})', text, re.I)
        if match:
            value = match.group(1)
            try:
                if '-' in value:
                    fields[key] = datetime.strptime(value, '%Y-%m-%d').date()
                else:
                    a, b, year = map(int, value.split('/'))
                    if a <= 12 and b <= 12 and a != b:
                        warnings.append(f'Ambiguous {label} date ({value}); select it manually.')
                    else:
                        fmt = '%d/%m/%Y' if a > 12 else '%m/%d/%Y'
                        fields[key] = datetime.strptime(value, fmt).date()
            except ValueError:
                warnings.append(f'Could not interpret {label} date; select it manually.')
    return fields, warnings


def export_csv(rows):
    frame = pd.DataFrame(rows)
    # Protect exported spreadsheet cells from user-provided formulas.
    for column in frame.select_dtypes(include=['object']).columns:
        frame[column] = frame[column].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
    return frame.to_csv(index=False).encode('utf-8-sig')
