from datetime import datetime
from hashlib import sha256
import json

import pandas as pd
import streamlit as st

from core import ROOT, load_assets, extract_document, score_invoice, export_csv

st.set_page_config(page_title='Invoice Lens', page_icon='📄', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {max-width:1180px;padding-top:2.2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#eef3f6;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stMetric"] {background:#f3f6f8;border:1px solid #e1e8ed;border-radius:12px;padding:16px;}
[data-testid="stMetricLabel"] {color:#536574;}
.stButton>button[kind="primary"] {border-radius:8px;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def assets():
    return load_assets()

try:
    bundle, metadata, categories = assets()
except Exception:
    st.error('The saved model could not be loaded. Install requirements.txt in a fresh Python 3.12 environment and keep the models folder beside app.py.')
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title('Invoice Lens')
    st.caption('PAYMENT RISK CHECKER')
    page = st.radio('Workspace', ['Analyse invoice', 'Session history', 'Compare models'], label_visibility='collapsed')
    st.divider()
    st.caption('Company-specific academic prototype. Synthetic/sample invoice data.')
    st.caption('5 classifiers · 3 regressors. Saved models; no retraining during analysis.')
    st.caption('Uploads and results stay in this browser session on the app server. History resets when the session ends; download it to keep a copy.')

if page == 'Session history':
    st.title('Your invoice reviews')
    st.write('Compare saved predictions from this session.')
    history = st.session_state.history
    if not history:
        st.info('Analyse an invoice, then select Save to session history.')
    else:
        a,b,c = st.columns(3)
        a.metric('Invoices saved', len(history))
        b.metric('Flagged likely late', sum(r['predicted_late'] for r in history))
        c.metric('Average estimated delay', f"{pd.DataFrame(history)['predicted_days_overdue'].mean():.1f} days")
        show = st.selectbox('Show', ['All invoices', 'Likely late', 'Likely on time'])
        rows = [r for r in history if show == 'All invoices' or r['predicted_late'] == int(show == 'Likely late')]
        public_rows = [{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]
        st.dataframe(pd.DataFrame(public_rows), hide_index=True, width='stretch')
        st.download_button('Download results as CSV', export_csv(public_rows), 'invoice_predictions.csv', 'text/csv')
        if st.button('Clear session history'):
            st.session_state.history = []
            st.rerun()
    st.stop()

if page == 'Compare models':
    from comparison import render_comparison
    render_comparison(bundle, metadata)
    st.stop()

st.title('A clearer view of payment risk')
st.write('Review an unpaid invoice and estimate whether it will be paid late.')
mode = st.radio('Start with', ['Demo invoice', 'Upload invoice', 'Enter manually'], horizontal=True)
fields, preview, raw_text, warnings, truth = {}, None, '', [], None
source = 'Manual entry'
source_id = 'manual'

if mode == 'Demo invoice':
    demos = json.loads((ROOT / 'demo_invoices/examples.json').read_text())
    choice = st.selectbox('Choose a prepared invoice', list(demos), help='Documents reconstructed from held-out sample records. Outcomes are hidden until you reveal them.')
    demo = demos[choice]
    source = demo['filename']
    content = (ROOT / 'demo_invoices' / source).read_bytes()
    source_id = sha256(content).hexdigest()[:16]
    with st.spinner('Reading the invoice…'):
        extracted = extract_document(content, source)
    fields, preview, raw_text, warnings = [extracted[k] for k in ['fields','preview','text','warnings']]
    truth = demo
    st.download_button('Download this demo PDF', content, source, 'application/pdf')
    st.caption('Demonstration document reconstructed from a test-set invoice. It is not an original company document.')
elif mode == 'Upload invoice':
    uploaded = st.file_uploader('Upload one unpaid invoice', type=['pdf','png','jpg','jpeg'], help='Maximum 10 MB, five PDF pages. English text works best.')
    if uploaded is None:
        st.info('Upload an invoice, or choose a demo to try the complete workflow.')
        st.stop()
    source = uploaded.name
    content = uploaded.getvalue()
    source_id = sha256(content).hexdigest()[:16]
    # Only session-local extraction caching; uploads are never cached globally.
    cached = st.session_state.get('extracted_document')
    if not cached or cached['id'] != source_id:
        try:
            with st.spinner('Reading the invoice…'):
                parsed = extract_document(content, source)
            st.session_state.extracted_document = {'id': source_id, 'result': parsed}
        except Exception as error:
            st.error(f'Could not read this document: {error}')
            st.info('Choose Enter manually to continue, or upload a clearer supported document.')
            st.stop()
    extracted = st.session_state.extracted_document['result']
    fields, preview, raw_text, warnings = [extracted[k] for k in ['fields','preview','text','warnings']]

left, right = st.columns([1,1.05], gap='large')
with left:
    st.subheader('1. Review the document')
    if preview:
        st.image(preview, width='stretch')
        with st.expander('Extracted text'):
            st.text(raw_text)
    else:
        st.info('Enter the invoice details on the right. No upload is required.')
        st.markdown('**Before you analyse**\n\nUse an unpaid invoice for a known company customer. Confirm the total and billing method from the record.')
    for warning in warnings:
        st.warning(warning)

with right:
    st.subheader('2. Confirm the details')
    st.caption('Check every field. Scanning reads the document; the saved models estimate payment behaviour.')
    if fields.get('customerID') and fields['customerID'] not in categories['customerID']:
        st.warning(f"Detected customer {fields['customerID']} is not supported. Only correct this if extraction was wrong; do not substitute another customer.")
    with st.form('review_' + source_id):
        invoice_id = st.text_input('Invoice number', value=fields.get('invoiceNumber',''), max_chars=80)
        def pick(label, key):
            values = categories[key]
            detected = fields.get(key)
            index = values.index(detected) if detected in values else None
            return st.selectbox(label, values, index=index, placeholder='Select…')
        customer = pick('Customer ID', 'customerID')
        country = pick('Country code', 'countryCode')
        billing = pick('Billing method', 'PaperlessBill')
        amount = st.number_input('Invoice amount (dataset units)', min_value=0.0, value=fields.get('InvoiceAmount'), step=0.01, format='%.2f')
        st.caption('Use the company’s original amount units. The dataset does not establish a currency or currency conversion.')
        c1,c2 = st.columns(2)
        issued = c1.date_input('Invoice date (optional)', value=fields.get('InvoiceDate'), min_value=datetime(1900,1,1), max_value=datetime(2100,12,31))
        due = c2.date_input('Due date (optional)', value=fields.get('DueDate'), min_value=datetime(1900,1,1), max_value=datetime(2100,12,31))
        st.markdown('**Choose your models**')
        classifier_name = st.selectbox('Classification model', list(bundle['classifiers']), index=list(bundle['classifiers']).index(bundle['default_classifier']), format_func=lambda n: n + (' — Recommended' if n == bundle['default_classifier'] else ''))
        regressor_name = st.selectbox('Delay estimation model', list(bundle['regressors']), index=list(bundle['regressors']).index(bundle['default_regressor']), format_func=lambda n: n + (' — Recommended' if n == bundle['default_regressor'] else ''))
        st.caption('Recommended models were selected on validation results. Each classifier uses its own validation-tuned cutoff.')
        checked = st.checkbox('I have checked these details against the invoice.')
        submitted = st.form_submit_button('Analyse payment risk', type='primary', width='stretch')
    if submitted:
        st.session_state.pop('analysis', None)
        if not checked:
            st.error('Please review and confirm the details first.')
        else:
            inputs = {'customerID':customer,'countryCode':country,'PaperlessBill':billing,'InvoiceAmount':amount,'InvoiceDate':issued,'DueDate':due}
            try:
                prediction = score_invoice(inputs, bundle, categories, classifier_name, regressor_name)
                st.session_state.analysis = {'source_id':source_id, 'source':source,'invoiceNumber':invoice_id or 'Not supplied', 'inputs':inputs, 'prediction':prediction}
            except ValueError as error:
                st.error(str(error))

analysis = st.session_state.get('analysis')
if analysis and analysis['source_id'] == source_id:
    st.divider()
    st.subheader('3. Your payment-risk result')
    p = analysis['prediction']
    st.caption(f"Classification: {p['classifier']} · Delay estimate: {p['regressor']}")
    if analysis['inputs'].get('InvoiceDate') and analysis['inputs'].get('DueDate') and (analysis['inputs']['DueDate'] - analysis['inputs']['InvoiceDate']).days != 30:
        st.warning('The training data only contained 30-day payment terms. Predictions for other terms have not been validated; the constant terms feature was removed during training.')
    c1,c2,c3 = st.columns(3)
    c1.metric('Payment prediction', 'Likely late' if p['predicted_late'] else 'Likely on time')
    c2.metric('Late-payment risk score', f"{p['late_score'] * 100:.1f} / 100")
    c3.metric('Estimated days overdue', f"{p['predicted_days_overdue']:.1f} days")
    st.caption(f"Score cutoff: {p['threshold'] * 100:.2f}/100. Scores are not calibrated probabilities. Predictions are estimates, not guarantees.")
    st.write('**Suggested review:** ' + ('Prioritise this invoice for follow-up.' if p['predicted_late'] else 'Keep this invoice in the normal follow-up queue.'))
    st.caption('This follow-up suggestion is a simple rule based on the predicted class. It is not a separate model explanation.')
    with st.expander('Details used for this result'):
        st.json({**{k:str(v) for k,v in analysis['inputs'].items()}, 'invoiceNumber':analysis['invoiceNumber']})
        st.caption('After editing the form, press Analyse again to update the result. Classification and regression are independent; a small positive delay can accompany a likely-on-time prediction.')
    with st.expander('Compare all models on this invoice'):
        class_rows = []
        for name in bundle['classifiers']:
            result = score_invoice(analysis['inputs'], bundle, categories, name, p['regressor'])
            class_rows.append({'Model':name, 'Risk score / 100':round(result['late_score']*100,2), 'Cutoff / 100':round(result['threshold']*100,2), 'Prediction':'Likely late' if result['predicted_late'] else 'Likely on time'})
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, width='stretch')
        reg_rows = [{'Model':name, 'Estimated days overdue':round(score_invoice(analysis['inputs'], bundle, categories, p['classifier'], name)['predicted_days_overdue'],2)} for name in bundle['regressors']]
        st.dataframe(pd.DataFrame(reg_rows), hide_index=True, width='stretch')
        st.caption('These are predictions for the same reviewed inputs, not evaluation metrics. Models may disagree.')
    if st.button('Save to session history', type='primary'):
        entry = {'invoiceNumber':analysis['invoiceNumber'], 'source':source, **{k:str(v) if k in ['InvoiceDate','DueDate'] else v for k,v in analysis['inputs'].items()}, **p}
        identity = json.dumps(entry, sort_keys=True)
        if any(r.get('_identity') == identity for r in st.session_state.history):
            st.info('This exact result is already saved.')
        else:
            entry['reviewed_at_utc'] = datetime.now(__import__('datetime').timezone.utc).isoformat(timespec='seconds')
            entry['_identity'] = identity
            st.session_state.history.append(entry)
            st.success('Saved. Open Session history to compare or download results.')
    if truth:
        # Compare outcomes only when the model inputs still match this demo record.
        same = all(str(analysis['inputs'].get(k)) == str(truth['features'][k]) for k in ['customerID','countryCode','PaperlessBill']) and abs(float(analysis['inputs']['InvoiceAmount']) - truth['features']['InvoiceAmount']) < 1e-8
        if same:
            with st.expander('Reveal the recorded demo outcome'):
                st.write(f"**Actual payment:** {'Late' if truth['actual_late'] else 'On time'}")
                st.write(f"**Actual days overdue:** {truth['actual_days_overdue']}")
                st.write(f"**Absolute delay error:** {abs(p['predicted_days_overdue'] - truth['actual_days_overdue']):.2f} days")
                st.caption('This outcome comes from the held-out test record. It is never sent to either model.')
        else:
            st.info('You changed the demo inputs, so its recorded outcome is not shown as a comparison.')
