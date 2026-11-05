import io
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, load_assets, score_invoice, extract_document, parse_fields, export_csv
from streamlit.testing.v1 import AppTest


class ModelAndExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.metadata, cls.categories = load_assets()
        cls.demos = json.loads((ROOT/'demo_invoices/examples.json').read_text())
        cls.saved = pd.read_csv(ROOT/'models/test_predictions.csv', dtype={'invoiceNumber':str})

    def test_demo_predictions_match_notebook(self):
        for demo in self.demos.values():
            expected = self.saved[self.saved.invoiceNumber == demo['invoiceNumber']].iloc[0]
            actual = score_invoice(demo['features'], self.bundle, self.categories)
            self.assertAlmostEqual(actual['late_score'], expected.late_score, places=10)
            self.assertAlmostEqual(actual['predicted_days_overdue'], expected.predicted_days_overdue, places=10)
            self.assertEqual(actual['predicted_late'], expected.predicted_late)

    def test_text_pdf_fields(self):
        for demo in self.demos.values():
            name=demo['filename']
            result=extract_document((ROOT/'demo_invoices'/name).read_bytes(), name)
            for key,value in demo['features'].items():
                self.assertEqual(result['fields'].get(key), value)
            self.assertTrue(result['preview'])

    def test_png_ocr(self):
        demo=list(self.demos.values())[1]
        name=demo['filename'].replace('.pdf','.png')
        result=extract_document((ROOT/'demo_invoices'/name).read_bytes(), name)
        self.assertEqual(result['fields'].get('InvoiceAmount'), demo['features']['InvoiceAmount'])
        self.assertEqual(result['fields'].get('customerID'), demo['features']['customerID'])

    def test_unknown_customer_blocked(self):
        fields=dict(next(iter(self.demos.values()))['features'],customerID='UNKNOWN')
        with self.assertRaises(ValueError):score_invoice(fields,self.bundle,self.categories)

    def test_invalid_amounts_blocked(self):
        for amount in [None, -1, 0, float('inf'), float('nan'), 'oops']:
            with self.subTest(amount=amount):
                fields=dict(next(iter(self.demos.values()))['features'],InvoiceAmount=amount)
                with self.assertRaises(ValueError):score_invoice(fields,self.bundle,self.categories)

    def test_ambiguous_total_not_guessed(self):
        fields,warnings=parse_fields('Total: 120.00\nAmount due: 100.00')
        self.assertNotIn('InvoiceAmount',fields)
        self.assertTrue(warnings)

    def test_ambiguous_date_not_guessed(self):
        fields,warnings=parse_fields('Invoice date: 03/04/2013')
        self.assertNotIn('InvoiceDate',fields)
        self.assertTrue(warnings)

    def test_invalid_file_rejected(self):
        with self.assertRaises(ValueError):extract_document(b'test','file.txt')
        with self.assertRaises(ValueError):extract_document(b'','file.pdf')

    def test_csv_formulas_escaped(self):
        result=export_csv([{'invoiceNumber':'=1+1'}]).decode('utf-8-sig')
        self.assertIn("'=1+1",result)


class InterfaceTests(unittest.TestCase):
    def test_demo_prediction_history_and_performance(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertEqual(len(app.exception),0)
        app.checkbox[0].check().run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.metric),3)
        self.assertTrue(any('Risk' in m.label or 'risk' in m.label for m in app.metric))
        save=next(b for b in app.button if b.label=='Save to session history')
        save.click().run()
        self.assertEqual(len(app.session_state['history']),1)
        app.sidebar.radio[0].set_value('Session history').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.dataframe),1)
        self.assertNotIn('_identity',app.dataframe[0].value.columns)
        app.sidebar.radio[0].set_value('Model performance').run()
        self.assertEqual(len(app.exception),0)
        self.assertGreater(len(app.metric),4)

    def test_unconfirmed_invoice_blocked(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        app.button[0].click().run()
        self.assertTrue(any('confirm' in e.value for e in app.error))
        self.assertEqual(len(app.metric),0)

    def test_manual_missing_fields_blocked(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        app.radio[0].set_value('Enter manually').run()
        app.checkbox[0].check().run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertGreater(len(app.error),0)

if __name__=='__main__':unittest.main()
