"""Recorded evaluation, kept separate from live invoice predictions."""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_recall_curve
from core import ROOT

DESCRIPTIONS={
 'Logistic Regression':'A weighted combination of encoded inputs, converted to a score by the sigmoid function. A simple linear baseline with regularisation.',
 'Decision Tree':'A sequence of feature-based rules. Easy to follow, but individual trees can overfit.',
 'Random Forest':'Averages many trees trained on varied samples and features. Captures non-linear relationships.',
 'Gradient Boosting':'Builds trees sequentially, with later trees correcting earlier errors.',
 'XGBoost':'Regularised gradient-boosted trees with row and feature sampling. Selected for the highest validation average precision.',
 'Ridge':'A linear regression with a penalty on large coefficients. Simple and stable.',
 'Random Forest Regressor':'Averages predictions from many regression trees. Selected for the lowest validation MAE.',
 'Gradient Boosting Regressor':'Sequential regression trees that correct residual errors.',
 'Dummy baseline':'Always uses the training late-payment prevalence as its score; a reference, not an app model.',
 'Median baseline':'Always predicts the training median delay (zero days here); a reference, not an app model.'}

def render_comparison(bundle,metadata):
    st.title('Compare the models')
    st.write('One dataset, the same chronological split, and saved results for every model.')
    a,b=st.columns(2)
    a.info('**Recommended classifier**\n\n'+bundle['default_classifier']+' · selected by validation average precision')
    b.info('**Recommended delay model**\n\n'+bundle['default_regressor']+' · selected by validation MAE')
    a,b,c=st.columns(3)
    a.metric('Training invoices','1,390');b.metric('Validation invoices','409');c.metric('Test invoices','500')
    split=st.radio('Evaluation dataset',['Test','Validation'],horizontal=True).lower()
    st.caption('Validation results choose models and cutoffs. Test results describe later invoices; all test customers were seen in training. '+metadata['test_note'])
    classification,regression,method=st.tabs(['Classification','Regression','How it works'])
    with classification:
        df=pd.read_csv(ROOT/f'models/{split}_classification.csv')
        selected=df[df.Model==bundle['default_classifier']].iloc[0]
        a,b,c=st.columns(3)
        a.metric('Recommended model · average precision',f'{selected.AP:.3f}')
        b.metric('Recall',f'{selected.Recall:.1%}')
        c.metric('F1',f'{selected.F1:.3f}')
        shown=df.copy();shown.insert(1,'Selected on validation',shown.Model==bundle['default_classifier'])
        st.dataframe(shown.style.format({c:'{:.4f}' for c in df.columns if c!='Model'}).apply(lambda row:['background-color: #e2f3ed' if row['Selected on validation'] else '' for _ in row],axis=1),hide_index=True,width='stretch')
        metric=st.selectbox('Classification metric',['AP','ROC_AUC','Precision','Recall','F1','Accuracy'])
        fig=px.bar(df.sort_values(metric),x=metric,y='Model',orientation='h',color=df.sort_values(metric).Model.eq(bundle['default_classifier']).map({True:'Recommended',False:'Other'}),color_discrete_map={'Recommended':'#168274','Other':'#9aaeba'},range_x=[0,1],title=f'{split.title()} classification — {metric}')
        fig.update_layout(showlegend=False,yaxis_title=None)
        st.plotly_chart(fig,width='stretch')
        model=st.selectbox('Inspect classifier',list(bundle['classifiers']),index=list(bundle['classifiers']).index(bundle['default_classifier']))
        st.write(DESCRIPTIONS[model])
        predictions=pd.read_csv(ROOT/f'models/{split}_classification_predictions.csv')
        p=predictions[predictions.Model==model]
        cm=pd.crosstab(p.actual_late,p.predicted_late).reindex(index=[0,1],columns=[0,1],fill_value=0)
        cm.index=['Actually on time','Actually late'];cm.columns=['Predicted on time','Predicted late']
        st.write('**Confusion matrix — invoice counts**');st.dataframe(cm,width='stretch')
        precision,recall,_=precision_recall_curve(p.actual_late,p.late_score)
        st.plotly_chart(px.line(x=recall,y=precision,labels={'x':'Recall','y':'Precision'},title=f'{model}: precision–recall curve',range_x=[0,1],range_y=[0,1]),width='stretch')
        with st.expander('Metrics explained'):
            st.write('Average precision (AP): summarises precision–recall performance across cutoffs; higher is better. ROC-AUC: ranking of late versus on-time invoices; 0.5 is chance. Precision: how many alerts were actually late. Recall: how many late invoices were caught. F1: balances precision and recall. Accuracy: fraction of all predictions correct; can hide missed late invoices. Threshold: the model-specific cutoff selected to maximise validation F1.')
        with st.expander('Fitted model parameters'):st.json(metadata['parameters'][model])
        st.download_button('Download classification comparison',df.to_csv(index=False),f'{split}_classification.csv','text/csv')
    with regression:
        df=pd.read_csv(ROOT/f'models/{split}_regression.csv')
        selected=df[df.Model==bundle['default_regressor']].iloc[0]
        a,b,c=st.columns(3)
        a.metric('Recommended model · MAE',f'{selected.MAE:.2f} days');b.metric('RMSE',f'{selected.RMSE:.2f} days');c.metric('Late-only MAE',f'{selected.Late_only_MAE:.2f} days')
        st.warning('Random Forest won on validation MAE by a small margin. On the test set, Ridge is better than Random Forest, and the zero-day median baseline has the lowest overall MAE. The baseline still misses the delays of late invoices. This is a limitation, not a reason to change the winner using test results.')
        shown=df.copy();shown.insert(1,'Selected on validation',shown.Model==bundle['default_regressor'])
        st.dataframe(shown.style.format({c:'{:.4f}' for c in df.columns if c!='Model'}),hide_index=True,width='stretch')
        metric=st.selectbox('Regression metric',['MAE','RMSE','Late_only_MAE','R2'])
        st.caption('Lower is better for errors measured in days. Higher is better for R²; negative R² is possible.')
        fig=px.bar(df,x='Model',y=metric,title=f'{split.title()} regression — {metric}',color_discrete_sequence=['#168274'])
        st.plotly_chart(fig,width='stretch')
        model=st.selectbox('Inspect regressor',list(bundle['regressors']),index=list(bundle['regressors']).index(bundle['default_regressor']))
        st.write(DESCRIPTIONS[model])
        predictions=pd.read_csv(ROOT/f'models/{split}_regression_predictions.csv');p=predictions[predictions.Model==model]
        fig=px.scatter(p,x='actual_days_overdue',y='predicted_days_overdue',opacity=.5,labels={'actual_days_overdue':'Actual days overdue','predicted_days_overdue':'Predicted days overdue'},title=f'{model}: actual versus predicted delay')
        upper=max(p.actual_days_overdue.max(),p.predicted_days_overdue.max())
        fig.add_shape(type='line',x0=0,y0=0,x1=upper,y1=upper,line=dict(color='#87949e',dash='dash'))
        st.plotly_chart(fig,width='stretch')
        with st.expander('Metrics explained'):
            st.write('MAE: average absolute prediction error in days. RMSE: gives larger errors more weight. R²: improvement in squared error over predicting the evaluation set mean; 1 is perfect and values below zero are possible. Late-only MAE: error restricted to invoices that actually paid late. These are average errors, not prediction intervals. Negative delay predictions are clipped to zero for both evaluation and the app.')
        with st.expander('Fitted regressor parameters'):st.json(metadata['parameters'][model])
        st.download_button('Download regression comparison',df.to_csv(index=False),f'{split}_regression.csv','text/csv')
    with method:
        st.subheader('From document to prediction')
        st.write('Upload or choose a demo → read text/OCR → confirm invoice fields → encode and scale with saved training transformations → remove constant features → run the two selected models → show risk and estimated delay.')
        st.write('The classifier returns its class-1 score through predict_proba. The score is compared with that classifier’s validation-tuned cutoff. It has not been calibrated as a real-world payment probability. The regressor separately predicts days overdue, clipped at zero; it is not multiplied by the risk score.')
        st.subheader('Training and model selection')
        st.write(metadata['selection'])
        st.write('Training precedes 26 February 2013; validation runs from that date to before 12 July 2013; test starts on 12 July 2013. Training and validation also require settlement before the next period begins. This excludes 167 outcomes not yet known at the relevant cutoff.')
        st.write('Inputs: customer ID, country code, billing method, invoice amount and derived payment terms. One-hot encoding and standardisation are fitted only on training data. VarianceThreshold removes the constant 30-day payment-terms feature: 109 encoded features become 108. Customer ID captures patterns learned during training; no live payment history is fetched.')
        st.write('The notebook compares weighting and sampling strategies. The saved configurations use the original training distribution; Logistic Regression and Random Forest also searched class weights. Imbalance is addressed through suitable metrics and validation-tuned thresholds, not by altering validation/test prevalence. Class imbalance alone does not establish overfitting.')
        st.subheader('Scope and limitations')
        st.write('This predicts late payment, not permanent non-payment or fraud. The synthetic/sample company dataset contains 2,466 invoices from 2012–2013. Unknown customers are blocked. Other companies, currencies, amount ranges and payment terms are not validated. OCR can make mistakes, so human review is required. The test period has been inspected during earlier iterations; an independent future dataset is needed for stronger claims.')
        st.caption('Baselines are evaluation references, not selectable production models. All five trained classifiers and all three trained regressors are available on the analysis page.')
