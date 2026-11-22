"""Refit the notebook's CV-selected configurations, export pipelines and comparisons.
This does not repeat GridSearchCV: the submitted notebook records that search.
Run from any directory: python scripts/build_models.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,accuracy_score,precision_recall_curve,mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
CATS=['customerID','countryCode','PaperlessBill']; NUMS=['InvoiceAmount','payment_terms_days']; FEATURES=CATS+NUMS

def pipeline(model):
    return Pipeline([('preprocess',ColumnTransformer([
      ('categories',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATS),
      ('numbers',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMS)])),
      ('select',VarianceThreshold(0)),('model',model)])

def main():
    df=pd.read_csv(ROOT/'data/WA_Fn-UseC_-Accounts-Receivable.csv')
    for c in ['InvoiceDate','DueDate','SettledDate']:df[c]=pd.to_datetime(df[c],format='%m/%d/%Y')
    for c in CATS:df[c]=df[c].astype(str)
    df=df.sort_values('InvoiceDate',kind='stable')
    df['payment_terms_days']=(df.DueDate-df.InvoiceDate).dt.days
    df['actual_days_overdue']=(df.SettledDate-df.DueDate).dt.days.clip(lower=0)
    df['actual_late']=(df.actual_days_overdue>0).astype(int)
    vs=pd.Timestamp('2013-02-26');ts=pd.Timestamp('2013-07-12')
    train=df[(df.InvoiceDate<vs)&(df.SettledDate<vs)]
    val=df[(df.InvoiceDate>=vs)&(df.InvoiceDate<ts)&(df.SettledDate<ts)]
    test=df[df.InvoiceDate>=ts]
    assert [len(train),len(val),len(test)]==[1390,409,500]
    classifiers={
      'Logistic Regression':LogisticRegression(C=10,max_iter=3000,random_state=42),
      'Decision Tree':DecisionTreeClassifier(max_depth=6,min_samples_leaf=4,random_state=42),
      'Random Forest':RandomForestClassifier(n_estimators=100,min_samples_leaf=2,n_jobs=1,random_state=42),
      'Gradient Boosting':GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=2,random_state=42),
      'XGBoost':XGBClassifier(n_estimators=200,learning_rate=.1,max_depth=3,subsample=.8,colsample_bytree=.8,eval_metric='logloss',n_jobs=1,random_state=42),
      'Dummy baseline':DummyClassifier(strategy='prior')}
    regressors={'Ridge':Ridge(alpha=.1),
      'Random Forest Regressor':RandomForestRegressor(n_estimators=100,min_samples_leaf=2,n_jobs=1,random_state=42),
      'Gradient Boosting Regressor':GradientBoostingRegressor(n_estimators=200,learning_rate=.1,max_depth=3,random_state=42),
      'Median baseline':DummyRegressor(strategy='median')}
    params={k:v.get_params() for k,v in {**classifiers,**regressors}.items()}
    classifiers={k:pipeline(v).fit(train[FEATURES],train.actual_late) for k,v in classifiers.items()}
    regressors={k:pipeline(v).fit(train[FEATURES],train.actual_days_overdue) for k,v in regressors.items()}
    thresholds={}
    for name,m in classifiers.items():
        p=m.predict_proba(val[FEATURES])[:,1];pr,re,th=precision_recall_curve(val.actual_late,p)
        f=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-15)
        thresholds[name]=float(th[np.argmax(f)]) if name!='Dummy baseline' else .5
    out=ROOT/'models';out.mkdir(exist_ok=True)
    tables={}
    for split,frame in [('validation',val),('test',test)]:
        cr=[];rr=[];cp=[];rp=[]
        for name,m in classifiers.items():
            p=m.predict_proba(frame[FEATURES])[:,1];z=p>=thresholds[name];y=frame.actual_late
            cr.append(dict(Model=name,Threshold=thresholds[name],AP=average_precision_score(y,p),ROC_AUC=roc_auc_score(y,p),Precision=precision_score(y,z,zero_division=0),Recall=recall_score(y,z,zero_division=0),F1=f1_score(y,z,zero_division=0),Accuracy=accuracy_score(y,z)))
            cp.append(pd.DataFrame({'invoiceNumber':frame.invoiceNumber,'Model':name,'actual_late':y,'late_score':p,'predicted_late':z.astype(int)}))
        for name,m in regressors.items():
            p=np.maximum(0,m.predict(frame[FEATURES]));y=frame.actual_days_overdue;late=y>0
            rr.append(dict(Model=name,MAE=mean_absolute_error(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),R2=r2_score(y,p),Late_only_MAE=mean_absolute_error(y[late],p[late])))
            rp.append(pd.DataFrame({'invoiceNumber':frame.invoiceNumber,'Model':name,'actual_days_overdue':y,'predicted_days_overdue':p}))
        tables[split]=(pd.DataFrame(cr),pd.DataFrame(rr))
        tables[split][0].to_csv(out/f'{split}_classification.csv',index=False)
        tables[split][1].to_csv(out/f'{split}_regression.csv',index=False)
        pd.concat(cp).to_csv(out/f'{split}_classification_predictions.csv',index=False)
        pd.concat(rp).to_csv(out/f'{split}_regression_predictions.csv',index=False)
    default_c=tables['validation'][0].sort_values('AP',ascending=False).iloc[0].Model
    default_r=tables['validation'][1].sort_values('MAE').iloc[0].Model
    assert default_c=='XGBoost' and default_r=='Random Forest Regressor'
    # Verify against the executed academic notebook (rounded to four decimal places).
    expected_c={'Logistic Regression':.7052,'Decision Tree':.4180,'Random Forest':.6982,'Gradient Boosting':.6502,'XGBoost':.7176}
    expected_r={'Ridge':2.2875,'Random Forest Regressor':2.3527,'Gradient Boosting Regressor':2.6552}
    for _,r in tables['test'][0].iterrows():
        if r.Model in expected_c:assert abs(r.AP-expected_c[r.Model])<.000051,(r.Model,r.AP)
    for _,r in tables['test'][1].iterrows():
        if r.Model in expected_r:assert abs(r.MAE-expected_r[r.Model])<.000051,(r.Model,r.MAE)
    bundle=dict(classifiers={k:v for k,v in classifiers.items() if k!='Dummy baseline'},regressors={k:v for k,v in regressors.items() if k!='Median baseline'},thresholds=thresholds,default_classifier=default_c,default_regressor=default_r,features=FEATURES,categorical_features=CATS,regression_clip_min=0,classifier=classifiers[default_c],regressor=regressors[default_r],classification_threshold=thresholds[default_c])
    joblib.dump(bundle,out/'invoice_models.joblib',compress=3)
    pd.DataFrame({'invoiceNumber':test.invoiceNumber,'actual_late':test.actual_late,'late_score':classifiers[default_c].predict_proba(test[FEATURES])[:,1].astype(float),'predicted_late':(classifiers[default_c].predict_proba(test[FEATURES])[:,1]>=thresholds[default_c]).astype(int),'actual_days_overdue':test.actual_days_overdue,'predicted_days_overdue':np.maximum(0,regressors[default_r].predict(test[FEATURES]))}).to_csv(out/'test_predictions.csv',index=False)
    import sklearn,scipy,xgboost
    metadata=dict(classifier=default_c,regressor=default_r,parameters=params,rows={'train':len(train),'validation':len(val),'test':len(test),'excluded':len(df)-len(train)-len(val)-len(test)},versions={'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'scipy':scipy.__version__,'xgboost':xgboost.__version__},selection='Hyperparameters from notebook time-aware training CV; classifier chosen by validation AP; regressor by validation MAE; thresholds maximise validation F1.',test_note='The test period was inspected in earlier project iterations. These are held-out chronological results, not a fresh external evaluation.',encoded_features=109,retained_features=108)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    print('Exported 8 fitted pipelines. Notebook test AP and MAE reproduced. Defaults:',default_c,default_r)
    print(metadata['versions'])
if __name__=='__main__':main()
