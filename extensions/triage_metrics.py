"""Clinical case metrics require independent outcomes and case-risk scores.

This module never converts segmentation masks, image corruption levels or pixel
confusion counts into patient severity labels. The executable refuses to run
without separate outcome and score files and an explicit endpoint definition.
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def diagnostic_metrics(labels: np.ndarray, risk: np.ndarray, threshold: float) -> dict:
    raw_y=np.asarray(labels,dtype=float);p=np.asarray(risk,dtype=float)
    if not np.isfinite(raw_y).all() or not set(np.unique(raw_y)) <= {0,1}:
        raise ValueError("Independent outcomes must be exactly 0 or 1")
    y=raw_y.astype(int)
    if y.ndim!=1 or p.shape!=y.shape or not len(y):
        raise ValueError('Expected matching nonempty case-level arrays')
    if not set(np.unique(y)) <= {0,1} or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Outcomes must be binary and risk must be finite in [0,1]')
    if not 0<threshold<1:
        raise ValueError('Threshold must be in (0,1)')
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    def ratio(a: int,b: int) -> float:
        return float(a/b)if b else float('nan')
    return dict(sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),
        ppv=ratio(tp,tp+fp),npv=ratio(tn,tn+fn),auroc=float(roc_auc_score(y,p))if len(np.unique(y))==2 else float('nan'),
        tp=int(tp),tn=int(tn),fp=int(fp),fn=int(fn),cases=len(y))


def decision_curve(labels: np.ndarray,risk: np.ndarray,thresholds: np.ndarray) -> pd.DataFrame:
    y=np.asarray(labels);p=np.asarray(risk,dtype=float)
    rows=[]
    for threshold in thresholds:
        m=diagnostic_metrics(y,p,float(threshold));odds=threshold/(1-threshold)
        rows.append(dict(threshold=float(threshold),net_benefit=(m['tp']-m['fp']*odds)/len(y),
            refer_all=float(y.mean()-(1-y.mean())*odds),refer_none=0.0))
    return pd.DataFrame(rows)


def evaluate(outcomes: Path,scores: Path,output: Path,threshold: float,endpoint: str,provenance: str) -> None:
    y=pd.read_csv(outcomes,dtype={'patient_id':str});p=pd.read_csv(scores,dtype={'patient_id':str})
    if not {'patient_id','actionable'}.issubset(y)or not {'patient_id','model','risk'}.issubset(p):
        raise ValueError('Outcomes: patient_id,actionable; scores: patient_id,model,risk')
    if y.empty or p.empty:
        raise ValueError('Nonempty independent outcomes and risk scores are required')
    if not p.model.map(lambda m: isinstance(m,str)and re.fullmatch(r'[A-Za-z0-9_.-]+',m)is not None and m not in {'.','..'}).all():
        raise ValueError('Unsafe model identifier')
    if y.patient_id.duplicated().any()or p.duplicated(['patient_id','model']).any():
        raise ValueError('One independent outcome and one score per model/patient required')
    if not endpoint.strip()or not provenance.strip():
        raise ValueError('Independent clinical endpoint and label provenance are required')
    if not set(y.actionable.unique())<={0,1}:
        raise ValueError('Independent actionable labels must be 0/1')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for model,group in p.groupby('model'):
        if set(group.patient_id)!=set(y.patient_id):
            raise ValueError('Every model must score the same independently labeled patients')
        data=y.merge(group,on='patient_id',validate='one_to_one')
        rows.append(dict(model=model,threshold=threshold,**diagnostic_metrics(data.actionable.to_numpy(),data.risk.to_numpy(),threshold)))
        decision_curve(data.actionable.to_numpy(),data.risk.to_numpy(),np.linspace(.01,.99,99)).assign(model=model).to_csv(output/f'{model}_decision_curve.csv',index=False)
        data[(data.actionable==1)&(data.risk<threshold)].to_csv(output/f'{model}_false_negative_cases.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'clinical_case_metrics.csv',index=False)
    (output/'endpoint_provenance.json').write_text(json.dumps({'endpoint':endpoint,'label_provenance':provenance,
        'threshold':threshold,'caution':'Net benefit requires clinically justified thresholds and suitable calibrated case-risk scores; pixel probabilities do not supply clinical case risk.'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outcomes',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--label-provenance',required=True)
    a=p.parse_args();evaluate(a.outcomes,a.scores,a.output,a.threshold,a.endpoint,a.label_provenance)
