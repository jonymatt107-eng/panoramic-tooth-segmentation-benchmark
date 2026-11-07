"""Hand-computed diagnostic arithmetic and boundary validity tests."""
import numpy as np
import pytest
from triage_metrics import diagnostic_metrics,decision_curve
from failure_audit import boundary_f1


def test_case_diagnostic_formulas() -> None:
    y=np.array([1,1,0,0]);p=np.array([.9,.2,.8,.1])
    m=diagnostic_metrics(y,p,.5)
    assert m['sensitivity']==m['specificity']==m['ppv']==m['npv']==.5
    assert m['auroc']==.75
    d=decision_curve(y,p,np.array([.5])).iloc[0]
    assert d.net_benefit==0 and d.refer_all==0 and d.refer_none==0


def test_undefined_metrics_stay_undefined() -> None:
    m=diagnostic_metrics(np.array([0,0]),np.array([.1,.2]),.5)
    assert np.isnan(m['sensitivity'])and np.isnan(m['ppv'])and np.isnan(m['auroc'])
    assert m['npv']==1 and m['specificity']==1


def test_invalid_probabilities_rejected() -> None:
    with pytest.raises(ValueError):diagnostic_metrics(np.array([1,0]),np.array([1.2,.3]),.5)


def test_boundary_f1_validity_and_tolerance() -> None:
    target=np.zeros((32,32),bool);target[8:24,8:24]=True
    valid=np.ones_like(target)
    assert boundary_f1(target,target,valid)==1
    shifted=np.roll(target,1,axis=1)
    assert boundary_f1(shifted,target,valid,tolerance=2)==1
    assert boundary_f1(np.zeros_like(target),target,valid)==0
    padded=np.pad(target,4);v=np.pad(valid,4)
    assert boundary_f1(padded,padded,v)==1


def test_fractional_case_labels_rejected() -> None:
    with pytest.raises(ValueError):diagnostic_metrics(np.array([.2,1]),np.array([.1,.8]),.5)
