"""Disjoint subject roles and named classification metrics; no fitting or I/O."""
import numpy as np

def roles(users,held):
    users=np.asarray(users);assert users.ndim==1 and set(np.unique(users))==set(range(1,7))
    assert held in range(1,7)
    validation=held%6+1
    result=dict(train=np.flatnonzero((users!=held)&(users!=validation)),validation=np.flatnonzero(users==validation),diagnostic=np.flatnonzero(users==held))
    assert len(np.unique(np.concatenate(list(result.values()))))==len(users)
    assert not set(users[result['train']])&{held,validation}
    return validation,result

def classification_metrics(truth,prediction,n_classes=12):
    truth=np.asarray(truth);prediction=np.asarray(prediction)
    assert truth.ndim==prediction.ndim==1 and truth.shape==prediction.shape and len(truth)>0
    assert np.issubdtype(truth.dtype,np.integer) and np.issubdtype(prediction.dtype,np.integer)
    assert set(truth)<=set(range(n_classes)) and set(prediction)<=set(range(n_classes))
    support=np.bincount(truth,minlength=n_classes)
    correct=np.bincount(truth[truth==prediction],minlength=n_classes)
    present=support>0;recall=np.divide(correct,support,out=np.zeros(n_classes,dtype=float),where=present)
    return dict(n=int(len(truth)),accuracy=float(np.mean(truth==prediction)),present_class_macro_recall=float(recall[present].mean()),class_support=support.tolist(),class_recall=[float(recall[k]) if present[k] else None for k in range(n_classes)],missing_classes=np.flatnonzero(~present).tolist(),metric_definition='macro recall averaged over classes with nonzero true support; absent-class recalls null, never zero-filled or silently called twelve-class macro')

def paired_sixfold(rows):
    assert len(rows)==6 and sorted(r['held'] for r in rows)==list(range(1,7))
    rows=sorted(rows,key=lambda r:r['held'])
    for r in rows:assert r['baseline']['n']==r['method']['n'] and r['baseline']['class_support']==r['method']['class_support']
    delta=np.array([r['method']['present_class_macro_recall']-r['baseline']['present_class_macro_recall'] for r in rows])
    result={}
    for arm in ['baseline','method']:
        result[arm]=dict(equal_user_mean_macro_recall=float(np.mean([r[arm]['present_class_macro_recall'] for r in rows])),equal_user_mean_accuracy=float(np.mean([r[arm]['accuracy'] for r in rows])),worst_user_macro_recall=float(min(r[arm]['present_class_macro_recall'] for r in rows)),pooled_sample_accuracy=float(sum(r[arm]['n']*r[arm]['accuracy'] for r in rows)/sum(r[arm]['n'] for r in rows)))
    result.update(paired_macro_recall_deltas=delta.tolist(),mean_paired_macro_recall_delta=float(delta.mean()),strict_user_wins=int((delta>0).sum()),worst_paired_macro_recall_delta=float(delta.min()),independent_validation=False)
    return result

def synthetic_checks():
    # Unequal support makes accuracy and macro recall differ.
    r=classification_metrics(np.array([0,0,0,1]),np.array([0,0,0,0]))
    assert r['accuracy']==.75 and r['present_class_macro_recall']==.5 and r['class_recall'][2] is None
    # Predicting an absent class remains an error for the supported true class.
    r=classification_metrics(np.array([0,0,1,1]),np.array([0,2,1,1]))
    assert r['accuracy']==r['present_class_macro_recall']==.75
    users=np.repeat(np.arange(1,7),3)
    for h in range(1,7):
        v,q=roles(users,h);assert v==h%6+1 and len(q['train'])==12 and len(q['validation'])==len(q['diagnostic'])==3
    pairs=[]
    for h in range(1,7):
        n=1 if h==1 else 3;y=np.zeros(n,dtype=int);pred=np.zeros(n,dtype=int) if h==1 else np.ones(n,dtype=int)
        pairs.append(dict(held=h,baseline=classification_metrics(y,pred),method=classification_metrics(y,np.zeros(n,dtype=int))))
    aggregate=paired_sixfold(pairs)
    assert aggregate['baseline']['equal_user_mean_accuracy']==1/6 and aggregate['baseline']['pooled_sample_accuracy']==1/16
    assert aggregate['strict_user_wins']==5 and aggregate['worst_paired_macro_recall_delta']==0
    return dict(state='pass',synthetic_role_disjointness_all6=True,unbalanced_accuracy_vs_macro_distinguished=True,missing_class_recall_null=True,absent_class_prediction_counted_wrong=True,equal_user_mean_vs_sample_weighted_pool_distinguished=True,paired_subject_count_and_class_support_equal=True,new_fit_started=False)
