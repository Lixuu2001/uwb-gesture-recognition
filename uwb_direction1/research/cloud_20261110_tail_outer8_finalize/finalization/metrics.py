import numpy as np

def classification_metrics(truth,prediction,n_classes=12):
    truth=np.asarray(truth);prediction=np.asarray(prediction)
    assert truth.ndim==prediction.ndim==1 and truth.shape==prediction.shape and len(truth)>0
    assert np.issubdtype(truth.dtype,np.integer) and np.issubdtype(prediction.dtype,np.integer)
    assert set(truth)<=set(range(n_classes)) and set(prediction)<=set(range(n_classes))
    support=np.bincount(truth,minlength=n_classes)
    correct=np.bincount(truth[truth==prediction],minlength=n_classes)
    present=support>0;recall=np.divide(correct,support,out=np.zeros(n_classes,dtype=float),where=present)
    return dict(n=int(len(truth)),accuracy=float(np.mean(truth==prediction)),present_class_macro_recall=float(recall[present].mean()),class_support=support.tolist(),class_recall=[float(recall[k]) if present[k] else None for k in range(n_classes)],missing_classes=np.flatnonzero(~present).tolist(),metric_definition='macro recall averaged over classes with nonzero true support; absent-class recalls null, never zero-filled or silently called twelve-class macro')

