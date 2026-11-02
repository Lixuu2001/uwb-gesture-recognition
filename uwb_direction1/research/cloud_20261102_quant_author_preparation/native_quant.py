"""GPL-3.0-compatible QUANT integration;original author code remains unchanged.

Complete depth6/div4 transform and200-tree/.1/entropy classifier. Batched
feature extraction and Fortran storage change execution layout only. This is
future preparation,not a registered UWB experiment or an originality claim.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,importlib.util,hashlib,json
sys.path.insert(0,'/workspace/uwb-multirocket-20261029/vendor/site-packages')
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from joblib.externals import cloudpickle
torch.set_num_threads(1);torch.set_num_interop_threads(1)
B=Path(__file__).resolve().parent
index=json.loads((B/'SOURCE_INDEX.json').read_bytes())
for row in index['files']:
    assert hashlib.sha256((B/'local_author_sources'/row['path']).read_bytes()).hexdigest()==row['sha256']
spec=importlib.util.spec_from_file_location('pinned_author_quant',B/'local_author_sources/code/quant.py')
author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
cloudpickle.register_pickle_by_value(author)

def make():
    return dict(transform=author.Quant(depth=6,div=4),classifier=ExtraTreesClassifier(n_estimators=200,max_features=.1,criterion='entropy',n_jobs=4,random_state=20261029))

def features(model,x,y=None,fit=False,batch_size=32):
    """All rows/all features retained;only training feature initialization fits.

    x may be a numpy array or a shape+slice provider. No full input materialization
    is required. The feature matrix is dense float32 in RAM;no large disk array.
    """
    assert len(x.shape)==3 and x.shape[2]==90 and len(x)>0
    assert (y is not None)==fit
    q=model['transform'];out=None;digests=[]
    for start in range(0,len(x),batch_size):
        batch=np.ascontiguousarray(x[start:start+batch_size],dtype=np.float32);assert np.isfinite(batch).all()
        with torch.no_grad():
            z=(q.fit_transform(torch.from_numpy(batch),y[start:start+len(batch)]) if fit and start==0 else q.transform(torch.from_numpy(batch))).numpy()
        assert z.shape==(len(batch),951*x.shape[1]) and np.isfinite(z).all()
        if out is None:out=np.empty((len(x),z.shape[1]),dtype=np.float32,order='F')
        out[start:start+len(batch)]=z;digests.append(hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest())
    assert out.flags.f_contiguous and out.dtype==np.float32
    return out,digests

def fit(model,x,y,batch_size=32):
    f,hashes=features(model,x,y,True,batch_size);model['classifier'].fit(f,y)
    model['training_feature_batch_hashes']=hashes;model['training_cases']=len(x);model['feature_count']=f.shape[1]
    return model

def predict(model,x,batch_size=32):
    probabilities=[];hashes=[]
    for start in range(0,len(x),batch_size):
        f,d=features(model,x[start:start+batch_size],None,False,batch_size)
        p=model['classifier'].predict_proba(f);assert np.isfinite(p).all()
        probabilities.append(p);hashes.extend(d)
    prob=np.concatenate(probabilities)
    return dict(probabilities=prob,predictions=model['classifier'].classes_[prob.argmax(1)],feature_batch_hashes=hashes)

def scientific_arrays(model):
    for i in range(4):yield model['transform'].models[i].intervals.numpy()
    yield model['classifier'].classes_
    for estimator in model['classifier'].estimators_:
        tree=estimator.tree_
        for name in ['children_left','children_right','feature','threshold','impurity','n_node_samples','weighted_n_node_samples','value']:yield np.asarray(getattr(tree,name))
