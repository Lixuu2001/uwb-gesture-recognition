"""Unchanged modern aeon DrCIF at paper-scale500 trees with native CIT.

Published interval ensemble transfer;not original Java runtime equivalence/novelty.
Full500/10-of29/per-representation(4,sqrt-div)/CIT20 thresholds. Native
random interval duplicate removal,periodogram padding,NaN path are retained.
Float32 RMS values are exactly promoted to float64 before native execution to
avoid a demonstrated Catch22 Numba float32/float64 branch typing failure.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,pickle,copy
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-multirocket-20261029/vendor/site-packages')
import numpy as np
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.base._estimators.interval_based.base_interval_forest import BaseIntervalForest
B=Path(__file__).resolve().parent
def make(jobs=4):
 return DrCIFClassifier(base_estimator=ContinuousIntervalTree(),n_estimators=500,n_intervals=(4,'sqrt-div'),min_interval_length=3,max_interval_length=.5,att_subsample_size=10,time_limit_in_minutes=None,random_state=20261029,n_jobs=jobs,parallel_backend='threading',verbose=0)
def fit(m,x,y):return m.fit(np.ascontiguousarray(x,dtype=np.float64),y)
def assert_native(m,fitted=False,jobs=4):
 assert m.n_estimators==500 and m.n_intervals==(4,'sqrt-div') and m.att_subsample_size==10 and m.min_interval_length==3 and m.max_interval_length==.5
 assert m.random_state==20261029 and m.n_jobs==jobs and m.parallel_backend=='threading' and m.time_limit_in_minutes is None
 assert isinstance(m.base_estimator,ContinuousIntervalTree) and m.base_estimator.thresholds==20 and m.base_estimator.max_depth==sys.maxsize and m.replace_nan=='nan'
 if fitted:
  assert m._n_estimators==len(m.estimators_)==len(m.intervals_)==500 and len(m._n_intervals)==3
  assert np.array_equal(m.classes_,np.arange(12)) and len(m._series_transformers)==3
  lengths=[m.n_timepoints_,m.n_timepoints_-1,2**int(np.ceil(np.log2(m.n_timepoints_)))//2]
  assert m._n_intervals==[4+int(np.sqrt(t)*np.sqrt(m.n_channels_)/3) for t in lengths]
  for t,selectors in zip(m.estimators_,m.intervals_):
   assert isinstance(t,ContinuousIntervalTree) and t.thresholds==20 and len(selectors)==3
   for r,s in enumerate(selectors):assert s.n_intervals==m._n_intervals[r] and 1<=s.n_intervals_<=s.n_intervals
def node_state(n):
 return dict(best_split=int(n.best_split),best_threshold=float(n.best_threshold).hex(),best_gain=float(n.best_gain).hex(),best_margin=float(n.best_margin).hex(),depth=int(n.depth),leaf_distribution=[float(v).hex() for v in n.leaf_distribution],children=[node_state(c) if c is not None else None for c in n.children])
def feature_state(f):
 if hasattr(f,'get_params'):return dict(type=type(f).__module__+'.'+type(f).__name__,parameters=f.get_params(deep=False))
 return dict(function=getattr(f,'__module__','')+'.'+getattr(f,'__name__',type(f).__name__))
def learned_state(m):
 return dict(classes=m.classes_.tolist(),interval_counts=list(m._n_intervals),trees=[dict(classes=t.classes_.tolist(),root=node_state(t._root)) for t in m.estimators_],selectors=[[dict(intervals=[dict(start=int(a),end=int(b),channel=int(c),feature=feature_state(f),dilation=int(d)) for a,b,c,f,d in s.intervals_],transform_feature_mask=s._transform_features) for s in group] for group in m.intervals_])
def digest(m):
 return hashlib.sha256(json.dumps(learned_state(m),sort_keys=True,separators=(',',':'),default=lambda x:x.tolist() if hasattr(x,'tolist') else repr(x)).encode()).hexdigest()
def predict(m,x,batch_size=32):
 ps=[]
 for start in range(0,len(x),batch_size):
  p=m.predict_proba(np.ascontiguousarray(x[start:start+batch_size],dtype=np.float64));assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,rtol=1e-12,atol=1e-12);ps.append(p)
 p=np.concatenate(ps);return dict(probabilities=p,predictions=m.classes_[p.argmax(1)])
