"""Fixed role mapping,head state,independent numerical checks;no held labels."""
from pathlib import Path
import sys,json,pickle,hashlib,warnings
sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-source-20261102')
from native_resnet18 import np,torch,make,preprocess,extract,make_head,encoder_digest
from sklearn.exceptions import ConvergenceWarning
WEIGHTS_SHA='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
ENCODER_SHA='3be2865fb07f7181ca5436322189029c373de0c5abab965c8881d7c37b0c2f3d'
SCOPE='complete_audited_ResNet18_ImageNetV1_frozen512_LinearSVC_C1_source_held1'
class RoleMaps:
 def __init__(self,maps,indices):self.maps=maps;self.indices=np.asarray(indices);self.shape=(len(indices),3,90,189)
 def __len__(self):return self.shape[0]
 def __getitem__(self,k):
  assert isinstance(k,slice);ix=self.indices[k];a=np.asarray(self.maps[ix]);assert a.shape==(len(ix),90,567)
  return np.ascontiguousarray(a.reshape(len(ix),90,3,189).transpose(0,2,1,3),dtype=np.float32)
def fit_head(features,labels):
 assert features.shape==(4700,512) and labels.shape==(4700,) and np.isfinite(features).all()
 m=make_head()
 with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);m.fit(features,labels)
 assert_head(m,True);return m
def assert_head(m,fitted=False):
 assert m.get_params()==make_head().get_params()
 if fitted:
  assert m.coef_.shape==(12,512) and m.intercept_.shape==(12,) and np.array_equal(m.classes_,np.arange(12))
  assert np.isfinite(m.coef_).all() and np.isfinite(m.intercept_).all() and m.n_iter_<10000
 else:assert not hasattr(m,'coef_')
def payload(head):return dict(encoder_weights_sha256=WEIGHTS_SHA,encoder_state_sha256=ENCODER_SHA,head=head)
def assert_payload(p,fitted=False):
 assert p['encoder_weights_sha256']==WEIGHTS_SHA and p['encoder_state_sha256']==ENCODER_SHA;assert_head(p['head'],fitted)
def compare_heads(a,b):
 assert_head(a,True);assert_head(b,True)
 for name in ['coef_','intercept_','classes_','n_iter_']:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
def decisions(head,features):
 scores=head.decision_function(features);pred=head.predict(features);assert np.isfinite(scores).all() and scores.shape==(len(features),12)
 np.testing.assert_array_equal(pred,head.classes_[scores.argmax(1)]);return scores,pred
def pixels_reference(x):
 x=x.astype(np.float64);lo=x.min(axis=(-2,-1),keepdims=True);hi=x.max(axis=(-2,-1),keepdims=True);u=(x-lo)/np.maximum(hi-lo,1e-8)
 yy=(np.arange(224)+.5)*90/224-.5;xx=(np.arange(224)+.5)*189/224-.5;y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int);wy=yy-y0;wx=xx-x0;y1=np.clip(y0+1,0,89);x1=np.clip(x0+1,0,188);y0=np.clip(y0,0,89);x0=np.clip(x0,0,188)
 ref=u[:,:,y0[:,None],x0[None,:]]*(1-wy)[:,None]*(1-wx)[None,:]+u[:,:,y1[:,None],x0[None,:]]*wy[:,None]*(1-wx)[None,:]+u[:,:,y0[:,None],x1[None,:]]*(1-wy)[:,None]*wx[None,:]+u[:,:,y1[:,None],x1[None,:]]*wy[:,None]*wx[None,:]
 return (ref-np.array([.485,.456,.406])[None,:,None,None])/np.array([.229,.224,.225])[None,:,None,None]
def private_scope(m,v,r,bound):
 assert m['scope']==v['scope']==r['scope']==SCOPE and m['complete_modelset'] and m['shared_frozen_encoder_included_once']
 assert v['complete_modelset_verified'] and r['complete_modelset_verified'] and r['numeric_audit_certified'] and r['state']=='private_readback_verified'
 assert m['archive_sha256']==v['archive_sha256']==r['archive_sha256']
 assert bound['source_preparation/weights/resnet18-f37072fd.pth']==WEIGHTS_SHA
