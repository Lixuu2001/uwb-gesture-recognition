from pathlib import Path
import sys,time,json,resource,pickle,shutil,warnings
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
 (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,3,90,189]),indent=2)+'\n')
 sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-source-20261102');from native_resnet18 import np,make,extract,make_head,encoder_digest
 from sklearn.exceptions import ConvergenceWarning
 class Fixture:
  def __init__(self,n):self.shape=(n,3,90,189)
  def __len__(self):return self.shape[0]
  def __getitem__(self,k):
   assert isinstance(k,slice);a,b,s=k.indices(len(self));assert s==1
   return np.stack([np.random.default_rng(20261029+i).standard_normal((3,90,189),dtype=np.float32) for i in range(a,b)])
 def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\n')
 x=Fixture(4700);y=np.arange(4700)%12;state('full4700_complete_pretrained_ResNet18_frozen_feature_extraction_then_fixed_C1_SVM');t=time.monotonic();m=make();before=encoder_digest(m);features,hashes=extract(m,x,16);head=make_head()
 with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);head.fit(features,y)
 fit=time.monotonic()-t;assert head.n_iter_<10000 and features.shape==(4700,512) and encoder_digest(m)==before
 ev=[];saves=[];loads=[];sizes=[]
 for i in range(3):
  state('timing_complete_frozen_encoder_and_SVM_predict64',index=i);t=time.monotonic();evalf,_=extract(m,x[i*64:(i+1)*64],16);scores=head.decision_function(evalf);pred=head.predict(evalf);ev.append(time.monotonic()-t);assert np.isfinite(scores).all();np.testing.assert_array_equal(pred,head.classes_[scores.argmax(1)])
 for i in range(3):
  state('timing_weight_bound_head_checkpoint_save_reload',index=i);payload=dict(encoder_weights_sha256=p['weights_sha256'],encoder_state_sha256=before,head=head);path=P/f'head_synthetic_{i}.pkl';t=time.monotonic()
  with path.open('xb') as f:pickle.dump(payload,f,protocol=5)
  saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic()
  with path.open('rb') as f:q=pickle.load(f)
  loads.append(time.monotonic()-t);assert q['encoder_weights_sha256']==p['weights_sha256'] and q['encoder_state_sha256']==encoder_digest(m);np.testing.assert_array_equal(q['head'].coef_,head.coef_);np.testing.assert_array_equal(q['head'].intercept_,head.intercept_);np.testing.assert_array_equal(q['head'].decision_function(evalf),scores);np.testing.assert_array_equal(q['head'].predict(evalf),pred)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,fit,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_pretrained_ResNet18_frozen512_LinearSVC_C1_synthetic_cost',full_native_fit_seconds=fit,evaluation64_seconds=ev,complete_head_save_seconds=saves,complete_head_reload_seconds=loads,peak_RSS_KiB=rss,head_checkpoint_sizes=sizes,encoder_weights_sha256=p['weights_sha256'],encoder_state_sha256=before,full_encoder_frozen_unchanged=True,full_encoder_layers=[2,2,2,2],features=512,full_training_cases=4700,head_n_iter=int(head.n_iter_),head_converged=True,loaded_head_coefficients_decisions_predictions_exact=True,training_feature_batch_hashes=hashes,actual_UWB_values_labels_fit=False)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
