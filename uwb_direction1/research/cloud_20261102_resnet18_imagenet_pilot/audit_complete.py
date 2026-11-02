from pathlib import Path
import time,pickle
from common import B,ready,read,write,sha,load_inputs
def main():
 c=ready();start=time.monotonic();out=B/'run01';term=read(B/'supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout'] and not term['resource_stop_reason'] and term['cgroup_oom_kill_delta']==0
 for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n
 from engine import np,torch,make,extract,preprocess,encoder_digest,RoleMaps,fit_head,assert_payload,compare_heads,decisions,pixels_reference,ENCODER_SHA
 ids,users,loc,maps,y=load_inputs(c,True)
 with np.load(out/'split.npz') as z:
  for key,value in dict(ids=ids,users=users,**loc).items():np.testing.assert_array_equal(z[key],value)
 with (out/'initial_unfitted.pkl').open('rb') as f:initial=pickle.load(f)
 assert_payload(initial,False);model=make();assert encoder_digest(model)==ENCODER_SHA;features,hashes=extract(model,RoleMaps(maps,loc['train']),16)
 with np.load(out/'train_features.npz') as z:
  for key,value in dict(ids=ids[loc['train']],features=features,feature_batch_digests=np.asarray(hashes)).items():np.testing.assert_array_equal(z[key],value)
 head=fit_head(features,y)
 for name in ['native_fitted.pkl','final_fitted.pkl']:
  with (out/name).open('rb') as f:p=pickle.load(f)
  assert_payload(p,True);compare_heads(head,p['head'])
 for role in ['validation','diagnostic']:
  f,h=extract(model,RoleMaps(maps,loc[role]),16);scores,pred=decisions(head,f);independent=f.astype(np.float64)@head.coef_.T+head.intercept_;np.testing.assert_allclose(scores,independent,rtol=1e-12,atol=1e-12);np.testing.assert_array_equal(head.classes_[independent.argmax(1)],pred)
  with np.load(out/(role+'_unscored.npz')) as z:
   for key,value in dict(ids=ids[loc[role]],features=f,scores=scores,predictions=pred,feature_batch_digests=np.asarray(h)).items():np.testing.assert_array_equal(z[key],value)
 assert encoder_digest(model)==ENCODER_SHA;provider=RoleMaps(maps,np.arange(6900))
 for offset in range(0,6900,8):
  x=provider[offset:offset+8];np.testing.assert_allclose(preprocess(x).numpy(),pixels_reference(x),rtol=2e-5,atol=2e-5)
 raw=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
 for offset in range(0,6900,64):
  a=np.array(raw[ids[offset:offset+64]],copy=True);r=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32));ref=(a/np.maximum(r,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,maps[offset:offset+64],rtol=2e-5,atol=2e-6)
  d=a.astype(np.float64);r64=np.sqrt(np.mean(d*d,axis=(2,3),keepdims=True));double=(d/np.maximum(r64,1e-8)).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,double,rtol=2e-5,atol=2e-6)
 for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n
 result=dict(state='pass',full_frozen_encoder_all6900_feature_replay_exact=True,all4700_training_features_and_batch_hashes_exact=True,full_C1_SVM_refit_coefficients_intercept_classes_n_iter_exact=True,all2200_unscored_features_decisions_predictions_exact=True,independent_linear_decision_values_within_predeclared_tolerance=True,all6900_pixel_preprocessing_independent_FP64_verified=True,all6900_raw_RMS_cache_FP64_verified=True,shared_full_encoder_weights_and_BN_frozen_unchanged=True,initial_unfitted_native_final_payloads_verified=True,diagnostic_labels_read=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,auditor_sha256=sha(__file__),artifact_hashes={**{str(out/n):v for n,v in read(out/'FILES.json').items()},str(out/'FILES.json'):sha(out/'FILES.json')},audit_wall_seconds=time.monotonic()-start)
 write(B/'COMPLETE_AUDIT.json',result);print(__import__('json').dumps({k:v for k,v in result.items() if k!='artifact_hashes'}))
if __name__=='__main__':main()
