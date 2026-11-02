from pathlib import Path
import time,sys
from common import B,ready,read,write,sha,save,load_inputs
def main():
 c=ready();start=time.monotonic();out=B/'run01';out.mkdir(exist_ok=False)
 from engine import np,make,extract,make_head,encoder_digest,RoleMaps,payload,assert_payload,fit_head,decisions,ENCODER_SHA
 ids,users,loc,maps,y=load_inputs(c,True);np.savez(out/'split.npz',ids=ids,users=users,**loc)
 identity=dict(source_manifest_sha256=c['source_manifest_sha256'],source_maps_sha256=c['source_maps_sha256'],protocol_sha256=sha(B/'PROTOCOL.json'),training_users=[3,4,5,6],counts=c['counts'],validation_labels_read=False,diagnostic_labels_read=False,users7_8_labels_used=False)
 write(out/'IDENTITY.json',identity);model=make();assert encoder_digest(model)==ENCODER_SHA;initial=payload(make_head());assert_payload(initial,False);save(out/'initial_unfitted.pkl',initial)
 write(out/'ENCODER_STATE.json',dict(encoder_weights_sha256=c['weights_sha256'],encoder_state_sha256=ENCODER_SHA,all11176512_parameters_frozen=True,BN_eval_unchanged=True,shared_encoder_path=c['encoder_weights']))
 write(out/'STATE.json',dict(state='full_frozen_encoder_train3_to6_and_fixed_C1_SVM',diagnostic_labels_read=False,users7_8_labels_used=False));t=time.monotonic();features,hashes=extract(model,RoleMaps(maps,loc['train']),16);head=fit_head(features,y);fit=time.monotonic()-t
 np.savez_compressed(out/'train_features.npz',ids=ids[loc['train']],features=features,feature_batch_digests=np.asarray(hashes));fitted=payload(head);assert_payload(fitted,True);save(out/'native_fitted.pkl',fitted);save(out/'final_fitted.pkl',fitted)
 write(out/'TRAIN_SELECTION.json',dict(selection='Noheadparametergrid;fixedC1nativeLinearSVCtrainingonly',C=1,head_n_iter=int(head.n_iter_),head_converged=True,training_users=[3,4,5,6],training_cases=4700,feature_count=512,full_encoder_frozen=True,native_fit_seconds=fit,encoder_state_sha256=encoder_digest(model),validation_labels_read=False,diagnostic_labels_read=False,users7_8_labels_used=False))
 for role in ['validation','diagnostic']:
  f,h=extract(model,RoleMaps(maps,loc[role]),16);scores,pred=decisions(head,f);np.savez_compressed(out/(role+'_unscored.npz'),ids=ids[loc[role]],features=f,scores=scores,predictions=pred,feature_batch_digests=np.asarray(h))
 assert encoder_digest(model)==ENCODER_SHA;write(out/'STATE.json',dict(state='producer_complete_unscored',producer_wall_seconds=time.monotonic()-start,diagnostic_labels_read=False,users7_8_labels_used=False))
 write(out/'FILES.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='FILES.json'});print(__import__('json').dumps(dict(state='producer_complete_unscored',fit_seconds=fit,wall_seconds=time.monotonic()-start)))
if __name__=='__main__':main()
