from pathlib import Path
import os,time,json
from common import B,ready,load_inputs,save,write,sha
def main():
    c=ready();out=B/'run01';out.mkdir(exist_ok=False);start=time.monotonic()
    write(out/'STATE.json',dict(state='loading_registered_source_train_only',pid=os.getpid(),diagnostic_labels_read=False))
    from engine import np,make,assert_native,fitted_digest,predict
    ids,users,loc,maps,y=load_inputs(c,training_labels=True)
    np.savez(out/'split.npz',ids=ids,users=users,**loc)
    write(out/'IDENTITY.json',dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_sha256=sha(B/'FROZEN_HASHES.json'),seed=20261029,train=4700,validation=1000,diagnostic=1200))
    m=make();save(out/'initial_unfitted.pkl',m)
    write(out/'STATE.json',dict(state='native_full_pipeline_fit_train_users3_to6_only',pid=os.getpid(),diagnostic_labels_read=False))
    x=np.ascontiguousarray(maps[loc['train']].transpose(0,2,1));t=time.monotonic();m.fit(x,y);fit_seconds=time.monotonic()-t;del x
    assert_native(m,True);assert m._scaler.n_samples_seen_==4700
    save(out/'native_fitted.pkl',m);digest=fitted_digest(m)
    write(out/'TRAIN_SELECTION.json',dict(alpha=float(m.estimator_.alpha_),best_training_LOO_negative_MSE=float(m.estimator_.best_score_),fit_seconds=fit_seconds,fitted_digest=digest,train_only=True,validation_labels_used=False,diagnostic_labels_used=False))
    for role in ['validation','diagnostic']:
        write(out/'STATE.json',dict(state='predicting_'+role+'_without_labels',pid=os.getpid(),diagnostic_labels_read=False))
        result=predict(m,maps,loc[role],ids);np.savez(out/(role+'_unscored.npz'),**result)
    save(out/'final_fitted.pkl',m);assert fitted_digest(m)==digest
    write(out/'STATE.json',dict(state='producer_complete_unscored',pid=None,fit_seconds=fit_seconds,wall_seconds=time.monotonic()-start,diagnostic_labels_read=False,users7_8_labels_used=False,training_only_alpha_selection=True))
    write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='FILES.json'})
    print(json.dumps(read_state(out)))
def read_state(out):
    from common import read
    return read(out/'STATE.json')
if __name__=='__main__':main()
