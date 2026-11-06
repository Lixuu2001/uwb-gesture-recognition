"""Blind native-output/roles/selection compatibility, never compute new held metrics."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
B=Path(__file__).resolve().parent;G=Path('/workspace/uwb-cloud-current/uwb_direction1')
sys.path.insert(0,'/workspace/uwb-mmd-gap64-20261106');import engine as e
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic();paths=json.loads((B/'BASELINE_ARTIFACTS.json').read_text())['restored_folds'];audit=json.loads((G/'results/cloud_20261030_gap_source6/COMPLETE_AUDIT.json').read_text())
    with np.load('/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz') as z:ids=z['ids'];u=z['users']
    maps=np.load('/workspace/uwb-timemil-20261029/run02/source_maps.npy',mmap_mode='r');rows=[]
    for held in range(2,7):
        fold=Path(paths[str(held)]);v=held%6+1;loc=dict(train=np.flatnonzero((u!=held)&(u!=v)),validation=np.flatnonzero(u==v),diagnostic=np.flatnonzero(u==held))
        split=json.loads((fold/'split.json').read_text())
        for k in loc:np.testing.assert_array_equal(split[k],ids[loc[k]])
        for n in ['best.pt','initial.pt','first_update.pt','last.pt','predictions_unscored.npz','epochs.jsonl']:
            key=f'/workspace/uwb-gap-sixfold-20261030/run01/held{held}/{n}';assert sha(fold/n)==audit['artifact_hashes'][key],key
        initial=e.torch.load(fold/'initial.pt',map_location='cpu',weights_only=False);assert e.semantic(initial['model'])=='2aae73c58e38ca324027e888955b130a82fbaece7167df3465522ec51986d2d3'
        best=e.torch.load(fold/'best.pt',map_location='cpu',weights_only=False);records=[json.loads(x) for x in (fold/'epochs.jsonl').read_text().splitlines()];assert len(records)==64
        selected=max(records,key=lambda q:(q['validation_macro'],-q['validation_loss'],-q['epoch']))
        assert best['epoch']==selected['epoch']==audit['folds'][held-1]['selected_epoch']
        m,_,_=e.make();m.load_state_dict(best['model'],strict=True);actual=e.evaluate(m,maps,loc['diagnostic'])
        with np.load(fold/'predictions_unscored.npz') as z:np.testing.assert_array_equal(z['ids'],ids[loc['diagnostic']]);stored=np.array(z['logits'],copy=True)
        assert actual.shape==stored.shape and np.isfinite(actual).all();np.testing.assert_array_equal(actual.argmax(1),stored.argmax(1))
        rows.append(dict(held=held,validation_user=v,training_users=sorted(set(u[loc['train']].tolist())),counts={k:len(ix) for k,ix in loc.items()},epochs=64,selected_epoch=best['epoch'],checkpoint_sha256=sha(fold/'best.pt'),prediction_sha256=sha(fold/'predictions_unscored.npz'),initial_model_exact=True,role_IDs_exact=True,validation_only_selection_exact=True,logits_bitwise_equal=bool(np.array_equal(actual,stored)),all_argmax_exact=True,max_logit_absolute_difference=float(np.abs(actual-stored).max()),original_runtime=best.get('identity',{}).get('torch')))
        print(json.dumps(rows[-1]),flush=True);del m
    q=dict(state='recovered_existing_GAP_held2_6_native_output_role_initialization_selection_compatibility_pass',folds=rows,current_torch=e.torch.__version__,no_held_truth_or_new_accuracy_computed=True,no_new_training_or_optimizer_updates=True,wall_seconds=time.monotonic()-start)
    (B/'NATIVE_BASELINE_AUDIT.json').write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()
