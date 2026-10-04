"""Fixed source-only orientation diagnostic. No classifier fitting."""
import argparse,json,time,platform,sys
from pathlib import Path
import numpy as np
import scipy
from audit_restored import digest,dump
from orientation_screen import extract,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','ids','protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();a.out.mkdir();c=json.loads(a.protocol.read_text());assert digest(a.ids)==c['ids_sha256'];ids=np.load(a.ids);assert len(ids)==568 and len(np.unique(ids))==568
    paths={k:a.data/'cache'/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]};hashes={k:digest(v) for k,v in paths.items()};assert hashes==c['input_sha256']
    raw=np.load(paths['raw'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users']);assert set(u[ids])==set(range(3,9));assert not np.any((u[ids]==6)&(y[ids]==3))
    for user in range(3,9):
        for label in range(12):assert np.sum((u[ids]==user)&(y[ids]==label))==(0 if user==6 and label==3 else 8)
    identity={'inputs':hashes,'ids_sha256':digest(a.ids),'protocol_sha256':digest(a.protocol),'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['orientation_screen.py','run_orientation_screen.py']},'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),'argv':sys.argv}
    dump(a.out/'IDENTITY.json',identity);dump(a.out/'PROTOCOL.json',c);np.save(a.out/'ids.npy',ids)
    t=time.perf_counter();outputs=[extract(raw[int(i)],int(i),c['seed']) for i in ids[:2]];elapsed=time.perf_counter()-t;estimate=2*elapsed/2*len(ids)
    cost={'timed_source_clips':2,'seconds':elapsed,'estimate_with_2x_margin_seconds':estimate,'budget_seconds':c['budget_seconds'],'passed':estimate<c['budget_seconds'],'no_quality_scores_used_for_configuration':True};dump(a.out/'COST.json',cost);print(json.dumps({'cost':cost}),flush=True)
    if not cost['passed']:dump(a.out/'STATE.json',{'state':'cost_gate_failed','classifier_trained':False});return
    for i,sid in enumerate(ids[2:],2):
        outputs.append(extract(raw[int(sid)],int(sid),c['seed']))
        if (i+1)%100==0:print(json.dumps({'processed':i+1}),flush=True)
    tensors=np.stack([z[0] for z in outputs]);permutations=np.stack([z[1] for z in outputs]);np.savez_compressed(a.out/'tensors.npz',ids=ids,tensors=tensors,permutations=permutations)
    summary=summarize(tensors,u[ids]);summary.update(source_clips=len(ids),tensor_locations_per_arm=len(ids)*3*126,seconds=time.perf_counter()-t,classifier_trained=False)
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'STATE.json',{'state':'complete','gate_passed':summary['gate_passed'],'classifier_trained':False});dump(a.out/'FILES.json',{p.name:digest(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    print(json.dumps({'state':'complete','gate_passed':summary['gate_passed'],'view_pass':summary['view_pass'],'seconds':summary['seconds']}),flush=True)

if __name__=='__main__':main()
