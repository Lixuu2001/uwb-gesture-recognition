"""Frozen source-only envelope matching screen; no classifier training."""
import argparse,json,time,platform,sys
from pathlib import Path
import numpy as np
import scipy
from audit_restored import digest,dump
from local_echo_correspondence import envelope,match,select_ids,summarize


def extract(raw,c):
    e=envelope(raw);t=np.array(c['frame_starts']);a=e[:,t].reshape(-1,e.shape[-1]);out={}
    for arm,dest in [('adjacent',t+1),('distant',(t+31)%90)]:
        b=e[:,dest].reshape(a.shape);r=match(a,b,np.array(c['centers']),c)
        for key,value in r.items():out[arm+'_'+key]=value.reshape((3,len(t),len(c['centers']))+value.shape[2:])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh independent output directory')
    a.out.mkdir(parents=True);c=json.loads(a.protocol.read_text());data=a.data/'cache'
    names=['X_clutterremoved.npy','y_clutterremoved.npy','subj_clutterremoved.npy'];hashes={n:digest(data/n) for n in names}
    assert hashes['X_clutterremoved.npy']=='3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519'
    y=np.load(data/names[1]);u=np.load(data/names[2]);ids=select_ids(y,u,c);np.save(a.out/'ids.npy',ids)
    raw=np.load(data/names[0],mmap_mode='r');assert raw.shape==(9593,3,90,189)
    identity={'raw_files_sha256':hashes,'protocol_sha256':digest(a.protocol),'code_sha256':{n:digest(Path(__file__).parent/n) for n in ['run_local_echo_screen.py','local_echo_correspondence.py']},'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),'argv':sys.argv,'source_records':len(ids)}
    dump(a.out/'IDENTITY.json',identity);dump(a.out/'PROTOCOL.json',c)
    started=time.perf_counter();outputs=[extract(raw[i],c) for i in ids[:2]];elapsed=time.perf_counter()-started
    estimate=2*elapsed/2*len(ids);cost={'records_timed':2,'seconds':elapsed,'estimate_with_100percent_margin_seconds':estimate,'budget_seconds':c['cost_seconds_budget'],'passed':estimate<=c['cost_seconds_budget'],'quality_metrics_not_used_for_configuration':True};dump(a.out/'COST.json',cost);print(json.dumps({'cost':cost}),flush=True)
    if not cost['passed']:dump(a.out/'STATE.json',{'state':'cost_gate_failed'});return
    for j,i in enumerate(ids[2:],2):
        outputs.append(extract(raw[i],c))
        if (j+1)%100==0:print(json.dumps({'processed':j+1,'total':len(ids)}),flush=True)
    arrays={k:np.stack([r[k] for r in outputs]) for k in outputs[0]}
    # Keep original float64 cost volumes: independent confidence decisions can be replayed exactly.
    np.savez_compressed(a.out/'correspondences.npz',ids=ids,**arrays)
    summary=summarize(ids,u,arrays,c);summary.update(identity=identity,seconds=time.perf_counter()-started,source_records=len(ids),patch_comparisons_per_arm=int(len(ids)*3*len(c['frame_starts'])*len(c['centers'])))
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{n:digest(a.out/n) for n in ['ids.npy','correspondences.npz','IDENTITY.json','PROTOCOL.json','COST.json','SUMMARY.json']});dump(a.out/'STATE.json',{'state':'complete','gate_passed':summary['gate_passed'],'classifier_trained':False})
    print(json.dumps({'state':'complete','gate_passed':summary['gate_passed'],'passing_users_per_view':summary['passing_users_per_view'],'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':main()
