"""Offline scoring and independent raw-forward audit, after immutable stream runs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from audit_restored import digest,dump
from audit_circular_signal import independent_forward
from circular_signal import get_batch


def main(root):
    torch.set_num_threads(1);c=json.loads((root/'protocol.json').read_text());run=json.loads((root/'RUN.json').read_text())
    data=Path('/workspace/uwb-restored-20260930/cache');yp=data/'y_clutterremoved.npy';up=data/'subj_clutterremoved.npy'
    assert digest(yp)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
    assert digest(up)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
    y=np.load(yp);u=np.load(up);cache=np.load(c['cache'],mmap_mode='r');scores={o:{a:[] for a in c['arms']} for o in c['orders']};verified=0;rawchecks=0;maxerr=0;update_counts={};resets={}
    for rec in run['folds']:
        h=rec['held'];d=root/f'held{h}'
        for name,sha in rec['files'].items():assert digest(d/name)==sha;verified+=1
        assert digest(rec['checkpoint'])==rec['checkpoint_sha256'];initial=torch.load(rec['checkpoint'],weights_only=False,map_location='cpu')['states']['tail']
        split=json.loads((d/'split.json').read_text());v=h+1 if h<8 else 3;good=(u>=3)&(u<=8)&~((u==6)&(y==3))
        for role,mask in [('train',good&(u!=h)&(u!=v)),('val',good&(u==v)),('diagnostic',good&(u==h))]:np.testing.assert_array_equal(split[role],np.flatnonzero(mask))
        pred=np.load(d/'predictions.npz');logs=json.loads((d/'updates.json').read_text())
        for o in c['orders']:
            ids=pred[o+'_ids'];ref=np.array(split['diagnostic']);ref=ref if o=='storage' else np.random.default_rng(c['order_seed']+h).permutation(ref);np.testing.assert_array_equal(ids,ref)
            for arm in c['arms']:
                z=pred[o+'_'+arm];assert z.shape==(len(ids),12) and np.isfinite(z).all();correct=z.argmax(1)==y[ids]
                macro=float(np.mean([correct[y[ids]==k].mean() for k in np.unique(y[ids])]))
                scores[o][arm].append(macro);row=next(r for r in logs if r['order']==o and r['arm']==arm)
                np.testing.assert_array_equal(np.concatenate([r['ids'] for r in row['batches']]),ids)
                states=torch.load(d/f'{o}_{arm}_states.pt',weights_only=False,map_location='cpu');assert len(states)==len(row['batches'])
                for j,s in enumerate(states):
                    assert all(torch.isfinite(t).all() for t in s['model'].values());assert s['steps']==j+1
                    assert s['updates']==sum(r['update'] for r in row['batches'][:j+1]);assert s['resets']==sum(r['reset'] for r in row['batches'][:j+1])
                for j in sorted(set([0,1,len(states)-1])):
                    state=dict(initial)
                    if j:state.update(states[j-1]['model'])
                    b=j*c['batch'];x=get_batch(cache,ids[b:b+c['batch']],'legacy');refz=independent_forward(x,state).numpy();err=float(abs(refz-z[b:b+c['batch']]).max());maxerr=max(maxerr,err);assert err<2e-5;rawchecks+=1
                key=o+'_'+arm;update_counts[key]=update_counts.get(key,0)+row['updates'];resets[key]=resets.get(key,0)+row['resets']
    gates={};details={}
    for o,ss in scores.items():
        primary=np.array(ss['sar']);base=np.array(ss['frozen']);diff=primary-base
        gates[o]=bool(diff.mean()>=c['gate']['mean_gain_vs_frozen'] and (diff>0).sum()>=c['gate']['min_wins'] and diff.min()>=-c['gate']['max_worst_decline'] and all(primary.mean()>=np.mean(ss[a]) for a in ['entropy','filtered']))
        details[o]=dict(gain_vs_frozen=float(diff.mean()),wins=int((diff>0).sum()),worst_decline=float(diff.min()))
    result=dict(scores=scores,means={o:{a:float(np.mean(s)) for a,s in ss.items()} for o,ss in scores.items()},gate_per_order=gates,passed=all(gates.values()),details=details,updates=update_counts,resets=resets,seconds=run['seconds'],scope='Source users3..8 development; prequential unlabeled input adaptation; not an original contribution or independent8-fold result')
    dump(root/'SUMMARY.json',result);dump(root/'AUDIT.json',dict(passed=True,artifact_sha256_checks=verified,independent_raw_forward_batches=rawchecks,max_raw_forward_error=maxerr,roles_rebuilt=True,order_rebuilt_without_labels=True,all_update_counters_recomputed=True))
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
