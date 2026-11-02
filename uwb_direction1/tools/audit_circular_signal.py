"""Independently replay completed signal-CNN checkpoints and source inputs.

Does not import the production model or training code. A partial audit only
covers completed runs and must not be described as a full experiment audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import torch
from torch.nn import functional as F

from audit_restored import digest, dump, expected_parts, metrics


def independent_encode(raw):
    z=hilbert(raw.astype(np.float64),axis=-1)
    z-=z.mean(axis=-2,keepdims=True)
    prod=z[...,1:,:]*np.conj(z[...,:-1,:])
    w=np.sqrt(np.abs(prod))
    angle=w*np.angle(prod)
    real=np.divide(prod.real,w,out=np.zeros_like(w),where=w>0)
    imag=np.divide(prod.imag,w,out=np.zeros_like(w),where=w>0)
    increments=np.stack([w,angle,np.abs(angle),real,imag],axis=2).astype(np.float32)
    increments=np.concatenate([increments[...,:1,:],increments],axis=-2)
    x=np.concatenate([np.abs(raw).astype(np.float32)[:,:,None],increments],axis=2)
    x=(x-x.mean(axis=(-2,-1),keepdims=True))/(x.std(axis=(-2,-1),keepdims=True)+1e-6)
    x=x.astype(np.float16).astype(np.float32)
    return x.reshape(len(x),3,6,90,63,3).mean(-1).astype(np.float16)


@torch.no_grad()
def independent_forward(x,state):
    n=len(x)
    x=x.reshape(n*3,4,90,63)
    for stage in range(3):
        base=f'encoder.{stage}.layers.'
        x=F.conv2d(x,state[base+'0.weight'],stride=2,padding=1)
        x=F.gelu(F.group_norm(x,4,state[base+'1.weight'],state[base+'1.bias']))
        x=F.conv2d(x,state[base+'3.weight'],padding=1)
        x=F.gelu(F.group_norm(x,4,state[base+'4.weight'],state[base+'4.bias']))
    x=F.adaptive_avg_pool2d(x,(3,4)).reshape(n,-1)
    x=F.gelu(F.linear(x,state['head.1.weight'],state['head.1.bias']))
    return F.linear(x,state['head.4.weight'],state['head.4.bias'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','prepared','runs','out','benchmark'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Use a new audit output file')
    config=json.loads((args.runs/'protocol.json').read_text())
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    cache=np.load(args.prepared/'signals.npy',mmap_mode='r')
    prep=json.loads((args.prepared/'PREPARED.json').read_text())
    assert digest(args.prepared/'signals.npy')==prep['sha256']
    probe=json.loads(args.benchmark.read_text())
    raw=np.load(args.data/'cache/X_clutterremoved.npy',mmap_mode='r')
    ids=np.array(probe['source_ids'])
    recomputed=independent_encode(raw[ids])
    np.testing.assert_array_equal(recomputed,cache[ids])
    y=np.load(args.data/'cache/y_clutterremoved.npy')
    users=np.load(args.data/'cache/subj_clutterremoved.npy')
    assert np.isin(ids,expected_parts(y,users,1)['train']).all()
    evidence=[]
    channels={'legacy':(0,2,-1,-1),'angle_control':(0,1,2,3),'circular':(0,1,4,5)}
    for file in sorted(args.runs.glob('fold*/*/result.json')):
        r=json.loads(file.read_text());folder=file.parent
        for name,sha in r['files'].items():assert digest(folder/name)==sha
        parts=expected_parts(y,users,r['held_user'])
        splits=json.loads((folder/'split.json').read_text())
        for name,indices in parts.items():np.testing.assert_array_equal(indices,splits[name])
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        assert best['identity']==r['identity']
        chosen=min(r['history'],key=lambda x:x['validation_ce'])
        assert chosen['epoch']==r['selected_epoch']==best['epoch']
        assert chosen['validation_ce']==r['selected_val_ce']==best['validation_ce']
        # Reconstruct every saved source epoch order without importing sampler.
        for row in r['history'][1:]:
            rng=np.random.default_rng(config['seed']+10000*r['held_user']+100000*row['epoch'])
            order=rng.permutation(parts['train'])
            import hashlib
            assert hashlib.sha256(order.tobytes()).hexdigest()==row['source_ids_sha256']
        diffs={}
        with np.load(folder/'predictions.npz') as pred:
            for role,ikey,lkey in [('val','validation_ids','validation_logits'),('test','ids','logits')]:
                np.testing.assert_array_equal(pred[ikey],parts[role])
                outputs=[]
                for i in range(0,len(parts[role]),config['batch']):
                    x=np.asarray(cache[parts[role][i:i+config['batch']]],dtype=np.float32)
                    x=np.stack([x[:,:,c] if c>=0 else np.zeros_like(x[:,:,0]) for c in channels[r['arm']]],axis=2)
                    outputs.append(independent_forward(torch.from_numpy(x),best['model']).numpy())
                logits=np.concatenate(outputs)
                diffs[role]=float(np.max(np.abs(logits-pred[lkey])))
                np.testing.assert_array_equal(logits,pred[lkey])
            np.testing.assert_array_equal(pred['y'],y[parts['test']])
            assert metrics(pred['y'],pred['logits'].argmax(1))==r['metrics']
            ce=F.cross_entropy(torch.from_numpy(pred['validation_logits']),torch.from_numpy(y[parts['val']]).long()).item()
            np.testing.assert_allclose(ce,r['selected_val_ce'],atol=1e-7,rtol=0)
        evidence.append({'fold':r['held_user'],'arm':r['arm'],'max_absolute_logit_error':diffs})
    record={'state':'verified','scope':'completed_runs_only','completed_runs_audited':len(evidence),
            'expected_runs':8*len(config['arms']),'full_experiment_complete':len(evidence)==8*len(config['arms']),
            'source_input_replay_exact':True,'source_input_ids':ids.tolist(),'prepared_sha256':prep['sha256'],
            'runs':evidence,'audit_script_sha256':digest(__file__)}
    dump(args.out,record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('runs','source_input_ids')}))


if __name__=='__main__':main()
