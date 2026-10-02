"""Audit completed moment-branch runs with functional forward and source-stat checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from audit_restored import digest,dump,expected_parts,metrics


def descriptor(seq,arm):
    z=np.stack([seq[...,j:j+6].mean(-1) for j in range(0,90,6)],axis=-1)
    if arm=='static':z=np.repeat(z.mean(-1,keepdims=True),15,-1)
    return z.reshape(len(seq),90)


@torch.no_grad()
def forward(x,m,s):
    n=len(x);x=x.reshape(n*3,4,90,63)
    for i in range(3):
        b=f'encoder.{i}.layers.'
        x=F.conv2d(x,s[b+'0.weight'],stride=2,padding=1)
        x=F.gelu(F.group_norm(x,4,s[b+'1.weight'],s[b+'1.bias']))
        x=F.conv2d(x,s[b+'3.weight'],padding=1)
        x=F.gelu(F.group_norm(x,4,s[b+'4.weight'],s[b+'4.bias']))
    x=F.adaptive_avg_pool2d(x,(3,4)).reshape(n,-1)
    m=(m-s['moment_mean'])/s['moment_scale']
    m=F.gelu(F.linear(m,s['moment_branch.0.weight'],s['moment_branch.0.bias']))
    m=F.gelu(F.linear(m,s['moment_branch.2.weight'],s['moment_branch.2.bias']))
    z=F.gelu(F.linear(torch.cat([x,m],1),s['head.1.weight'],s['head.1.bias']))
    return F.linear(z,s['head.4.weight'],s['head.4.bias'])


def main():
    ap=argparse.ArgumentParser()
    for n in ['data','prepared','moments','runs','out']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise ValueError('Use a new audit output')
    identity=json.loads((a.runs/'identity.json').read_text());config=identity['protocol']
    results=list(a.runs.glob('fold*/*/result.json'))
    if len(results)!=16:raise ValueError('Require all sixteen completed runs')
    assert digest(a.prepared/'signals.npy')==identity['signal_sha256']
    assert digest(a.moments/'moments.npy')==identity['moments_sha256']
    cache=np.load(a.prepared/'signals.npy',mmap_mode='r');seq=np.load(a.moments/'moments.npy',mmap_mode='r')
    y=np.load(a.data/'cache/y_clutterremoved.npy');u=np.load(a.data/'cache/subj_clutterremoved.npy')
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    rows=[]
    for path in sorted(results):
        r=json.loads(path.read_text());d=path.parent
        assert r['identity']==dict(identity,fold=r['held_user'],arm=r['arm'])
        for n,h in r['files'].items():assert digest(d/n)==h
        parts=expected_parts(y,u,r['held_user']);split=json.loads((d/'split.json').read_text())
        for role,ids in parts.items():np.testing.assert_array_equal(split[role],ids)
        ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False);s=ck['model']
        best=min(r['history'],key=lambda z:z['validation_ce'])
        assert best['epoch']==ck['epoch']==r['selected_epoch']
        assert best['validation_ce']==ck['validation_ce']==r['selected_val_ce']
        scaler=StandardScaler().fit(descriptor(seq[parts['train']],r['arm']))
        np.testing.assert_array_equal(scaler.mean_.astype(np.float32),s['moment_mean'].numpy())
        np.testing.assert_array_equal(scaler.scale_.astype(np.float32),s['moment_scale'].numpy())
        import hashlib
        for epoch in r['history'][1:]:
            order=np.random.default_rng(config['seed']+10000*r['held_user']+100000*epoch['epoch']).permutation(parts['train'])
            assert hashlib.sha256(order.tobytes()).hexdigest()==epoch['source_ids_sha256']
        pred=np.load(d/'predictions.npz')
        for role,idkey,key in [('val','validation_ids','validation_logits'),('test','ids','logits')]:
            ids=parts[role];np.testing.assert_array_equal(pred[idkey],ids)
            outputs=[]
            for j in range(0,len(ids),config['batch']):
                ix=ids[j:j+config['batch']];raw=np.asarray(cache[ix],dtype=np.float32)
                zero=np.zeros_like(raw[:,:,0]);x=np.stack([raw[:,:,0],raw[:,:,2],zero,zero],axis=2)
                m=descriptor(seq[ix],r['arm']).astype(np.float32)
                outputs.append(forward(torch.from_numpy(x),torch.from_numpy(m),s).numpy())
            logits=np.concatenate(outputs);np.testing.assert_array_equal(logits,pred[key])
            if role=='val':
                ce=F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y[ids]).long()).item()
                assert abs(ce-r['selected_val_ce'])<1e-7
            else:assert metrics(y[ids],logits.argmax(1))==r['metrics']
        rows.append({'fold':r['held_user'],'arm':r['arm'],'state':'bitwise_verified','epoch':best['epoch']})
    dump(a.out,{'state':'verified','runs':16,'artifact_sha_checks':64,'source_scaler_checked':True,
        'functional_forward_bitwise_equal':True,'rows':rows,'audit_script_sha256':digest(__file__)})
    print(json.dumps({'state':'verified','runs':16}))


if __name__=='__main__':main()
