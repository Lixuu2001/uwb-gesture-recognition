"""Functional inference and independent split/selection/order checks."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audit_restored import digest, metrics
from full_view_heads import array_hash


@torch.no_grad()
def functional_forward(x, state, arm):
    def encode(z, prefix):
        for stage in range(3):
            b=f'{prefix}.{stage}.layers.'
            z=F.conv2d(z,state[b+'0.weight'],stride=2,padding=1)
            z=F.gelu(F.group_norm(z,4,state[b+'1.weight'],state[b+'1.bias']))
            d=(2,1) if arm=='time2' else (1,2) if arm=='range2' else (1,1)
            z=F.conv2d(z,state[b+'3.weight'],padding=d,dilation=d)
            z=F.gelu(F.group_norm(z,4,state[b+'4.weight'],state[b+'4.bias']))
        return F.adaptive_avg_pool2d(z,(3,4))
    if 'encoder.0.layers.0.weight' in state:
        z=encode(x.reshape(len(x)*3,4,90,63),'encoder').reshape(len(x),-1)
    else:
        z=torch.cat([encode(x[:,v].contiguous(),f'encoders.{v}').flatten(1) for v in range(3)],1)
    z=F.gelu(F.linear(z,state['head.1.weight'],state['head.1.bias']))
    return F.linear(z,state['head.4.weight'],state['head.4.bias'])


def audit_run(folder, cache, y, users, c):
    r=json.loads((folder/'result.json').read_text());h=r['held_source_user']
    for name,sha in r['files'].items(): assert digest(folder/name)==sha
    good=(users>=3)&(users<=8)&~((users==6)&(y==3));v=h+1 if h<8 else 3
    roles={'train':np.flatnonzero(good&(users!=h)&(users!=v)),
           'val':np.flatnonzero(good&(users==v)), 'diagnostic':np.flatnonzero(good&(users==h))}
    split=json.loads((folder/'split.json').read_text())
    for name,ids in roles.items(): np.testing.assert_array_equal(split[name],ids)
    s=torch.load(folder/'selected.pt',weights_only=False,map_location='cpu')
    p=torch.load(folder/'progress.pt',weights_only=False,map_location='cpu')
    assert s['identity']==p['identity']==r['identity']
    assert len(r['history'])==c['epochs']+1
    assert min(r['history'],key=lambda x:x['validation_ce'])['epoch']==s['selection']['erm_epoch']
    for k,t in s['states']['erm'].items(): assert torch.equal(t,p['best'][k])
    for k,t in s['states']['tail'].items(): assert torch.equal(t,p['tail']['weights']['module.'+k])
    nsteps=int(np.ceil(len(roles['train'])/c['batch']))
    assert s['selection']['tail_steps']==nsteps*(c['epochs']-c['tail_start_epoch']+1)
    for row in r['history'][1:]:
        order=np.random.default_rng(c['seed']+10000*h+100000*row['epoch']).permutation(roles['train'])
        assert array_hash(order)==row['source_ids_sha256']
        assert row['step']==row['epoch']*nsteps
        assert digest(folder/f"segment_{row['epoch']:02d}.pt")==row['segment_sha256']
    if 'arm' in r and r['arm'] in ('sam24','erm48'):
        with np.load(folder/'source_orders.npz') as orders:
            assert len(orders.files)==c['epochs']
            for row in r['history'][1:]:
                ids=orders[f"epoch_{row['epoch']:02d}"]
                np.testing.assert_array_equal(np.sort(ids),roles['train'])
                assert array_hash(ids)==row['source_ids_sha256']
    with np.load(folder/'predictions.npz') as pred:
        np.testing.assert_array_equal(pred['ids'],roles['diagnostic'])
        np.testing.assert_array_equal(pred['validation_ids'],roles['val'])
        np.testing.assert_array_equal(pred['y'],y[roles['diagnostic']])
        for kind in ['erm','tail']:
            state=s['states'][kind]
            for role,key in [('val',kind+'_validation'),('diagnostic',kind)]:
                outputs=[]
                for ids in np.array_split(roles[role],np.arange(c['batch'],len(roles[role]),c['batch'])):
                    raw=np.asarray(cache[ids],dtype=np.float32)
                    x=torch.from_numpy(np.stack([raw[:,:,0],raw[:,:,2],np.zeros_like(raw[:,:,0]),np.zeros_like(raw[:,:,0])],axis=2))
                    outputs.append(functional_forward(x,state,'shared128'))
                np.testing.assert_array_equal(torch.cat(outputs).numpy(),pred[key])
            assert metrics(y[roles['diagnostic']],pred[kind].argmax(1))==r['metrics'][kind]
    return {'held':h,'arm':r.get('arm','shared128'),'functional_logits_bitwise_equal':True,
            'source_roles_order_selection_verified':True,'artifact_hashes':len(r['files'])+c['epochs']}
