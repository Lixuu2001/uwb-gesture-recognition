"""Functional inference and independent split/selection/order checks."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audit_restored import digest, metrics
from full_view_heads import array_hash


from audit_source_sam import functional_forward, audit_run as original_audit
from source_fishr import balanced_order

def audit_run(folder, cache, y, users, c):
    r=json.loads((folder/'result.json').read_text());h=r['held_source_user']
    if r.get('arm','shared128')=='shared128':return original_audit(folder,cache,y,users,c)
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
    nsteps=len(balanced_order(roles['train'],users,c,h,1))//c['batch']
    assert p['ema']['steps']==p['step'] and p['ema']['scale'] is not None
    assert set(p['ema']['values'])=={'real','permuted'}
    assert p['ema']['calibration']['step']==nsteps*c['warmup_epochs']+1
    assert s['selection']['tail_steps']==nsteps*(c['epochs']-c['tail_start_epoch']+1)
    for row in r['history'][1:]:
        order=balanced_order(roles['train'],users,c,h,row['epoch'])
        assert array_hash(order)==row['source_ids_sha256']
        assert row['step']==row['epoch']*nsteps
        assert digest(folder/f"segment_{row['epoch']:02d}.pt")==row['segment_sha256']
    if 'arm' in r and r['arm'] in c['arms']:
        with np.load(folder/'source_orders.npz') as orders:
            assert len(orders.files)==c['epochs']
            for row in r['history'][1:]:
                ids=orders[f"epoch_{row['epoch']:02d}"]
                np.testing.assert_array_equal(ids,balanced_order(roles['train'],users,c,h,row['epoch']))
                assert set(ids)==set(roles['train'])
                for batch in ids.reshape(-1,c['batch']):
                    counts=np.unique(users[batch],return_counts=True)[1];assert len(counts)==4 and np.all(counts==c['per_domain'])
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
