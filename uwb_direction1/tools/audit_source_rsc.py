"""Independent functional inference, complete source order, mask and tail checks."""
import json
import numpy as np
import torch
from audit_source_sam import audit_run as generic_audit
from full_view_heads import array_hash

def audit_run(folder,cache,y,users,c):
    receipt=generic_audit(folder,cache,y,users,c)
    r=json.loads((folder/'result.json').read_text());arm=r.get('arm','shared128')
    if arm not in ['rsc24','random24']:return receipt
    h=r['held_source_user'];train=np.asarray(json.loads((folder/'split.json').read_text())['train']);batch=c['batch'];steps=0
    with np.load(folder/'source_orders.npz') as orders:
        assert len(orders.files)==c['epochs']
        for row in r['history'][1:]:
            order=np.random.default_rng(c['seed']+10000*h+100000*row['epoch']).permutation(train)
            np.testing.assert_array_equal(orders[f"epoch_{row['epoch']:02d}"],order)
            assert len(row['mask_batches'])==int(np.ceil(len(train)/batch))
            for k,m in enumerate(row['mask_batches']):
                ids=order[k*batch:(k+1)*batch]
                assert m['source_ids_sha256']==array_hash(ids) and m['samples']==len(ids)
                assert m['step']==steps and m['mask_seed']==c['mask_seed_base']+h*100000+steps
                assert 0<m['selected_samples']<=len(ids)
                assert 0<=m['removed_coordinates']<=m['selected_samples']*1728
                assert m['coordinates']==len(ids)*1728 and np.isfinite(m['aux_probability_drop'])
                steps+=1
    # Segment averages and dense global averages have different rounding paths.
    selected=torch.load(folder/'selected.pt',weights_only=False)['states']['tail']
    accum={k:torch.zeros_like(v,dtype=torch.float64) for k,v in selected.items()};count=0
    for epoch in range(c['tail_start_epoch'],c['epochs']+1):
        s=torch.load(folder/f'segment_{epoch:02d}.pt',weights_only=False)['average']['weights']
        n=int(s['n_averaged']);count+=n
        for k in accum:accum[k]+=s['module.'+k].double()*n
    maxerr=0.
    for k,v in selected.items():
        avg=accum[k]/count;maxerr=max(maxerr,float((v.double()-avg).abs().max()))
        torch.testing.assert_close(v.double(),avg,atol=2e-6,rtol=2e-5)
    receipt.update(mask_metadata_all_batches_verified=True,source_orders_all_epochs_verified=True,tail_reconstructed_from_epoch_means_max_abs=maxerr,tail_reconstruction_tolerance={'atol':2e-6,'rtol':2e-5})
    return receipt
