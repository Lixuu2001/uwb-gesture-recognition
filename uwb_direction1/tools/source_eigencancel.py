"""Source-only energy-selectivity veto for known empty-echo eigencancellation."""
import math
import numpy as np
import torch
from circular_signal import encode,get_batch
from empty_subspace import moments,projectors,fractions,evaluate


def learn(raw,y,u,tr,held,c):
    cfg=dict(c['diagnostic'],seed=c['seed']+1000*held);users=sorted(set(u[tr]));empty={int(h):tr[(u[tr]==h)&(y[tr]==11)] for h in users};covs={h:moments(raw,ids,c['batch_size']) for h,ids in empty.items()};rows=[];ids_by_user={}
    for h in users:
        fit=[k for k in users if k!=h];temp=projectors(np.mean([covs[k] for k in fit],0),int(h),cfg)
        ids=tr[u[tr]==h];ids=np.concatenate([ids[y[ids]==k][:cfg['examples_per_class']] for k in np.unique(y[ids])]);ids_by_user[str(int(h))]=ids.tolist()
        a,b=fractions(raw,ids,temp,c['batch_size']);rr=evaluate(y[ids],a,b,temp,cfg);rows.extend([dict(source_user=int(h),**r) for r in rr])
    counts=np.array([sum(r['passed'] for r in rows if r['view']==v) for v in range(3)]);selected=counts>=math.ceil(2*len(users)/3)
    cov=np.mean([covs[h] for h in users],0);model=projectors(cov,0,cfg);model.update(cov=cov,selected_views=selected,fit_ids=np.concatenate([empty[h] for h in users]),training_users=np.array(users))
    return model,dict(rows=rows,successful_users=counts.tolist(),minimum_successful_users=math.ceil(2*len(users)/3),selected_views=selected.tolist(),inner_diagnostic_ids=ids_by_user)


def filtered_raw(raw,model,random=False,selected_only=False):
    x=np.asarray(raw,dtype=np.float64).copy()
    for v in range(3):
        if selected_only and not model['selected_views'][v]:continue
        b=model[f'v{v}_'+('random_basis' if random else 'basis')];x[:,v]-=(x[:,v]@b)@b.T
    return x.astype(np.float32)


@torch.no_grad()
def extract(net,raw,cache,ids,model,batch):
    # Shared-view encoder: compute three filtered views, selected random views,
    # and complementary original views once; then assemble each arm's head.
    out={k:[] for k in ['selective','all_views','random_selected']};mask=np.asarray(model['selected_views']);n_selected=int(mask.sum())
    for st in range(0,len(ids),batch):
        part=ids[st:st+batch];n=len(part);r=np.asarray(raw[part]);all_x=get_batch(encode(filtered_raw(r,model)),np.arange(n),'legacy');all_pool=net.encoder(all_x.reshape(-1,4,90,63)).reshape(n,3,-1)
        chosen=all_pool.clone();rand=all_pool.clone()
        if n_selected<3:
            original=get_batch(cache,part,'legacy')[:,~mask];pooled=net.encoder(original.reshape(-1,4,90,63)).reshape(n,3-n_selected,-1);chosen[:,~mask]=pooled;rand[:,~mask]=pooled
        if n_selected:
            rand_x=get_batch(encode(filtered_raw(r,model,random=True,selected_only=True)),np.arange(n),'legacy')[:,mask]
            rand[:,mask]=net.encoder(rand_x.reshape(-1,4,90,63)).reshape(n,n_selected,-1)
        for arm,pool in [('selective',chosen),('all_views',all_pool),('random_selected',rand)]:out[arm].append(net.head[:3](pool.flatten(1)).numpy())
    return {k:np.concatenate(v).astype(np.float64) for k,v in out.items()}
