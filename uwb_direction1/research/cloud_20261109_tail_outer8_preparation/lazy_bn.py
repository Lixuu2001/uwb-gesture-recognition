"""Same source-only unweighted-BN-batch refresh as frozen native6 adaptation."""
import copy,random,math,time
import numpy as np
import torch
from core import semantic,order
def calibrate(m,data):
    ids=np.array(data.roles['sample_ids']['train'],dtype=np.int64)
    if not len(ids):raise ValueError('Empty training BN pass')
    users=data.roles['users']['train']
    if len(users)!=6 or len(set(users))!=6:raise ValueError('Outer8 requires six BN source users')
    mods=dict(m.named_modules());bn={k:v for k,v in mods.items() if isinstance(v,torch.nn.modules.batchnorm._BatchNorm)}
    modes={k:v.training for k,v in mods.items()};mom={k:v.momentum for k,v in bn.items()}
    params=semantic(dict(m.named_parameters()));buffers={k:v.clone() for k,v in m.named_buffers()}
    rng=torch.get_rng_state().clone();py=random.getstate();n=copy.deepcopy(np.random.get_state());start=time.monotonic()
    shuffled=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(21161030)).numpy()]
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(21161030);m.train()
            for v in bn.values():v.reset_running_stats();v.momentum=None
            for i in range(0,len(ids),64):m(torch.from_numpy(data.batch('train',shuffled[i:i+64])))
    finally:
        for k,v in bn.items():v.momentum=mom[k]
        for k,v in mods.items():v.training=modes[k]
    assert torch.equal(rng,torch.get_rng_state()) and py==random.getstate()
    now=np.random.get_state();assert now[0]==n[0] and np.array_equal(now[1],n[1]) and now[2:]==n[2:]
    assert semantic(dict(m.named_parameters()))==params
    permitted={k+'.'+s for k in bn for s in ['running_mean','running_var','num_batches_tracked']}
    assert {k for k,v in m.named_buffers() if not torch.equal(v,buffers[k])}<=permitted
    assert all(int(v.num_batches_tracked)==math.ceil(len(ids)/64) for v in bn.values())
    return dict(wall_seconds=time.monotonic()-start,train_samples=len(ids),train_users=users,source_IDs_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(),seed=21161030,batch=64,BN_batch_weighting='equal batch momentum None, same as existing native6 adaptation; not claimed identical original SWA sample-weighted BN',only_BN_buffers_changed=True,RNG_modes_momenta_restored=True,labels_read=False)
