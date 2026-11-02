"""Fixed-percentile RSC adaptation; known method, not a novel contribution.

Independent mathematical implementation of DomainBed's RSC rule. Auxiliary
head runs without dropout on detached encoder features. Only final CE updates
encoder/head. Random control keeps RSC-derived sample selection and per-row
mask cardinality, randomizing only removed coordinates in its own model state.
"""
import torch
from torch.nn import functional as F


def auxiliary(head,z):
    for layer in head:
        if not isinstance(layer,torch.nn.Dropout):z=layer(z)
    return z


def select_mask(features,head,y,feature_drop=1/3,batch_drop=1/3,random_coordinates=False,mask_seed=0):
    if not (0<feature_drop<1 and 0<batch_drop<1):raise ValueError('Drop fractions must be inside(0,1)')
    f=features.detach().requires_grad_(True);logits=auxiliary(head,f);chosen=logits.gather(1,y[:,None]).sum();g=torch.autograd.grad(chosen,f)[0].detach()
    threshold=torch.quantile(g.double(),1-feature_drop,dim=1).to(g.dtype)
    keep=g<threshold[:,None]
    with torch.no_grad():
        muted=auxiliary(head,f*keep);changes=logits.softmax(1).gather(1,y[:,None]).squeeze(1)-muted.softmax(1).gather(1,y[:,None]).squeeze(1)
        cutoff=torch.quantile(changes.double(),1-batch_drop).to(changes.dtype);unmasked=changes<cutoff
        if random_coordinates:
            generator=torch.Generator(device='cpu').manual_seed(mask_seed)
            rank=torch.rand(g.shape,generator=generator).argsort(1).argsort(1)
            keep=rank>=(~keep).sum(1)[:,None]
        mask=keep|unmasked[:,None]
    if not torch.isfinite(g).all() or not torch.isfinite(changes).all():raise ValueError('Nonfinite masking statistics')
    return mask.detach(),dict(selected_samples=int((~unmasked).sum()),removed_coordinates=int((~mask).sum()),coordinates=int(mask.numel()),aux_probability_drop=float(changes.mean()))


def source_step(net,optimizer,x,y,arm='rsc24',feature_drop=1/3,batch_drop=1/3,mask_seed=0):
    if not net.training:raise ValueError('Expected train mode')
    optimizer.zero_grad(set_to_none=True)
    if arm=='shared128':
        loss=F.cross_entropy(net(x),y);stats={'selected_samples':0,'removed_coordinates':0,'coordinates':len(y)*1728,'aux_probability_drop':0.}
    else:
        if arm not in ['rsc24','random24']:raise ValueError('Unknown arm')
        n,v,c,t,r=x.shape;z=net.encoder(x.reshape(n*v,c,t,r)).reshape(n,-1)
        mask,stats=select_mask(z,net.head,y,feature_drop,batch_drop,arm=='random24',mask_seed)
        loss=F.cross_entropy(net.head(z*mask),y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite source loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('Invalid model gradients')
    optimizer.step();return float(loss.detach()),stats
