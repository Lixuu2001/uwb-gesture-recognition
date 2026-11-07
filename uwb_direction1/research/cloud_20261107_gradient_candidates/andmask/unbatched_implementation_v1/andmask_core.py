"""Independent AND-mask arithmetic: source sign agreement with native sample-weighted mean."""
import torch

def combine(domain_gradients, counts, tau=1.0):
    assert len(domain_gradients)==len(counts)>0 and all(n>0 for n in counts)
    assert 0 <= tau <= 1 and len({len(g) for g in domain_gradients})==1
    total=sum(counts);combined=[];densities=[];kept=[]
    for columns in zip(*domain_gradients):
        assert all(torch.isfinite(g).all() for g in columns)
        values=torch.stack(columns)
        mask=(values.sign().mean(0).abs()>=tau).to(values.dtype)
        density=mask.mean()
        average=sum(g*(n/total) for g,n in zip(columns,counts))
        update=(mask*average)/(1e-10+density)
        assert torch.isfinite(update).all()
        combined.append(update);densities.append(float(density));kept.append(int(mask.sum()))
    return combined,dict(domain_counts=list(counts),tensor_mask_densities=densities,tensor_kept_coordinates=kept)

def gradients_from_pooled_logits(logits, labels, domains, parameters, tau=1.0):
    """One shared native BN/dropout forward, gradients for each present source user's BCE."""
    params=list(parameters);labels=torch.as_tensor(labels,dtype=torch.long)
    domains=torch.as_tensor(domains,dtype=torch.long)
    assert logits.shape==(len(labels),12) and labels.shape==domains.shape
    target=torch.nn.functional.one_hot(labels,12).to(logits.dtype)
    per_example=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1)
    unique=sorted(domains.unique().tolist());counts=[];gradients=[]
    for i,domain in enumerate(unique):
        take=domains==domain;counts.append(int(take.sum()))
        gradients.append(torch.autograd.grad(per_example[take].mean(),params,retain_graph=i+1<len(unique)))
    update,diagnostic=combine(gradients,counts,tau)
    diagnostic['present_domains']=unique
    global_bce=torch.nn.functional.binary_cross_entropy_with_logits(logits.detach(),target)
    return global_bce,update,diagnostic
