"""IRM dummy-scale gradient product for native twelve-output one-hot BCE."""
import torch

def global_bce(logits,labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.nn.functional.one_hot(labels,12).to(logits.dtype))

def scale_gradient(logits,labels):
    """Exact analytic d/ds BCE(s*z,y) at scalar s=1, retaining the full logits graph."""
    target=torch.nn.functional.one_hot(labels,12).to(logits.dtype)
    return ((logits.sigmoid()-target)*logits).mean()

def objective(logits,labels,domains,weight):
    labels=torch.as_tensor(labels,dtype=torch.long);domains=torch.as_tensor(domains,dtype=torch.long)
    assert logits.shape==(len(labels),12) and labels.shape==domains.shape and len(labels)>0
    penalties=[];valid=[];omitted=[];counts={}
    for u in sorted(domains.unique().tolist()):
        take=domains==u;z=logits[take];y=labels[take];counts[str(u)]=len(y)
        if len(y)<2:omitted.append(u);continue
        penalties.append(scale_gradient(z[::2],y[::2])*scale_gradient(z[1::2],y[1::2]));valid.append(u)
    penalty=torch.stack(penalties).mean() if penalties else logits.sum()*0.
    classification=global_bce(logits,labels)
    return classification+weight*penalty,classification,penalty,dict(valid_training_domains=valid,omitted_penalty_domains=omitted,domain_counts=counts)
