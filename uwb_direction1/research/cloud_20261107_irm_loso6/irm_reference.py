"""Independent reference: explicit dummy-scale autograd, not the analytic production core."""
import torch

def bce(z,y):
    return torch.nn.functional.binary_cross_entropy_with_logits(z,torch.nn.functional.one_hot(y,12).to(z.dtype))

def objective(logits,labels,domains,weight):
    domains=torch.as_tensor(domains,dtype=torch.long);labels=torch.as_tensor(labels,dtype=torch.long)
    penalties=[]
    for u in sorted(domains.unique().tolist()):
        z=logits[domains==u];y=labels[domains==u]
        if len(y)<2:continue
        scale=torch.tensor(1.,device=z.device,dtype=z.dtype,requires_grad=True)
        even=bce(z[::2]*scale,y[::2]);odd=bce(z[1::2]*scale,y[1::2])
        ge=torch.autograd.grad(even,scale,create_graph=True)[0]
        go=torch.autograd.grad(odd,scale,create_graph=True)[0]
        penalties.append(ge*go)
    penalty=torch.stack(penalties).mean() if penalties else logits.sum()*0.
    return bce(logits,labels)+weight*penalty,penalty
