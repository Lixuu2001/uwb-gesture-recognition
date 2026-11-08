"""Independent float64 NumPy BCE/q/per-logit derivative; no core call/autograd loss."""
import numpy as np
import torch
def reference(logits,labels,domains,domain_users,q_previous,eta=.01):
    z=logits.detach().cpu().numpy().astype(np.float64);y=np.asarray(labels,dtype=int);u=np.asarray(domains,dtype=int)
    groups=tuple(map(int,domain_users));assert len(groups)==4 and set(u)<=set(groups) and eta==.01
    hot=np.eye(12,dtype=np.float64)[y];point=(np.logaddexp(0,z)-hot*z).mean(1)
    counts=np.array([(u==g).sum() for g in groups]);present=counts>0
    risks=np.array([point[u==g].mean() if n else 0. for g,n in zip(groups,counts)])
    q=np.asarray(q_previous,dtype=np.float64).copy();q[present]*=np.exp(eta*risks[present]);q/=q.sum()
    weights=q.copy()
    if not present.all():weights[present]/=weights[present].sum()
    gradient=(1/(1+np.exp(-z))-hot)/12
    for j,g in enumerate(groups):
        if counts[j]:gradient[u==g]*=weights[j]/counts[j]
    return torch.as_tensor(gradient,dtype=logits.dtype),float(np.dot(weights[present],risks[present])),torch.as_tensor(q,dtype=logits.dtype),counts.tolist()
