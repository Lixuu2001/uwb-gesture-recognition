"""Published multiplicative GroupDRO component, fixed native BCE/pooled-batch transfer.

DomainBed b93c22a1..., MIT: q[g] *= exp(eta * detached source risk), normalize,
then dot(q, risks). No original-algorithm claim. Missing native-batch groups use
the prospectively declared conditional-present-risk rule, not extra sampling.
"""
import copy
import torch
import torch.nn.functional as F

class GroupDROState:
    def __init__(self,domain_users,eta=.01,dtype=torch.float32):
        self.domains=tuple(map(int,domain_users));assert len(self.domains)==4 and self.domains==tuple(sorted(set(self.domains)))
        assert eta==.01;self.eta=eta;self.q=torch.ones(4,dtype=dtype);self.updates=0
    def objective(self,logits,labels,domains):
        assert logits.ndim==2 and logits.shape[1]==12 and len(logits)>0
        labels=torch.as_tensor(labels,dtype=torch.long,device=logits.device);domains=torch.as_tensor(domains,dtype=torch.long,device=logits.device)
        assert labels.shape==domains.shape==(len(logits),) and set(domains.tolist())<=set(self.domains)
        assert self.q.dtype==logits.dtype and self.q.device==logits.device
        target=F.one_hot(labels,12).to(logits.dtype)
        per_record=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1)
        counts=[int((domains==u).sum()) for u in self.domains]
        risks=torch.stack([per_record[domains==u].mean() if n else logits.new_zeros(()) for u,n in zip(self.domains,counts)])
        present=torch.as_tensor([n>0 for n in counts],dtype=torch.bool,device=logits.device)
        assert torch.isfinite(risks).all() and torch.isfinite(self.q).all() and (self.q>=0).all()
        with torch.no_grad():
            q=self.q.clone();q[present]=q[present]*torch.exp(self.eta*risks[present].detach());q=q/q.sum()
            assert torch.isfinite(q).all() and (q>=0).all() and q[present].sum()>0
            self.q.copy_(q);self.updates+=1
        weights=self.q.detach().clone()
        loss=torch.dot(weights,risks) if all(counts) else torch.dot(weights[present]/weights[present].sum(),risks[present])
        global_bce=F.binary_cross_entropy_with_logits(logits,target)
        assert torch.isfinite(loss) and torch.isfinite(global_bce)
        return loss,global_bce,dict(domain_users=self.domains,domain_counts=counts,source_risks=risks.detach().clone(),q=self.q.clone(),present=present)
    def state_dict(self):return dict(domain_users=self.domains,eta=self.eta,q=self.q.clone(),updates=self.updates)
    def load_state_dict(self,state):
        assert tuple(state['domain_users'])==self.domains and state['eta']==self.eta
        assert isinstance(state['updates'],int) and state['updates']>=0 and state['q'].shape==(4,) and state['q'].dtype==self.q.dtype
        q=state['q'].detach().clone();assert torch.isfinite(q).all() and (q>=0).all() and q.sum()>0
        if state['updates']==0:assert torch.equal(q,torch.ones_like(q))
        else:assert torch.allclose(q.sum(),q.new_tensor(1.),rtol=0,atol=1e-6)
        self.q=q;self.updates=state['updates']

class NativeGroupDROOptimizer:
    """Native AdamWLookahead plus explicit q/update serialization; no scheduled reset."""
    def __init__(self,parameters,domain_users,native_optimizer_factory):
        self.params=list(parameters);self.factory=native_optimizer_factory;self.native=self.factory(self.params);self.robust=GroupDROState(domain_users)
    def zero_grad(self):self.native.zero_grad()
    def step(self):
        assert self.robust.updates==self.native.count+1
        self.native.step();assert self.robust.updates==self.native.count
    def state_dict(self):return dict(native_optimizer=self.native.state_dict(),groupDRO=self.robust.state_dict())
    def load_state_dict(self,state):
        self.native=self.factory(self.params);self.native.load_state_dict(copy.deepcopy(state['native_optimizer']));self.robust.load_state_dict(copy.deepcopy(state['groupDRO']));assert self.native.count==self.robust.updates
