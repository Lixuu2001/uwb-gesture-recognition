"""Prototype: all native MLP classifier BCE per-example gradients and published Fishr EMA.

No data loader, native backbone worker, training launch or model selection lives here.
Native BCE substitutes the author's CE explicitly. Missing native random-batch source
groups retain their EMA and are excluded from the current penalty (declared transfer).
"""
from collections import OrderedDict
import copy
import torch
from torch import nn
from torch.nn import functional as F

DEFAULTS = dict(penalty_lambda=1000.0, ema=0.95, anneal_completed_updates=1500,
                warmup_penalty=0.0, reset_entire_native_optimizer_at=1500)

def native_head_gradients(head, features, labels):
    """One native dropout draw; analytic gradients of per-example mean-12 BCE.

    Return all four classifier parameter gradients with their graph, so a Fishr
    objective can differentiate through them into classifier and backbone.
    """
    assert isinstance(head, nn.Sequential) and len(head) == 4
    assert isinstance(head[0], nn.Linear) and isinstance(head[1], nn.ReLU)
    assert isinstance(head[2], nn.Dropout) and isinstance(head[3], nn.Linear)
    assert (head[0].in_features, head[0].out_features, head[3].out_features) == (128,128,12)
    assert head[3].in_features == 128 and head[2].p == 0.2
    assert features.ndim == 2 and features.shape[1] == 128 and len(features) == len(labels) > 0
    hidden = head[0](features); activated = head[1](hidden); dropped = head[2](activated)
    logits = head[3](dropped)
    target = F.one_hot(labels,12).to(logits.dtype)
    delta = (logits.sigmoid()-target)/12
    # An inactive ReLU has zero derivative regardless of its unobserved dropout
    # draw. On active entries the nonzero output identifies the native mask.
    if head[2].training:
        mask = (dropped != 0).to(logits.dtype)/(1-head[2].p)
    else:
        mask = torch.ones_like(dropped)
    hidden_delta = (delta @ head[3].weight)*mask*(activated>0).to(logits.dtype)
    grads = OrderedDict([
        ('0.weight',(hidden_delta.unsqueeze(2)*features.unsqueeze(1)).flatten(1)),
        ('0.bias',hidden_delta),
        ('3.weight',(delta.unsqueeze(2)*dropped.unsqueeze(1)).flatten(1)),
        ('3.bias',delta),
    ])
    assert sum(g.shape[1] for g in grads.values()) == 18060
    return logits, grads

class PublishedEMA:
    """Detached history; author fixed 1/(1-ema), not time-bias correction."""
    def __init__(self, groups, ema=0.95):
        assert groups > 0 and 0 <= ema < 1
        self.ema=float(ema); self.data=[{} for _ in range(groups)]; self.updates=[0]*groups

    def update(self, group, variances):
        assert 0 <= group < len(self.data) and variances
        result=OrderedDict()
        for name,value in variances.items():
            old=self.data[group].get(name,torch.zeros_like(value))
            current=self.ema*old+(1-self.ema)*value
            self.data[group][name]=current.detach().clone()
            result[name]=current/(1-self.ema)
        self.updates[group]+=1
        return result

    def state_dict(self):
        return copy.deepcopy(dict(ema=self.ema,data=self.data,updates=self.updates))

    def load_state_dict(self,state):
        assert state['ema']==self.ema and len(state['data'])==len(self.data)
        assert len(state['updates'])==len(self.data)
        self.data=copy.deepcopy(state['data']); self.updates=list(state['updates'])
        assert all(not v.requires_grad for d in self.data for v in d.values())

def variance_penalty(grads,group_ids,ema):
    assert grads and group_ids.ndim==1 and len(group_ids)>0
    assert all(len(g)==len(group_ids) for g in grads.values())
    groups=sorted(group_ids.unique().tolist())
    assert all(0 <= group < len(ema.data) for group in groups)
    moments=[]
    for group in groups:
        current=OrderedDict()
        for name,g in grads.items():
            samples=g[group_ids==group]
            current[name]=(samples-samples.mean(0,keepdim=True)).pow(2).mean(0)
        moments.append(ema.update(group,current))
    mean=OrderedDict((name,torch.stack([d[name] for d in moments]).mean(0)) for name in grads)
    keys=sorted(mean)
    distances=[torch.cat([(d[k]-mean[k]).reshape(-1) for k in keys]).pow(2).mean() for d in moments]
    return torch.stack(distances).mean()

def schedule(completed_updates):
    assert completed_updates >= 0
    return dict(weight=0.0 if completed_updates<1500 else 1000.0,
                reset_before_this_update=completed_updates==1500)
