"""Matched source-only parameter-sharing intervention, not a novelty claim."""
import copy
import torch
from torch import nn
from circular_signal import SignalNet


class UntiedNet(nn.Module):
    def __init__(self, template):
        super().__init__()
        self.encoders = nn.ModuleList([copy.deepcopy(template.encoder) for _ in range(3)])
        self.head = copy.deepcopy(template.head)

    def forward(self, x):
        if tuple(x.shape[1:]) != (3,4,90,63):
            raise ValueError('Expected [N,3,4,90,63]')
        return self.head(torch.stack([e(x[:,v].contiguous()).flatten(1)
                                     for v,e in enumerate(self.encoders)],dim=1))


def make_model(arm, seed):
    torch.manual_seed(seed)
    base = SignalNet()
    if arm == 'shared128': return base
    if arm == 'untied128': return UntiedNet(base)
    if arm == 'shared177':
        # Encoder and post-initialization training RNG match the reference.
        # Wider head cannot have the same logits or Dropout mask dimensions.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+17)
            base.head = nn.Sequential(nn.Flatten(),nn.Linear(1728,177),nn.GELU(),
                                      nn.Dropout(.2),nn.Linear(177,12))
        return base
    raise ValueError(arm)
