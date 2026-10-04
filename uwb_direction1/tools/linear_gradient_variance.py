"""Analytic softmax last-layer gradient-variance core; preparation, not Fishr.

For one CE example: dL/dW = (softmax(Wz+b)-one_hot(y)) outer z.
No EMA, annealing, optimizer, source sampling or experiment selection here.
Independent equation implementation; no third-party source copied.
"""
import torch
from torch.nn import functional as F


def gradient_rows(features, weight, bias, labels):
    if features.ndim != 2 or weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError('Expected features[N,D], weight[C,D], bias[C]')
    if features.shape[1] != weight.shape[1] or labels.shape != (len(features),):
        raise ValueError('Inconsistent dimensions')
    residual = F.linear(features, weight, bias).softmax(-1) - F.one_hot(labels, weight.shape[0]).to(features.dtype)
    return torch.cat((residual.unsqueeze(-1).mul(features.unsqueeze(1)).flatten(1), residual), dim=1)


def domain_variances(features, weight, bias, labels, domains):
    if domains.shape != labels.shape or len(labels) == 0:
        raise ValueError('Nonempty labels and matching domain IDs required')
    rows = gradient_rows(features, weight, bias, labels)
    values = []
    for domain in torch.unique(domains, sorted=True):
        group = rows[domains == domain]
        if len(group) < 2:
            raise ValueError('At least two examples per domain required')
        values.append((group - group.mean(0)).square().mean(0))
    return torch.stack(values)


def variance_matching_penalty(features, weight, bias, labels, domains):
    v = domain_variances(features, weight, bias, labels, domains)
    return (v - v.mean(0)).square().mean()
