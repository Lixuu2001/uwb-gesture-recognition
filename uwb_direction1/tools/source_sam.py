"""CPU-only supervised SAM step with one ordinary AdamW state to checkpoint.

Algorithm: Foret et al., ICLR2021, https://github.com/google-research/sam.
Independent PyTorch implementation; CE-only perturbation and decoupled AdamW
are declared departures from the author's coupled-L2 JAX experiment.
Same CPU stochastic mask in the two passes; RNG advances once per update.
No running-stat buffers permitted. Parameters are restored before base update.
"""
import math
import torch
from torch.nn import functional as F


def source_step(net, optimizer, x, y, rho):
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    params=list(net.parameters())
    if any(p.device.type != 'cpu' for p in params) or list(net.buffers()):
        raise ValueError('This step supports CPU models without mutable buffers only')
    if not net.training:
        raise ValueError('Expected training mode')
    optimizer.zero_grad(set_to_none=True)
    rng=torch.get_rng_state()
    loss=F.cross_entropy(net(x),y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite first loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise ValueError('Missing or nonfinite first gradient')
    if rho == 0:
        optimizer.step()
        return float(loss.detach()),float(loss.detach())
    rng_after=torch.get_rng_state()
    norm=torch.linalg.vector_norm(torch.cat([p.grad.detach().flatten() for p in params]))
    if not torch.isfinite(norm):raise ValueError('Nonfinite norm')
    old=[p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            if norm > 0:
                for p in params:p.add_(p.grad*(rho/norm))
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        robust_loss=F.cross_entropy(net(x),y)
        if not torch.isfinite(robust_loss):raise ValueError('Nonfinite second loss')
        robust_loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('Missing or nonfinite second gradient')
    finally:
        with torch.no_grad():
            for p,w in zip(params,old):p.copy_(w)
        torch.set_rng_state(rng_after)
    optimizer.step()
    return float(loss.detach()),float(robust_loss.detach())
