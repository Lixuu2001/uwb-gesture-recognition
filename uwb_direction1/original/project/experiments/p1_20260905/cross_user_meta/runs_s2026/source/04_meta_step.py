"""A0-A3 objectives with an exact differentiable virtual SGD step.

The caller owns outer optimization. This module never calls backward/step and
never writes original parameters or their .grad. Standard dropout is retained;
the caller must replay RNG state when comparing objectives across arms.
"""
import math

import torch
from torch.func import functional_call


def differentiable_update(params, loss, alpha):
    """Return named fast weights theta-alpha*d(loss)/d(theta), graph intact.

    params is a name->Tensor mapping. Unused or frozen parameters are returned
    unchanged, equivalent to a zero inner gradient. No optimizer state is used.
    """
    active = {name: value for name, value in params.items() if value.requires_grad}
    grads = torch.autograd.grad(loss, tuple(active.values()), create_graph=True,
                                allow_unused=True) if active else ()
    updates = {name: value - alpha * grad if grad is not None else value
               for (name, value), grad in zip(active.items(), grads)}
    return {name: updates.get(name, value) for name, value in params.items()}


def episode_loss(model, support_x, support_y, query_x, query_y, arm, criterion,
                 alpha=1e-4, mv_weight=.5):
    """Return (outer loss graph, detached scalar diagnostics).

    S=CE(R_support) [+ mv_weight*CE(MV_support) for A2/A3].
    A0/A2: S + CE(R_query; theta).
    A1/A3: S + CE(R_query; theta-alpha*grad_theta(S)), exact second order.
    Query inputs always pass through model.forward, the strict Right boundary.
    """
    if arm not in ('A0', 'A1', 'A2', 'A3'):
        raise ValueError('arm must be A0, A1, A2 or A3')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    if not math.isfinite(mv_weight) or mv_weight < 0:
        raise ValueError('mv_weight must be finite and nonnegative')
    multiview = arm in ('A2', 'A3')
    right, mv = model.support_outputs(support_x, 'MV' if multiview else 'R')
    support_r = criterion(right, support_y)
    support_mv = criterion(mv, support_y) if multiview else support_r.new_zeros(())
    support = support_r + mv_weight * support_mv
    if arm in ('A1', 'A3'):
        params = dict(model.named_parameters())
        fast = differentiable_update(params, support, alpha)
        # Clone buffers to protect original state if a future model gains buffers.
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        query_logits = functional_call(model, (fast, buffers), (query_x,), strict=True)
    else:
        query_logits = model(query_x)
    query = criterion(query_logits, query_y)
    return support + query, {'support_R': support_r.detach(),
                             'support_MV': support_mv.detach(), 'query': query.detach()}
