"""Base classification objective, identical to the matched global-smoothing arm."""
import torch
from torch.nn import functional as F

import protocol as P


def classification_loss(logits, labels, matrix, temperature=2.0, soft_weight=0.5):
    """CE plus the temperature KL toward the frozen global soft target.

    matrix is the fold-matched global projection U; the target is
    0.5*(onehot + onehot@U), exactly the objective that produced the 74.77 pp
    single-radar control, so the new base arm is that control and not a guess.
    """
    z = logits.float()
    onehot = F.one_hot(labels, num_classes=P.N_CLASSES).float()
    target = 0.5 * (onehot + onehot @ matrix.detach().float())
    soft = F.kl_div(F.log_softmax(z / temperature, dim=-1), target,
                    reduction='batchmean') * temperature ** 2
    return F.cross_entropy(z, labels) + soft_weight * soft


def true_mass(matrix):
    """Probability the soft target puts on the true class, for the ledger."""
    diagonal = torch.diagonal(torch.as_tensor(matrix))
    return float((1 + diagonal[0]) / 2)
