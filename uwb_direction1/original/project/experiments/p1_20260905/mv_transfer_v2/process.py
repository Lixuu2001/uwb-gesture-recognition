"""Method B prediction head loss over the 15 segments."""
import torch
from torch.nn import functional as F


def process_loss(logits15, activity, progress, seg_ok, sample_ok):
    """Activity KL and CDF distance on the valid segments of the valid samples.

    Samples whose clip carries no measurable motion contribute nothing here and
    are never dropped from the classification loss.
    """
    if not bool(sample_ok.any()):
        zero = logits15.sum() * 0
        return zero, zero
    z = logits15[sample_ok].float()
    mask = seg_ok[sample_ok]
    log_p = F.log_softmax(z.masked_fill(~mask, -1e4), dim=-1)
    p = log_p.exp()
    a = activity[sample_ok].float()
    c = progress[sample_ok].float()
    kl = (a * (a.clamp_min(1e-8).log() - log_p)).sum(-1).mean()
    cdf = F.smooth_l1_loss(p.cumsum(-1), c, reduction='none').mean(-1).mean()
    return kl, cdf
