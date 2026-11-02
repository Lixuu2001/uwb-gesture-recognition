"""Losses: closed-set CE, evidential (EDL) + TMC fusion loss, optional SupCon."""
import torch
import torch.nn.functional as F


def kl_dirichlet_to_uniform(alpha, K):
    """KL( Dir(alpha) || Dir(1...1) ).  alpha: (B,K) -> (B,)"""
    beta = torch.ones_like(alpha)
    S_a = alpha.sum(-1, keepdim=True)
    S_b = beta.sum(-1, keepdim=True)
    lnB_a = torch.lgamma(S_a) - torch.lgamma(alpha).sum(-1, keepdim=True)
    lnB_b = torch.lgamma(beta).sum(-1, keepdim=True) - torch.lgamma(S_b)
    dg = torch.digamma(alpha) - torch.digamma(S_a)
    kl = lnB_a + lnB_b + ((alpha - beta) * dg).sum(-1, keepdim=True)
    return kl.squeeze(-1)


def edl_loss(alpha, y_onehot, K, anneal=1.0):
    """Sensoy 2018 Type-II MLE (Bayes risk of cross-entropy) + KL on the
    misleading evidence.  alpha: (B,K), y_onehot: (B,K)."""
    S = alpha.sum(-1, keepdim=True)
    ce = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)   # (B,)
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha                     # remove correct evidence
    kl = kl_dirichlet_to_uniform(alpha_tilde, K)
    return (ce + anneal * kl).mean()


def tmc_loss(out, y, K, anneal=1.0):
    """EDL on every view + on the DS-combined opinion (Han et al. 2021)."""
    y1h = F.one_hot(y, K).float()
    loss = edl_loss(out["alpha_c"], y1h, K, anneal)
    V = out["alpha_v"].size(1)
    for v in range(V):
        loss = loss + edl_loss(out["alpha_v"][:, v], y1h, K, anneal)
    return loss / (V + 1)


def ood_mixup_loss(evid_head, f_views, labels, n_classes,
                   lam_low=0.4, lam_high=0.6):
    """Manifold-mixup synthetic negatives (Eom Eq.12).

    Mix the features of two DIFFERENT known classes and force the evidential
    head to output maximum uncertainty (flat Dirichlet) on the mixture. This
    gives the OSR head explicit 'unfamiliar' examples at training time, which is
    what makes distance/uncertainty thresholds actually separate known from
    near-OOD unknown gestures. f_views: (B, V, d)."""
    B, V, d = f_views.shape
    device = f_views.device
    perm = torch.randperm(B, device=device)
    diff = labels != labels[perm]
    if diff.sum() == 0:
        return f_views.new_zeros(())
    lam = torch.empty(B, 1, 1, device=device).uniform_(lam_low, lam_high)
    mixed = (lam * f_views + (1.0 - lam) * f_views[perm])[diff].reshape(-1, d)
    alpha = evid_head(mixed) + 1.0
    return kl_dirichlet_to_uniform(alpha, n_classes).mean()


def pos_jitter_loss(evid_head, f_views, labels, n_classes, sigma=0.1):
    """Gaussian-jitter synthetic positives (Eom Eq.13): EDL on lightly perturbed
    real features so the positive (known) region is robust. f_views: (B,V,d)."""
    B, V, d = f_views.shape
    alpha = evid_head(f_views.reshape(B * V, d)
                      + sigma * torch.randn(B * V, d, device=f_views.device)) + 1.0
    y1h = F.one_hot(labels.repeat_interleave(V), n_classes).float()
    return edl_loss(alpha, y1h, n_classes, anneal=0.0)


def supcon_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss over a flat set of feature vectors.
    features: (N,d), labels: (N,). Pulls same-class (incl. sibling views)
    together -> view-invariant shared space."""
    device = features.device
    f = F.normalize(features, dim=-1)
    sim = f @ f.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = 1.0 - torch.eye(f.size(0), device=device)
    exp = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = (labels[:, None] == labels[None, :]).float() * self_mask
    pos_count = pos.sum(1).clamp(min=1.0)
    return -((pos * log_prob).sum(1) / pos_count).mean()
