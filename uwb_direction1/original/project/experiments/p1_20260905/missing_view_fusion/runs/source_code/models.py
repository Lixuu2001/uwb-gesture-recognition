"""Small residual adapters; view order is Left, Top, Right throughout.

These are bounded frozen-feature adaptations, not reproductions of the source
papers. Outputs are corrections to the original Right classifier's logits.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


MASKS = ((False, False, True), (True, False, True),
         (False, True, True), (True, True, True))


def _validate_features(features, dim):
    if features.ndim != 3 or features.shape[1:] != (3, dim):
        raise ValueError(f"Expected features [B,3,{dim}]")


def _validate_mask(mask):
    if len(mask) != 3:
        raise ValueError("A mask must have three entries in Left/Top/Right order")
    mask = tuple(bool(value) for value in mask)
    if not any(mask):
        raise ValueError("At least one view must be available")
    return mask


def _validate_right(right, dim):
    if right.ndim != 2 or right.shape[1] != dim:
        raise ValueError(f"Deployment accepts only Right features [B,{dim}]")


def _zero_head():
    head = nn.Linear(128, 12)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def _fusion_mlp(dim):
    return nn.Sequential(nn.Linear(3 * dim, 256), nn.GELU(),
                         nn.Linear(256, 128), nn.GELU())


class MissingFusion(nn.Module):
    """One fusion network evaluated on available views and missing markers."""

    def __init__(self, dim, learned_tokens=False):
        super().__init__()
        self.dim = dim
        # Zero creation consumes no random numbers, preserving fuser pairing.
        tokens = torch.zeros(3, dim)
        if learned_tokens:
            self.missing_tokens = nn.Parameter(tokens)
        else:
            self.register_buffer("missing_tokens", tokens)
        self.fusion = _fusion_mlp(dim)
        self.head = _zero_head()

    def forward(self, features, mask):
        _validate_features(features, self.dim)
        mask = _validate_mask(mask)
        slots = [features[:, view] if present else
                 self.missing_tokens[view].expand(features.shape[0], -1)
                 for view, present in enumerate(mask)]
        code = self.fusion(torch.cat(slots, dim=1))
        return self.head(code), code

    def deploy(self, right):
        _validate_right(right, self.dim)
        features = torch.stack((torch.zeros_like(right),
                                torch.zeros_like(right), right), dim=1)
        return self(features, MASKS[0])[0]


class SharedSpecific(nn.Module):
    """Shared/private feature decomposition with available-shared substitution."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.shared = nn.Linear(dim, 128)
        self.private = nn.ModuleList(nn.Linear(dim, 128) for _ in range(3))
        self.fusion = nn.Linear(256, 128)
        self.readout = nn.Sequential(nn.Linear(3 * 128, 128), nn.GELU())
        self.head = _zero_head()
        self.view_classifier = nn.Linear(128, 3)

    def forward(self, features, mask):
        _validate_features(features, self.dim)
        mask = _validate_mask(mask)
        shared = {view: self.shared(features[:, view])
                  for view, present in enumerate(mask) if present}
        substitute = torch.stack(list(shared.values()), dim=1).mean(dim=1)
        slots = []
        for view, present in enumerate(mask):
            if present:
                private = self.private[view](features[:, view])
                slots.append(self.fusion(torch.cat((shared[view], private), dim=1))
                             + shared[view])
            else:
                slots.append(substitute)
        code = self.readout(torch.cat(slots, dim=1))
        return self.head(code), code

    def auxiliary(self, features):
        """Train-only shared MSE and private view-ID CE, returned separately."""
        _validate_features(features, self.dim)
        shared = [self.shared(features[:, view]) for view in range(3)]
        align = sum(F.mse_loss(shared[a], shared[b])
                    for a, b in ((0, 1), (0, 2), (1, 2))) / 3
        private = torch.stack([self.private[v](features[:, v]) for v in range(3)], dim=1)
        view_labels = torch.arange(3, device=features.device).expand(features.shape[0], -1)
        view = F.cross_entropy(self.view_classifier(private).reshape(-1, 3),
                               view_labels.reshape(-1))
        return {"align": align, "view": view}

    def deploy(self, right):
        _validate_right(right, self.dim)
        features = torch.stack((torch.zeros_like(right),
                                torch.zeros_like(right), right), dim=1)
        return self(features, MASKS[0])[0]


class PrototypeCompletion(nn.Module):
    """Right-derived soft weights over training-only per-view prototype banks."""

    def __init__(self, dim, banks, trainable_bank=False):
        super().__init__()
        self.dim = dim
        if banks.shape != (3, 12, dim):
            raise ValueError(f"Expected prototype banks [3,12,{dim}]")
        banks = banks.detach().clone()
        if trainable_bank:
            self.banks = nn.Parameter(banks)
        else:
            self.register_buffer("banks", banks)
        self.weight_net = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 12))
        self.fusion = _fusion_mlp(dim)
        self.head = _zero_head()

    def forward(self, right):
        _validate_right(right, self.dim)
        weights = self.weight_net(right).softmax(dim=1)
        left, top = weights @ self.banks[0], weights @ self.banks[1]
        code = self.fusion(torch.cat((left, top, right), dim=1))
        return self.head(code), code

    def deploy(self, right):
        return self(right)[0]


def sigmoid_contrast(codes, labels, temperature=0.2):
    """Balanced supervised sigmoid pair loss; identity pairs never contribute.

    An absent positive or negative group contributes differentiable zero; the
    two group losses retain their fixed 1/2 weights in every minibatch.
    """
    if codes.ndim != 2 or labels.ndim != 1 or codes.shape[0] != labels.shape[0]:
        raise ValueError("Expected codes [N,H] and labels [N]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    codes = F.normalize(codes, p=2, dim=1)
    logits = codes @ codes.T / temperature
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=codes.device)
    same_class = labels[:, None] == labels[None, :]
    positive = logits[same_class & off_diagonal]
    negative = logits[~same_class & off_diagonal]
    zero = logits.sum() * 0
    pos_loss = F.softplus(-positive).mean() if positive.numel() else zero
    neg_loss = F.softplus(negative).mean() if negative.numel() else zero
    return (pos_loss + neg_loss) / 2
