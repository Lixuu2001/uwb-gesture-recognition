"""Reliability-gated single-view feature completion.

The visible view is always retained.  Each predicted missing-view feature is
multiplied by a learned scalar gate before the three-view readout.  A gate of
zero is therefore the same fallback as the fair zero-filled missing-view
baseline, while a gate near one trusts the completion network.
"""
import torch
from torch import nn

from reconstruction import reconstruction_loss


def compact_grad(enc, x):
    """Differentiable copy of the encoder's 10x21 token pooling."""
    _, t, s = enc(x)
    b, v, _, d = t.shape
    return torch.cat(
        [t.reshape(b, v, 10, 9, d).mean(3).flatten(2),
         s.reshape(b, v, 21, 9, d).mean(3).flatten(2)], -1
    ).float()


class GatedCompletion(nn.Module):
    """End-to-end single-view completion with two view-reliability gates."""

    def __init__(self, encoder, view, mode, mu, sd, dim=992, nclass=12):
        super().__init__()
        if view not in (0, 1, 2):
            raise ValueError("view must be 0, 1, or 2")
        if mode not in ("gate_capacity", "gate_joint", "joint", "capacity"):
            raise ValueError(f"unknown gated mode: {mode}")
        self.encoder = encoder
        self.view = view
        self.mode = mode
        self.dim = dim
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 2 * dim)
        )
        # Keep the first layer exposed as gate[0] for the gradient audit.
        self.gate = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 2)
        )
        self.head = nn.Sequential(
            nn.Linear(3 * dim, 256), nn.GELU(), nn.Dropout(.1),
            nn.Linear(256, nclass)
        )

    def forward(self, raw):
        if raw.ndim != 4 or raw.shape[1:] != (1, 90, 189):
            raise ValueError("Only target-view raw input is allowed")
        z = compact_grad(self.encoder, raw)[:, 0]
        z = (z - self.mu[:, self.view]) / self.sd[:, self.view]
        pred = self.decoder(z).reshape(-1, 2, self.dim)
        gates = torch.sigmoid(self.gate(z)).reshape(-1, 2, 1)
        missing = pred * gates
        rows = []
        j = 0
        for v in range(3):
            if v == self.view:
                rows.append(z)
            else:
                rows.append(missing[:, j])
                j += 1
        parts = torch.stack(rows, 1)
        return self.head(parts.flatten(1)), parts, gates


__all__ = ["GatedCompletion", "compact_grad", "reconstruction_loss"]
