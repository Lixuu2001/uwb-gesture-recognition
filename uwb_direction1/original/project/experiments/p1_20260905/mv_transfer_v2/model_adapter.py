"""Adapters around the frozen classifier plus the auxiliary heads of A and B.

The deployed network must stay byte-identical to the existing single-view
classifier, so nothing here may change how logits are produced. The adapter
only exposes intermediate tensors that the auxiliary losses need.
"""
import torch
from torch import nn

import protocol as P
from train import Classifier, digest_state          # noqa: F401  (re-exported)


def build_student(model_seed, aux_seed=P.AUXILIARY_SEED, relation=False, process=False,
                  relation_dim=128):
    """Backbone first with the model seed, aux heads afterwards with their own.

    Order matters: every arm of a comparison must get the same backbone and
    classifier initialisation, so the aux modules draw from a separate stream.
    """
    from train import setup
    setup(model_seed)
    model = Classifier([P.DEPLOY_VIEW], 'concat', 'mean').cuda()
    fingerprint = {'head': digest_state(model.head), 'encoder': digest_state(model.encoder)}
    torch.manual_seed(aux_seed)
    torch.cuda.manual_seed_all(aux_seed)
    heads = nn.ModuleDict()
    if relation:
        heads['relation'] = RelationProjector(relation_dim).cuda()
    if process:
        heads['process'] = ProcessHead().cuda()
    return model, heads, fingerprint


def classification_features(model, x):
    """Reproduce Classifier.forward exactly and also return the token tensors.

    The encoder's first output f carries view_emb; the deployed classifier
    ignores it and re-flattens the tokens, so the adapter does the same.
    """
    _, temporal, spatial = model.encoder(x)
    per_view = torch.cat([temporal.flatten(2), spatial.flatten(2)], dim=-1)
    merged = per_view.flatten(1) if model.readout == 'concat' else per_view.mean(1)
    return model.head(merged), merged, temporal, spatial


def student_aux_features(temporal, spatial):
    """B,64 summary of the single deployed view; both encoder paths contribute."""
    if temporal.shape[1] != 1 or spatial.shape[1] != 1:
        raise ValueError('student auxiliary features are single-view only')
    return torch.cat([temporal[:, 0].mean(1), spatial[:, 0].mean(1)], dim=-1)


def teacher_compact(temporal, spatial):
    """B,V,992 pooled teacher code, identical to reconstruction.compact."""
    b, v, t, d = temporal.shape
    if (t, spatial.shape[2], d) != (P.FRAMES, P.RANGE_BINS, 32):
        raise ValueError(f'unexpected teacher token shape {(t, spatial.shape[2], d)}')
    a = temporal.reshape(b, v, 10, 9, d).mean(3).flatten(2)
    c = spatial.reshape(b, v, 21, 9, d).mean(3).flatten(2)
    return torch.cat([a, c], dim=-1)


class RelationProjector(nn.Module):
    """64 -> 256 -> out, L2 normalised. Only the relation loss reads it."""

    def __init__(self, out_dim=128, in_dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, f):
        return torch.nn.functional.normalize(self.net(f).float(), dim=-1)


class ProcessHead(nn.Module):
    """Temporal tokens -> 15 segment activity logits (one head, no second task)."""

    def __init__(self, dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, temporal):                       # temporal: (B,1,90,32)
        if temporal.shape[1] != 1:
            raise ValueError('process head is single-view only')
        segments = temporal[:, 0].reshape(-1, P.SEGMENTS, P.FRAMES_PER_SEGMENT,
                                          temporal.shape[-1]).mean(2)
        return self.net(segments).squeeze(-1)


def deployment_state(model):
    """Exactly what may be exported: the classifier, nothing that reads L/T."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
