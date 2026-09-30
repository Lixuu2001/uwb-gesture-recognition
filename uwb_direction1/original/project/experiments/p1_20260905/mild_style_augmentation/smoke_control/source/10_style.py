"""Fixed shallow statistics mixing, with no change to the deployed Right model.

The donor schedule is independent of model/sampler random generators. Only
temporary training-epoch hooks touch projected tokens; donor examples traverse
the current linear projections under no_grad, never the encoder or dropout.
"""
from contextlib import contextmanager
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def style_mix(x, donor, mask, share=.1):
    """Mix detached population statistics over tokens of [batch, tokens, dim]."""
    if not isinstance(x, torch.Tensor) or not isinstance(donor, torch.Tensor):
        raise ValueError('Source and donor must be floating-point tensors')
    if (x.ndim != 3 or donor.shape != x.shape or any(n == 0 for n in x.shape)
            or not x.is_floating_point() or not donor.is_floating_point()
            or donor.device != x.device):
        raise ValueError('Require aligned floating-point [batch, tokens, dim] tensors')
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != (x.shape[0],) or mask.device != x.device):
        raise ValueError('Require a boolean batch mask on the source device')
    if not np.isfinite(share) or not 0 <= share <= 1:
        raise ValueError('Statistics share must be finite and within [0, 1]')
    if share == 0 or not bool(mask.any()):
        return x
    # Compute only enabled rows, preserving every disabled value exactly.
    source = x[mask].float()
    with torch.no_grad():
        clean = source.detach()
        reference = donor[mask].detach().float()
        mu = clean.mean(1, keepdim=True)
        sigma = clean.var(1, unbiased=False, keepdim=True).sqrt().clamp_min(1e-6)
        donor_mu = reference.mean(1, keepdim=True)
        donor_sigma = reference.var(1, unbiased=False, keepdim=True).sqrt().clamp_min(1e-6)
    mixed = source + share * ((donor_sigma - sigma) * (source - mu) / sigma + donor_mu - mu)
    result = x.clone()
    result[mask] = mixed.to(dtype=x.dtype)
    return result


def schedule(train_ids, labels, subjects, seed, probability=.25):
    """Return donors/masks aligned to IDs, drawing in ascending source-ID order."""
    ids = np.asarray(train_ids)
    labels, subjects = np.asarray(labels), np.asarray(subjects)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Require nonempty, unique integer training IDs')
    if (labels.ndim != 1 or subjects.shape != labels.shape
            or (ids < 0).any() or (ids >= len(labels)).any()):
        raise ValueError('Require aligned metadata and in-range training IDs')
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probability must be finite and within [0, 1]')
    ascending = np.sort(ids).astype(np.int64)
    rng = np.random.default_rng(seed)
    masks = rng.random(len(ascending)) < probability
    donors = np.empty(len(ascending), dtype=np.int64)
    # The two draws are explicitly user-uniform then instance-uniform. Neither
    # metadata nor examples outside these training IDs enter the candidate pool.
    for k, row in enumerate(ascending):
        candidates = ascending[(labels[ascending] == labels[row]) &
                               (subjects[ascending] != subjects[row])]
        users = np.unique(subjects[candidates])
        if len(users) == 0:
            raise ValueError(f'No same-class other-user training donor for ID {row}')
        user = rng.choice(users)
        donors[k] = rng.choice(candidates[subjects[candidates] == user])
    order = np.searchsorted(ascending, ids)
    return donors[order], masks[order]


def sampler_seed(state, fold):
    """Derive the fixed augmentation seed without advancing sampler state."""
    value = json.dumps({'sampler': state['rng'], 'fold': int(fold),
                        'augmentation_seed': 20260909}, sort_keys=True)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


class StyleReader:
    """Guarded base reader plus removable, strictly training-only style hooks."""

    def __init__(self, base, labels, subjects, enabled=True):
        self.base = base
        self.partitions = base.partitions
        self.labels = np.asarray(labels)
        self.subjects = np.asarray(subjects)
        self.enabled = bool(enabled)
        self.trace = hashlib.sha256()
        self.counts = {'clean': 0, 'style': 0}
        self._donor_ids = None
        self._masks = None
        self._model = None
        self._pending = None

    def configure(self, seed):
        if self._model is not None:
            raise RuntimeError('Configure the schedule before entering a training epoch')
        ids = np.asarray(self.partitions['train'])
        donors, masks = schedule(ids, self.labels, self.subjects, seed)
        self._donor_ids = np.full(len(self.labels), -1, dtype=np.int64)
        self._masks = np.zeros(len(self.labels), dtype=bool)
        self._donor_ids[ids] = donors
        self._masks[ids] = masks
        self.trace = hashlib.sha256()
        self.counts = {'clean': 0, 'style': 0}
        self._pending = None

    def load_ids(self, ids, role, mode, device='cuda'):
        # The existing partition/view/input guards ALWAYS execute first.
        x = self.base.load_ids(ids, role, mode, device)
        self._pending = None
        if self._model is None or not self._model.training or role != 'train':
            return x
        if mode != 'R':
            raise ValueError('This experiment accepts Right only during training')
        if self._donor_ids is None:
            raise RuntimeError('Configure the schedule before training reads')
        ids = np.asarray(ids, dtype=np.int64)
        donors = self._donor_ids[ids]
        masks = self._masks[ids] if self.enabled else np.zeros(len(ids), dtype=bool)
        if self.enabled and masks.any():
            donor = self.base.load_ids(donors[masks], 'train', 'R', device)
            if (x.ndim != 4 or x.shape[1] != 1 or donor.shape != (int(masks.sum()), *x.shape[1:])):
                raise ValueError('Require aligned single-Right source and donor batches')
            self._pending = {'donor': donor.detach(),
                             'where': torch.as_tensor(np.flatnonzero(masks), device=x.device),
                             'shape': tuple(x.shape), 'seen': set()}
        triples = np.column_stack([ids, donors, masks]).astype(np.int64)
        self.trace.update(triples.tobytes())
        self.counts['style'] += int(masks.sum())
        self.counts['clean'] += int((~masks).sum())
        return x

    def _project_hook(self, branch):
        def hook(module, inputs, output):
            pending = self._pending
            if self._model is None or not self._model.training or pending is None:
                return output
            if branch in pending['seen']:
                raise RuntimeError('Projection ran twice for one training read')
            batch, _, tokens, features = pending['shape']
            expected_input = (batch, tokens, features) if branch == 'temporal' else (batch, features, tokens)
            if (len(inputs) != 1 or tuple(inputs[0].shape) != expected_input
                    or tuple(output.shape) != (*expected_input[:2], module.out_features)):
                raise ValueError('Projection input does not match the guarded training read')
            raw = pending['donor'][:, 0]
            if branch == 'spatial':
                raw = raw.transpose(1, 2)
            with torch.no_grad():
                # Functional projection prevents recursive module hooks and does
                # not run position embeddings, encoder blocks or random layers.
                donor = F.linear(raw, module.weight, module.bias)
            where = pending['where']
            source = output[where]
            mixed = style_mix(source, donor, torch.ones(len(where), dtype=torch.bool, device=output.device))
            result = output.clone()
            result[where] = mixed
            pending['seen'].add(branch)
            if len(pending['seen']) == 2:
                self._pending = None
            return result
        return hook

    @contextmanager
    def epoch_context(self, model):
        """Attach hooks for one epoch, removing hooks/tensors even after failure."""
        if self._model is not None:
            raise RuntimeError('Nested style epochs are not supported')
        if self._donor_ids is None:
            raise RuntimeError('Configure the schedule before entering an epoch')
        handles = []
        self._model = model
        try:
            if self.enabled:
                encoder = model.right.encoder
                for branch in ('temporal', 'spatial'):
                    projection = getattr(encoder, branch).proj
                    if not isinstance(projection, nn.Linear):
                        raise ValueError('Style hooks require the original linear projections')
                    handles.append(projection.register_forward_hook(self._project_hook(branch)))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self._pending = None
            self._model = None
