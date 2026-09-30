"""Trainable radar encoder with missing-view fusion and strict Right inference."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))


def _parent_module(name):
    """Avoid collisions with an e2e data/model module in the same process."""
    key = '_missing_view_fusion_parent_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, PARENT / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


_data = _parent_module('data')
_models = _parent_module('models')
MissingFusion = _models.MissingFusion
MASKS = _models.MASKS
pool_tokens = _data.pool_tokens


def state_hash(model):
    """Hash named tensor values including shape/dtype, independent of device."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class EndToEndFusion(nn.Module):
    """Classification through a trainable encoder; no original-head addition."""

    def __init__(self, encoder, fuser, mu, sd):
        super().__init__()
        mu = torch.as_tensor(mu).detach().clone().float()
        sd = torch.as_tensor(sd).detach().clone().float()
        if mu.ndim != 3 or mu.shape[:2] != (1, 3) or mu.shape != sd.shape:
            raise ValueError('Expected normalization buffers mu/sd [1,3,D]')
        if not torch.isfinite(mu).all() or not torch.isfinite(sd).all() or not (sd > 0).all():
            raise ValueError('Normalization must be finite with strictly positive sd')
        self.encoder = encoder.requires_grad_(True)
        self.fuser = fuser.requires_grad_(True)
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)

    @staticmethod
    def _validate_input(x, views):
        if x.ndim != 4 or x.shape[1:] != (views, 90, 378):
            raise ValueError(f'Expected radar input [B,{views},90,378]')

    def _features(self, x):
        _, temporal, spatial = self.encoder(x)
        return pool_tokens(temporal, spatial)

    def training_outputs(self, x, mode):
        """One encoder graph followed by four compute-matched classification heads."""
        if mode == 'R':
            self._validate_input(x, 1)
            x = x.repeat(1, 3, 1, 1)
            masks = (MASKS[0],) * 4
        elif mode == 'MV':
            self._validate_input(x, 3)
            masks = MASKS
        else:
            raise ValueError("Training mode must be 'R' or 'MV'")
        features = (self._features(x) - self.mu) / self.sd
        return [self.fuser(features, mask) for mask in masks]

    def forward(self, right):
        """Only real Right radar input is accepted at validation/deployment."""
        self._validate_input(right, 1)
        features = self._features(right)[:, 0]
        right_features = (features - self.mu[:, 2]) / self.sd[:, 2]
        return self.fuser.deploy(right_features)


def validate_checkpoint(checkpoint, fold):
    """Require the established Right-only source model and training provenance."""
    if checkpoint.get('fold') != fold:
        raise ValueError('Source checkpoint held-user fold mismatch')
    expected = {'views': [2], 'seed': 2026, 'mode': 'concat', 'readout': 'mean',
                'tag': 'sv_R_label_global', 'teacher': '', 'aux': 'none',
                'soft_target': 'training_onehot_global_projection'}
    for field, value in expected.items():
        if checkpoint.get('args', {}).get(field) != value:
            raise ValueError(f'Source checkpoint provenance mismatch: {field}')


def build_model(fold, seed, mu, sd, device='cuda'):
    """Load only the source encoder; reset seed before creating a new fusion head."""
    path, _ = _data.source_paths(fold)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint, fold)
    matrix_path = _data.BASE / f'label_smoothing_results/global_matrix_h{fold}.npz'
    if checkpoint['args'].get('global_projection_matrix_sha') != _data.digest(matrix_path):
        raise ValueError('Source checkpoint label projection matrix hash mismatch')
    classifier = _data.Classifier([2], 'concat', 'mean')
    classifier.load_state_dict(checkpoint['state'])
    encoder = classifier.encoder
    if encoder.temporal.xmv != 0 or encoder.spatial.xmv != 0:
        raise ValueError('Source encoder must process each view independently')
    del classifier, checkpoint
    _data.setup(seed)
    fuser = MissingFusion(1664, learned_tokens=True)
    model = EndToEndFusion(encoder, fuser, mu, sd).to(device)
    provenance = {'checkpoint_path': str(path.resolve()), 'checkpoint_sha256': _data.digest(path),
                  'source_fold': fold, 'source_seed': 2026, 'model_seed': seed,
                  'initial_encoder_sha256': state_hash(model.encoder),
                  'initial_fuser_sha256': state_hash(model.fuser)}
    return model, provenance
