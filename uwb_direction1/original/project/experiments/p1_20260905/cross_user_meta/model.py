"""Original Right classifier plus a bounded, training-only multiview head.

No historical checkpoints are loaded. Every arm has identical initial topology.
The original Classifier class is compiled unchanged from train.py in an isolated
namespace, avoiding that script's import-time pipeline/config module collisions.
"""
import ast
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]


def _load_file(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_config = _load_file('_p1_cross_user_meta_vendor_config', BASE / 'vendor/config.py')
_vendor = _load_file('_p1_cross_user_meta_vendor_model', BASE / 'vendor/uwb_osr/model.py')
_source = ast.parse((BASE / 'train.py').read_text(encoding='utf-8'))
_class = next(node for node in _source.body if isinstance(node, ast.ClassDef) and node.name == 'Classifier')
_namespace = {'__name__': __name__, 'torch': torch, 'nn': nn, 'Config': _config.Config,
              'SharedDualPathEncoder': _vendor.SharedDualPathEncoder,
              'ClosedSetHead': _vendor.ClosedSetHead, 'reduce_tokens': _vendor.reduce_tokens}
exec(compile(ast.Module(body=[_class], type_ignores=[]), str(BASE / 'train.py'), 'exec'), _namespace)
Classifier = _namespace['Classifier']


def pool_tokens(temporal, spatial):
    """Same ordered 10 temporal / 42 spatial nine-token bins as frozen work."""
    b, v, t, d = temporal.shape
    if (t, d, spatial.shape[2:]) != (90, 32, (378, 32)):
        raise ValueError('Expected temporal90/spatial378 tokens with dimension32')
    return torch.cat((temporal.reshape(b, v, 10, 9, d).mean(3).flatten(2),
                      spatial.reshape(b, v, 42, 9, d).mean(3).flatten(2)), -1).float()


class AuxiliaryFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(1664) for _ in range(3))
        self.head = nn.Sequential(nn.Linear(3 * 1664, 256), nn.GELU(), nn.Linear(256, 12))

    def forward(self, features):
        return self.head(torch.cat([norm(features[:, i]) for i, norm in enumerate(self.norms)], -1))


class MetaRadar(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct first, preserving RNG-equal original Right weights/head.
        self.right = Classifier([2], 'concat', 'mean')
        self.fuser = AuxiliaryFuser()

    @staticmethod
    def _validate_input(x, views):
        if x.ndim != 4 or tuple(x.shape[1:]) != (views, 90, 378):
            raise ValueError(f'Expected radar input [B,{views},90,378]')

    def right_only(self, x):
        """Deployment boundary: accepts only [B,1,90,378], returns logits."""
        self._validate_input(x, 1)
        return self.right(x)[0]

    def forward(self, x):
        return self.right_only(x)

    def support_outputs(self, x, mode):
        """Return (Right logits, MV logits or None), using one encoder call.

        R mode requires one actual Right view and repeats it internally so the
        support encoder shape matches MV. Tokens remain independent across views.
        MV mode requires the physical [Left, Top, Right] order.
        """
        if mode == 'R':
            self._validate_input(x, 1)
            x = x.repeat(1, 3, 1, 1)
        elif mode == 'MV':
            self._validate_input(x, 3)
        else:
            raise ValueError("Support mode must be 'R' or 'MV'")
        _, temporal, spatial = self.right.encoder(x)
        right_features = torch.cat((temporal[:, 2].flatten(1), spatial[:, 2].flatten(1)), -1)
        right_logits = self.right.head(right_features)
        mv_logits = self.fuser(pool_tokens(temporal, spatial)) if mode == 'MV' else None
        return right_logits, mv_logits


def build_model(seed, device='cpu'):
    """Initialize every arm from the same seed; never load any checkpoint."""
    torch.manual_seed(seed)
    return MetaRadar().to(device)
