"""Standalone Right-only inference from an exported end-to-end checkpoint.

Requires architecture source files and deployment.pt. No original training
checkpoint, smoothing matrix, feature cache, labels, or hidden radar inputs are
opened by this loader. The input is the prepared amplitude+phase representation
[B,1,90,378], not an unprocessed radar ADC stream.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT.parent


def _architecture_module(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class RightOnlyDeployment(nn.Module):
    """Read-only inference graph; training mode is intentionally unavailable."""

    def __init__(self, encoder, fuser, mu, sd):
        super().__init__()
        if mu.shape != (1, 3, 1664) or sd.shape != mu.shape:
            raise ValueError('Expected saved mu/sd [1,3,1664]')
        if not torch.isfinite(mu).all() or not torch.isfinite(sd).all() or not (sd > 0).all():
            raise ValueError('Invalid saved normalization')
        self.encoder = encoder
        self.fuser = fuser
        self.register_buffer('mu', mu.detach().clone().float())
        self.register_buffer('sd', sd.detach().clone().float())
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        if mode:
            raise ValueError('This deployment model supports eval mode only')
        return super().train(False)

    @torch.inference_mode()
    def forward(self, right):
        if right.ndim != 4 or right.shape[1:] != (1, 90, 378) or len(right) == 0:
            raise ValueError('Deployment accepts only nonempty Right input [B,1,90,378]')
        if not right.is_floating_point() or not torch.isfinite(right).all():
            raise ValueError('Right input must contain finite floating-point values')
        # Restore eval even if a caller directly toggled a nested module.
        super().train(False)
        _, temporal, spatial = self.encoder(right.float())
        batch = len(right)
        features = torch.cat((temporal.reshape(batch, 1, 10, 9, 32).mean(3).flatten(2),
                              spatial.reshape(batch, 1, 42, 9, 32).mean(3).flatten(2)), dim=-1).float()[:, 0]
        features = (features - self.mu[:, 2]) / self.sd[:, 2]
        return self.fuser.deploy(features)


def load_deployment(checkpoint, device='cpu'):
    """Instantiate the fixed architecture and strictly load only exported state."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or 'state' not in saved:
        raise ValueError('Expected deployment export containing state')
    state = saved['state']
    config_module = _architecture_module(BASE / 'vendor/config.py', '_right_deploy_vendor_config')
    encoder_module = _architecture_module(BASE / 'vendor/uwb_osr/model.py', '_right_deploy_encoder_architecture')
    fuser_module = _architecture_module(PARENT / 'models.py', '_right_deploy_fusion_architecture')
    config = config_module.Config(views=['Left'], T=90, M=189, pool='flatten', use_fusion=False,
                                  doppler='concat', dim=32, depth=4, heads=4, d_key=0,
                                  mlp_ratio=2., dropout=.1, xmv_layers=0, xmv_bottleneck=4)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='enable_nested_tensor is True')
        encoder = encoder_module.SharedDualPathEncoder(config)
    fuser = fuser_module.MissingFusion(1664, learned_tokens=True)
    model = RightOnlyDeployment(encoder, fuser, state['mu'], state['sd'])
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Deployment export contains nonfinite weights')
    return model.to(device).eval()


@torch.inference_mode()
def predict_logits(model, right, batch=128, precision='fp32'):
    """Run an exported model in explicit FP32 or device-native BF16 autocast."""
    if right.ndim != 4 or right.shape[1:] != (1, 90, 378) or len(right) == 0:
        raise ValueError('Expected nonempty Right input [B,1,90,378]')
    if batch < 1 or precision not in ('fp32', 'bf16'):
        raise ValueError('Require positive batch and fp32/bf16 precision')
    device = model.mu.device
    logits = []
    for start in range(0, len(right), batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
            output = model(right[start:start + batch].to(device=device, dtype=torch.float32))
        logits.append(output.float().cpu())
    return torch.cat(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Right amplitude+phase .npy [B,1,90,378]')
    parser.add_argument('--output', type=Path, required=True, help='Output .npz with logits and pred')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(4)
    model = load_deployment(args.checkpoint, args.device)
    right = torch.from_numpy(np.load(args.input, allow_pickle=False))
    logits = predict_logits(model, right, args.batch, args.precision).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, logits=logits, pred=logits.argmax(1))
    print(f'DEPLOYED rows={len(logits)} device={args.device} precision={args.precision} output={args.output}')


if __name__ == '__main__':
    main()
