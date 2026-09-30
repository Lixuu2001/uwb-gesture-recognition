"""Load one exported residual model; input is preprocessed Right radar only."""
from pathlib import Path

import torch
from torch import nn

from data import Classifier, digest, pool_tokens
from models import MissingFusion, SharedSpecific, PrototypeCompletion


def restore_adapter(checkpoint):
    arm = checkpoint['config']['arm']
    dim = checkpoint['feature_dim']
    if arm.startswith('SS_'):
        adapter = SharedSpecific(dim)
    elif arm.startswith('PROTO_'):
        adapter = PrototypeCompletion(dim, checkpoint['state']['banks'], trainable_bank=arm == 'PROTO_MLP')
    else:
        adapter = MissingFusion(dim, learned_tokens=arm in ('JOINT_TOKEN', 'JOINT_CONTRAST'))
    adapter.load_state_dict(checkpoint['state'])
    return adapter.eval()


class RadarDeployment(nn.Module):
    def __init__(self, encoder, base_head, adapter, right_mu, right_sd, readout='residual'):
        super().__init__()
        self.encoder = encoder
        self.base_head = base_head
        self.adapter = adapter
        self.readout = readout
        self.register_buffer('right_mu', right_mu)
        self.register_buffer('right_sd', right_sd)

    @classmethod
    def load(cls, path, device='cuda'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        candidates = [Path(p) for p in checkpoint['encoder_identity'] if p.endswith('.pt')]
        if len(candidates) != 1:
            raise ValueError('export requires exactly one frozen encoder source')
        source = candidates[0]
        if digest(source) != checkpoint['encoder_identity'][str(source)]:
            raise ValueError('frozen encoder source changed')
        base = Classifier([2], 'concat', 'mean')
        base.load_state_dict(torch.load(source, map_location='cpu', weights_only=False)['state'])
        model = cls(base.encoder, base.head, restore_adapter(checkpoint),
                    checkpoint['right_mu'], checkpoint['right_sd'], checkpoint['config'].get('readout', 'residual'))
        return model.to(device).eval().requires_grad_(False)

    def forward(self, right):
        if right.ndim != 4 or right.shape[1:] != (1, 90, 378):
            raise ValueError('input must be Right-only [B,1,90,378] amplitude+phase')
        with torch.autocast(device_type=right.device.type, dtype=torch.bfloat16,
                            enabled=right.device.type == 'cuda'):
            _, temporal, spatial = self.encoder(right)
            base = self.base_head(torch.cat([temporal.flatten(2), spatial.flatten(2)], -1)[:, 0]).float()
            # Match the frozen cache's float16 storage before train-only scaling.
            features = pool_tokens(temporal, spatial)[:, 0].half().float()
        features = (features-self.right_mu)/self.right_sd
        correction = self.adapter.deploy(features)
        return correction if self.readout == 'direct' else base+correction
