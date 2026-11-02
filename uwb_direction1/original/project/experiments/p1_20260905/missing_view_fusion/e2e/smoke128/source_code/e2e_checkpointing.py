"""Atomic epoch checkpoints, including optimizer and all sampling RNG states."""
import random
from pathlib import Path
import numpy as np
import torch


def atomic_save(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(content, tmp)
    tmp.replace(path)


def save_progress(path, model, optimizer, sampler, info):
    atomic_save(path, {'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'sampler': sampler.get_state(), 'python_rng': random.getstate(),
                      'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                      'info': info})


def load_progress(path, model, optimizer, sampler, expected_identity=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if expected_identity is not None and saved['info'].get('run_identity') != expected_identity:
        raise ValueError('progress run identity mismatch; model and RNG were not restored')
    model.load_state_dict(saved['state'])
    optimizer.load_state_dict(saved['optimizer'])
    sampler.set_state(saved['sampler'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng']:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['info']
