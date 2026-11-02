"""Atomic epoch recovery and strict, first-minimum validation selection."""
import hashlib
import math
from pathlib import Path
import random
import numpy as np
import torch


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


class Selection:
    def __init__(self, initial_ce, best_epoch=0, stale=0):
        if not math.isfinite(initial_ce): raise ValueError('Nonfinite validation CE')
        self.best_ce, self.best_epoch, self.stale = float(initial_ce), int(best_epoch), int(stale)

    def observe(self, epoch, ce):
        if not math.isfinite(ce): raise ValueError('Nonfinite validation CE')
        improved = ce < self.best_ce
        if improved:
            self.best_ce, self.best_epoch, self.stale = float(ce), int(epoch), 0
        else: self.stale += 1
        return improved

    def should_stop(self, epoch, minimum, patience):
        return epoch >= minimum and self.stale >= patience

    def state_dict(self): return vars(self).copy()

    @classmethod
    def from_state(cls, state): return cls(state['best_ce'], state['best_epoch'], state['stale'])


class ShuffleSampler:
    def __init__(self, ids, seed):
        self.ids = np.array(ids, dtype=np.int64); self.rng = np.random.default_rng(seed)
    def next_epoch(self): return self.rng.permutation(self.ids)
    def state_dict(self): return {'ids': self.ids.copy(), 'rng': self.rng.bit_generator.state}
    def load_state_dict(self, state):
        if not np.array_equal(self.ids, state['ids']): raise ValueError('Shuffle partition changed')
        self.rng.bit_generator.state = state['rng']


def save_progress(path, model, optimizer, sampler, info):
    atomic_save(path, {'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'sampler': sampler.state_dict(), 'info': info,
                      'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                      'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None})


def load_progress(path, model, optimizer, sampler, expected_identity):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['info']['identity'] != expected_identity:
        raise ValueError('Resume identity changed; no state was restored')
    candidate = saved['sampler']
    current = sampler.state_dict()
    if 'identity' in current and current['identity'] != candidate.get('identity'):
        raise ValueError('Resume sampler identity changed; no state was restored')
    if 'ids' in current and not np.array_equal(current['ids'], candidate.get('ids')):
        raise ValueError('Resume sampler IDs changed; no state was restored')
    model.load_state_dict(saved['state'], strict=True)
    optimizer.load_state_dict(saved['optimizer']); sampler.load_state_dict(candidate)
    random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng'] is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['info']
