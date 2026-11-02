"""Raw inputs segregated by fold and available view before model execution."""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from data import BASE, extract, fit_scale, digest, dump


class ViewReader:
    def __init__(self, amplitude, phase, partitions):
        if amplitude.shape != phase.shape or amplitude.ndim != 4 or amplitude.shape[1] != 3:
            raise ValueError('expected aligned three-view amplitude and phase arrays')
        self.amplitude, self.phase, self.partitions = amplitude, phase, partitions
        sets = [set(partitions[k].tolist()) for k in ('train', 'val', 'test')]
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('overlapping fold partitions')

    def load(self, role, mode, device='cuda', batch=64):
        if role not in self.partitions or mode not in ('R', 'MV'):
            raise ValueError('unknown role or view mode')
        if role != 'train' and mode != 'R':
            raise ValueError('validation and test allow Right only')
        ids = self.partitions[role]
        views = slice(2, 3) if mode == 'R' else slice(0, 3)
        shape = (len(ids), 1 if mode == 'R' else 3, self.amplitude.shape[2], 2*self.amplitude.shape[3])
        out = torch.empty(shape, dtype=torch.float16, device=device)
        for begin in range(0, len(ids), batch):
            j = ids[begin:begin+batch]
            chunk = np.concatenate([self.amplitude[j, views], self.phase[j, views]], -1)
            if not np.isfinite(chunk).all():
                raise ValueError(f'nonfinite available input in {role}/{mode}')
            out[begin:begin+len(j)].copy_(torch.from_numpy(chunk))
        return out


def load_fold(fold):
    arrays, meta = extract(fold)
    partitions = {role: arrays[role+'_ids'] for role in ('train', 'val', 'test')}
    reader = ViewReader(np.load(BASE/'prepared/off.npy', mmap_mode='r'),
                        np.load(BASE/'prepared/only.npy', mmap_mode='r'), partitions)
    mu, sd = fit_scale(torch.from_numpy(arrays['train_x']))
    # Only split identity, labels and original predictions remain; no frozen
    # sample features can accidentally become this trainer's model inputs.
    records = {k: v for k, v in arrays.items() if not k.endswith('_x')}
    return reader, records, mu, sd, meta
