"""Label-free outer LOSO input preparation; no label/model/training API.

Retain frozen development IDs for users1..6 and all raw IDs for users7..8.
Historical contradiction exclusions concern only users2/6; do not recompute
the label-dependent mask using users7..8 truth. Real IDs stay private.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from outer_subject_roles import plan_roles

EXPECTED = {
    'raw': '3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519',
    'subjects': '5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480',
    'development_ids': 'd932f8910326ebf9c114c36e90026e5bf1cf8fdc71233816c9b389dabe3bdc50',
    'development_maps': 'ad38801896b65e676569c379eb03750ed61ecbf920557e3e78de351216683a57',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def rms_batch(raw, ids):
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 1 <= len(ids) <= 64:
        raise ValueError('Require 1..64 integer sample IDs')
    if np.any(ids < 0) or np.any(ids >= len(raw)) or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid or repeated sample ID')
    x = torch.from_numpy(np.array(raw[ids], copy=True))
    if x.dtype != torch.float32 or x.shape[1:] != (3,90,189) or not torch.isfinite(x).all():
        raise ValueError('Require finite full three-radar float32 input')
    x = x / x.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-8)
    result = x.permute(0,2,1,3).reshape(len(x),90,567).numpy()
    if not np.isfinite(result).all(): raise ValueError('Nonfinite RMS output')
    return result

class OuterInputs:
    """Verified, memory-mapped inputs. No gesture labels are loaded or returned."""
    def __init__(self, raw, subjects, development_ids):
        for name,path in [('raw',raw),('subjects',subjects),('development_ids',development_ids)]:
            if sha(path) != EXPECTED[name]: raise ValueError('Source SHA mismatch: '+name)
        self.raw = np.load(raw,mmap_mode='r',allow_pickle=False)
        self.subjects = np.load(subjects,mmap_mode='r',allow_pickle=False)
        if self.raw.shape != (9593,3,90,189) or self.raw.dtype != np.float32:
            raise ValueError('Raw shape/dtype mismatch')
        if self.subjects.shape != (9593,) or self.subjects.dtype != np.int64:
            raise ValueError('Subject shape/dtype mismatch')
        with np.load(development_ids,allow_pickle=False) as z:
            # Explicit key access; legacy NPZ also contains labels, never access it.
            self.dev_ids = z['ids'].copy(); dev_users = z['users'].copy()
        if self.dev_ids.shape != (6900,) or len(np.unique(self.dev_ids)) != 6900:
            raise ValueError('Frozen development IDs invalid')
        np.testing.assert_array_equal(self.subjects[self.dev_ids],dev_users)
        if set(dev_users) != set(range(1,7)): raise ValueError('Development users mismatch')
        extra = np.flatnonzero(np.isin(self.subjects,[7,8]))
        self.ids = np.sort(np.concatenate([self.dev_ids,extra])).astype(np.int64)
        self.users = np.array(self.subjects[self.ids],copy=True)
        if len(extra) != 2400 or len(self.ids) != 9300 or len(np.unique(self.ids)) != 9300:
            raise ValueError('Outer cohort size mismatch')
        np.testing.assert_array_equal(self.ids[self.users <= 6],self.dev_ids)
        for a in [self.ids,self.users,self.dev_ids]: a.flags.writeable = False

    def roles(self, held):
        return plan_roles(self.users.tolist(),self.ids.tolist(),held,8)

    def batch(self, held, role, sample_ids):
        if role not in ('train','validation','test'): raise ValueError('Unknown role')
        allowed = set(self.roles(held)['sample_ids'][role])
        ids = np.asarray(sample_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not set(ids.tolist()) <= allowed:
            raise ValueError('Out-of-role IDs blocked before raw input access')
        return rms_batch(self.raw,ids)

    def labels(self, *args, **kwargs):
        raise PermissionError('Preparation has no label capability; future worker needs separately frozen role guards')
