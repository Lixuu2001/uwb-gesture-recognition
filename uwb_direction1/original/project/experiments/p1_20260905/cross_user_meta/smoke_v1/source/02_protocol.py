"""Whole-user partitions, source-only episodes and guarded raw radar access."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def quality_mask(y, subjects):
    return ~(((subjects == 2) & np.isin(y, [10, 11])) | ((subjects == 6) & (y == 3)))


def subject_split(y, subjects, held):
    if held not in range(1, 9) or y.shape != subjects.shape or y.ndim != 1:
        raise ValueError('Require aligned labels/subjects and held user 1..8')
    if set(np.unique(subjects)) != set(range(1, 9)):
        raise ValueError('This protocol requires all eight source subject identities')
    good = quality_mask(y, subjects); val = held % 8 + 1
    parts = {'train': np.where(good & ~np.isin(subjects, [held, val]))[0],
             'val': np.where(good & (subjects == val))[0],
             'test': np.where(good & (subjects == held))[0]}
    if any(len(ids) == 0 for ids in parts.values()) or set(y[parts['train']]) != set(range(12)):
        raise ValueError('Empty partition or missing training class')
    return parts


class EpisodeSampler:
    """Rotate query users; uniformly sample available classes within each user."""
    def __init__(self, ids, labels, subjects, seed, per_user):
        self.ids = np.asarray(ids, dtype=np.int64)
        if self.ids.ndim != 1 or len(np.unique(self.ids)) != len(self.ids) or per_user < 1:
            raise ValueError('Unique training IDs and positive per-user episode count required')
        if (self.ids < 0).any() or (self.ids >= len(labels)).any() or len(labels) != len(subjects):
            raise ValueError('Invalid training IDs')
        self.users = np.unique(subjects[self.ids])
        if len(self.users) != 6:
            raise ValueError('Episode sampler requires exactly six training users')
        self.groups = {int(u): {int(c): self.ids[(subjects[self.ids] == u) & (labels[self.ids] == c)]
                               for c in np.unique(labels[self.ids[subjects[self.ids] == u]])}
                       for u in self.users}
        self.per_user = int(per_user); self.step = 0
        self.rng = np.random.default_rng(seed)
        self.identity = {'ids': array_hash(self.ids), 'labels': array_hash(labels[self.ids]),
                         'subjects': array_hash(subjects[self.ids]), 'per_user': self.per_user}

    def _sample(self, user):
        groups = self.groups[int(user)]; classes = np.array(sorted(groups))
        labels = []
        while len(labels) < self.per_user:
            labels.extend(self.rng.permutation(classes).tolist())
        return np.array([self.rng.choice(groups[c]) for c in labels[:self.per_user]], dtype=np.int64)

    def next(self):
        query = int(self.users[self.step % len(self.users)])
        support = np.concatenate([self._sample(u) for u in self.users if u != query])
        qids = self._sample(query)
        self.step += 1
        return {'support_ids': support, 'query_ids': qids, 'query_subject': query}

    def state_dict(self):
        return {'identity': copy.deepcopy(self.identity), 'step': self.step,
                'rng': copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state['identity'] != self.identity:
            raise ValueError('Episode resume identity differs from partition or per-user batch')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.step = int(state['step'])


class SubjectData:
    def __init__(self, amplitude, phase, partitions):
        if amplitude.shape != phase.shape or amplitude.ndim != 4 or amplitude.shape[1:] != (3, 90, 189):
            raise ValueError('Require aligned prepared arrays [N,3,90,189]')
        self.amplitude, self.phase = amplitude, phase
        self.partitions = partitions
        if set(partitions) != {'train', 'val', 'test'}:
            raise ValueError('Require train/val/test partitions')
        sets = [set(np.asarray(partitions[k]).tolist()) for k in ('train', 'val', 'test')]
        if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Partition overlap')

    def load_ids(self, ids, role, mode, device='cuda'):
        ids = np.asarray(ids)
        if role not in self.partitions or mode not in ('R', 'MV'):
            raise ValueError('Unknown partition/view mode')
        if role != 'train' and mode != 'R':
            raise ValueError('Whole-user validation and test require Right only')
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(ids) == 0:
            raise ValueError('Nonempty integer row IDs required')
        if not np.isin(ids, self.partitions[role]).all():
            raise ValueError('Requested IDs leave the allowed subject partition')
        view = slice(2, 3) if mode == 'R' else slice(0, 3)
        x = np.concatenate((self.amplitude[ids, view], self.phase[ids, view]), axis=-1)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite available radar input')
        return torch.from_numpy(x).to(device=device, dtype=torch.float16)


def open_fold(held):
    labels = np.load(BASE/'prepared/y.npy')
    subjects = np.load(BASE/'prepared/subj.npy')
    partitions = subject_split(labels, subjects, held)
    reader = SubjectData(np.load(BASE/'prepared/off.npy', mmap_mode='r'),
                         np.load(BASE/'prepared/only.npy', mmap_mode='r'), partitions)
    return reader, labels, subjects, partitions
