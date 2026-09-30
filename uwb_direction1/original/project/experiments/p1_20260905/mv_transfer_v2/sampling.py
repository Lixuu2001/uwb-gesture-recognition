"""Replayable batch order. Saved with every run so arms share exposure."""
import hashlib
import math

import numpy as np


def permutation_epochs(train_ids, epochs, batch, seed):
    """Ordinary shuffled order, the sampler the existing baselines already use."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(train_ids)
    steps = math.ceil(len(ids) / batch)
    out = np.full((epochs, steps, batch), -1, dtype=np.int64)
    for e in range(epochs):
        order = rng.permutation(ids)
        for s in range(steps):
            chunk = order[s * batch:(s + 1) * batch]
            out[e, s, :len(chunk)] = chunk
    return out


def relation_epochs(y, subj, train_ids, epochs, seed, classes=4, users=4, per=2,
                    groups_per_step=4, max_attempts=100):
    """Cross-user relation groups of classes x users x per samples each.

    Each group is one microbatch, so every anchor has cross-user neighbours and
    (users-1)*per same-class cross-user positives inside its own group. Groups
    never span an optimiser step.
    """
    y = np.asarray(y); subj = np.asarray(subj); train_ids = np.asarray(train_ids)
    pools = {}
    for cls in np.unique(y[train_ids]):
        for user in np.unique(subj[train_ids]):
            ids = train_ids[(y[train_ids] == cls) & (subj[train_ids] == user)]
            if len(ids) >= per:
                pools[(int(cls), int(user))] = ids
    class_list = sorted({c for c, _ in pools})
    steps = math.ceil(len(train_ids) / (classes * users * per * groups_per_step))
    rng = np.random.default_rng(seed)
    out = np.zeros((epochs, steps, groups_per_step, classes * users * per), dtype=np.int64)
    for e in range(epochs):
        queue = []
        for step in range(steps):
            for group in range(groups_per_step):
                chosen, eligible = None, []
                for attempt in range(max_attempts + 1):
                    if attempt == max_attempts:
                        raise RuntimeError('no eligible cross-user relation group; '
                                           'never silently shrink the group instead')
                    if len(queue) < classes:
                        queue = list(rng.permutation(class_list))
                    chosen = queue[:classes]
                    eligible = sorted({u for c, u in pools if c == chosen[0]})
                    for c in chosen[1:]:
                        eligible = [u for u in eligible if (c, u) in pools]
                    if len(eligible) >= users:
                        queue = queue[classes:]
                        break
                    queue = list(rng.permutation(class_list))
                picked = rng.choice(eligible, size=users, replace=False)
                rows = []
                for c in chosen:
                    for u in picked:
                        rows.extend(rng.choice(pools[(c, int(u))], size=per, replace=False))
                assert len(set(rows)) == len(rows), 'duplicate sample inside a relation group'
                out[e, step, group] = rows
    return out


def exposure(groups, y, subj):
    ids = groups.reshape(-1)
    ids = ids[ids >= 0]
    return {'samples_per_epoch': int(len(ids) / groups.shape[0]),
            'unique_samples': int(len(np.unique(ids))),
            'class_counts': np.bincount(y[ids], minlength=12).tolist(),
            'user_counts': {int(u): int((subj[ids] == u).sum()) for u in np.unique(subj[ids])}}


def manifest_hash(groups):
    return hashlib.sha256(np.ascontiguousarray(groups).tobytes()).hexdigest()[:32]
