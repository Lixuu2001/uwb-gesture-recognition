"""Label-coupled reversal of raw slow time; a known augmentation, not novelty."""
import numpy as np
import torch
from circular_signal import encode, get_batch

Q = np.array([1, 0, 3, 2, 7, 6, 5, 4, 9, 8, -1, -1])


def eligible(y, users):
    # Do not introduce a synthetic class absent from that user's retained data.
    keep = np.zeros(len(y), dtype=bool)
    for u in np.unique(users):
        present = np.unique(y[users == u])
        keep |= (users == u) & (y < 10) & np.isin(Q[y], present)
    return keep


@torch.no_grad()
def extract(net, raw, cache, ids, batch=16):
    correct, naive = [], []
    for start in range(0, len(ids), batch):
        part = ids[start:start + batch]
        rev = encode(np.asarray(raw[part])[:, :, ::-1, :])
        actual = get_batch(rev, np.arange(len(part)), 'legacy')
        wrong = get_batch(cache, part, 'legacy').flip(-2)
        for x, dest in [(actual, correct), (wrong, naive)]:
            pooled = net.encoder(x.reshape(-1, 4, 90, 63)).reshape(len(x), -1)
            dest.append(net.head[:3](pooled).numpy())
    return np.concatenate(correct).astype(np.float64), np.concatenate(naive).astype(np.float64)


def augment(x, y, users, reverse, naive, arm):
    mask = eligible(y, users)
    extra = naive if arm == 'naive_mix' else reverse
    labels = y[mask] if arm == 'unmapped_mix' else Q[y[mask]]
    assert len(extra) == mask.sum()
    return np.concatenate([x, extra]), np.r_[y, labels], np.r_[users, users[mask]]
