"""Bijective source-only controls for future clip-pairing diagnostics.

These are established permutation constructions, not a new learning method.
No scoring or training happens here. Caller must pass ONLY its training IDs.
"""
import numpy as np


def inputs(y,users,pool):
    y=np.asarray(y);users=np.asarray(users);pool=np.asarray(pool)
    if y.ndim!=1 or users.shape!=y.shape:raise ValueError('Expected matching vectors')
    if pool.ndim!=1 or pool.dtype.kind not in 'iu' or len(np.unique(pool))!=len(pool):
        raise ValueError('Pool must contain unique integer IDs')
    if len(pool) and (pool.min()<0 or pool.max()>=len(y)):raise ValueError('Invalid ID')
    return y,users,pool


def same_user_derangement(y,users,pool,seed):
    """Sattolo single cycle within each (class,user) group; singleton unchanged.

    Every non-singleton changes target; every source target is used once.
    This samples single cycles, NOT uniformly all possible derangements.
    """
    y,users,pool=inputs(y,users,pool);rng=np.random.default_rng(seed)
    lookup=np.arange(len(y),dtype=np.int64);singletons=[]
    for label in np.unique(y[pool]):
        for user in np.unique(users[pool[y[pool]==label]]):
            cell=pool[(y[pool]==label)&(users[pool]==user)]
            if len(cell)==1:singletons.append(int(cell[0]));continue
            target=cell.copy()
            for i in range(len(cell)-1,0,-1):
                j=int(rng.integers(i));target[i],target[j]=target[j],target[i]
            lookup[cell]=target
    assert np.array_equal(np.sort(lookup[pool]),np.sort(pool))
    assert np.array_equal(y[lookup[pool]],y[pool]) and np.array_equal(users[lookup[pool]],users[pool])
    return lookup,{'unchanged_singleton_ids':singletons,'changed':int(np.sum(lookup[pool]!=pool)),'total':len(pool)}


def different_user_permutation(y,users,pool,seed):
    """Class-preserving bijection with every target from a different user.

    Requires each user's count <= half of the corresponding class pool.
    Rotate randomized contiguous user blocks by the largest block size.
    This is a specified construction, not a uniform permutation sampler.
    Infeasible classes raise instead of silently changing marginals.
    """
    y,users,pool=inputs(y,users,pool);rng=np.random.default_rng(seed);lookup=np.arange(len(y),dtype=np.int64)
    for label in np.unique(y[pool]):
        cell=pool[y[pool]==label];groups=[cell[users[cell]==u] for u in rng.permutation(np.unique(users[cell]))]
        size=max(map(len,groups))
        if size*2>len(cell):raise ValueError(f'Class {label}: no cross-user bijection exists')
        ordered=np.concatenate([rng.permutation(g) for g in groups]);lookup[ordered]=np.roll(ordered,-size)
    assert np.array_equal(np.sort(lookup[pool]),np.sort(pool))
    assert np.array_equal(y[lookup[pool]],y[pool]) and np.all(users[lookup[pool]]!=users[pool])
    return lookup
