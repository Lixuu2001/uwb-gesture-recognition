"""S4: virtual-view synthesis as the second leg of paper one.

The B2 decoder was trained to reconstruct a masked radar view from the other
two (cross-view masked reconstruction, see pretrain.py). The same decoder is a
*generator*: at deploy time, if one radar fails, synthesise its echo from the
survivors and hand the downstream Stage-2 inspector a full three-view input
again. This turns "missing-view robustness" into a *use* of the generative
spine rather than a separate patch -- the same model that rejects unknowns
(e_recon) also fills missing views.

Experiment: for each view v, replace the *real* v with the decoder's synthetic
v (other two real), keep all three views, and score open-set with the inspector.
Compare against:
  * real3  -- all three real views (upper bound, == the main inspector head);
  * drop_v -- only the two surviving real views (E2 lower bound, no synthesis).
Go/no-go: synth_v OSCR drop vs real3 < 5 points (the synthetic view is good
enough to stand in for the failed radar).

The Stage-2 inspector bank is fit ONCE on the real three-view train set and
reused for every scenario, so the only thing that changes across real3 / synth_v
/ drop_v is the test-time input -- not the bank -- keeping the comparison clean.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .inspector import train_inspectors, extract, anomaly_scores, _softmax
from .metrics import auroc, oscr, best_threshold_acc_f1
from .pretrain import _encode_visible


class _PreNormDataset(Dataset):
    """Wraps an already-normalised (N, V, T, M) tensor + labels. Unlike
    MVDataset it does NOT re-normalise -- the views (one of them synthetic) are
    fed to the model as-is, matching the space the decoder reconstructs in."""

    def __init__(self, X, y):
        self.X = X                                   # float tensor (N,V,T,M)
        self.y = np.asarray(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], int(self.y[i])


@torch.no_grad()
def synth_replace_dataset(enc, dec, ds, cfg, device, vstar):
    """Return a _PreNormDataset whose view `vstar` is the decoder's synthetic
    view (reconstructed from the other views), all other views left real.

    `ds` yields already-normalised (V, T, M) samples with V == cfg.n_views and
    view slots 0..V-1 (the standard all-three-view eval set)."""
    enc.eval(); dec.eval()
    V = cfg.n_views
    others = [k for k in range(V) if k != vstar]
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers)
    xs, ys = [], []
    for x, y in loader:
        x = x.to(device)                              # (B, V, T, M)
        mem = _encode_visible(enc, x[:, others], others)
        synth = dec(mem)                              # (B, T, M), normalised space
        x = x.clone()
        x[:, vstar] = synth
        xs.append(x.cpu())
        ys.append(np.asarray(y))
    return _PreNormDataset(torch.cat(xs), np.concatenate(ys))


def _score_head(bank, vindices, model, known_ds, unk_ds, n_known, cfg, device):
    """Score one open-set scenario against a *given* bank. Mirrors the inspector
    head computation in evaluate_inspector (combined mode) so numbers are
    directly comparable to the main run."""
    Fk, Lk, yk = extract(model, known_ds, cfg, device, view_indices=vindices)
    Fu, Lu, _ = extract(model, unk_ds, cfg, device, view_indices=vindices)
    pred_k, pred_u = Lk.argmax(1), Lu.argmax(1)
    sk = anomaly_scores(bank, vindices, Fk, pred_k, cfg, device, mode="combined")
    su = anomaly_scores(bank, vindices, Fu, pred_u, cfg, device, mode="combined")
    conf_k, conf_u = -sk, -su
    is_known = np.concatenate([np.ones(len(conf_k)), np.zeros(len(conf_u))])
    correct_k = (pred_k == yk).astype(float)
    acc, f1, thr = best_threshold_acc_f1(pred_k, conf_k, yk, conf_u, n_known,
                                         pred_u=pred_u)
    return dict(auroc=auroc(np.concatenate([conf_k, conf_u]), is_known),
                oscr=oscr(conf_k, correct_k, conf_u),
                acc=acc, f1=f1, thr=float(thr), closed_acc=float(correct_k.mean()))


def evaluate_synth_views(cfg, model, enc, dec, train_ds, known_ds, unk_ds,
                         n_known, device, verbose=True):
    """S4 heads. Bank fit once on real train_ds; scored on real3, each
    synth_v (view v replaced by its synthesis), and each drop_v (the two
    surviving real views only). Returns a dict of heads, all `synth_*` keyed."""
    V = cfg.n_views
    if dec is None or V < 3:
        return {}
    bank, Vbank = train_inspectors(model, train_ds, cfg, n_known, device)
    names = list(cfg.views)                           # e.g. ['Left','Top','Right']
    heads = {}

    # Upper bound: all three real views, same bank -> equals the main inspector.
    heads["synth_real3"] = _score_head(bank, list(range(V)), model,
                                       known_ds, unk_ds, n_known, cfg, device)

    for v in range(V):
        tag = names[v][0]                             # L / T / R
        others = [k for k in range(V) if k != v]
        # synth_v: replace view v with its synthesis, keep all three views.
        ks = synth_replace_dataset(enc, dec, known_ds, cfg, device, v)
        us = synth_replace_dataset(enc, dec, unk_ds, cfg, device, v)
        heads[f"synth_{tag}"] = _score_head(bank, list(range(V)), model,
                                            ks, us, n_known, cfg, device)
        # drop_v: only the two surviving real views (E2-style lower bound).
        heads[f"synth_drop{tag}"] = _score_head(bank, others, model,
                                                known_ds, unk_ds, n_known, cfg, device)
        if verbose:
            print(f"  [S4] view {names[v]}: real3={heads['synth_real3']['oscr']*100:5.2f}"
                  f"  synth={heads[f'synth_{tag}']['oscr']*100:5.2f}"
                  f"  drop={heads[f'synth_drop{tag}']['oscr']*100:5.2f}  (OSCR)")
    return heads
