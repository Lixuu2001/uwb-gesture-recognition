"""P0 diagnostic for method A: is the teacher relation anything but the label matrix?

Runs inside the source training rows only. A high neighbour purity here is not
evidence of generalisation, because the teacher was trained on these very rows.
The question this answers is narrower and prior to that: has the teacher code
collapsed onto class centres, onto one blob, or onto the user identity, and does
its cross-user neighbourhood carry structure a label-only matrix does not.
"""
import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

import protocol as P
import sampling
import teacher_targets
from protocol import DIAG, dump

TAU = 0.2


def adjacency(code, allowed, tau=TAU):
    logits = (code @ code.T / tau).masked_fill(~allowed, -1e4)
    return F.softmax(logits, dim=1)


def label_only(labels, subjects, allowed):
    """Same class uniform, other class as close to zero as the mask allows."""
    positive = allowed & (labels[:, None] == labels[None, :])
    reference = positive.float()
    return reference / reference.sum(1, keepdim=True).clamp_min(1e-8)


def group_report(code, labels, subjects, rng_noise):
    allowed = subjects[:, None] != subjects[None, :]
    positive = allowed & (labels[:, None] == labels[None, :])
    similarity = (code @ code.T).masked_fill(~allowed, -2.0)
    order = similarity.argsort(dim=1, descending=True)
    top1 = labels[order[:, 0]] == labels
    top3 = (labels[order[:, :3]] == labels[:, None]).float().mean(1)
    relation = adjacency(code, allowed)
    entropy = -(relation.clamp_min(1e-12).log() * relation).sum(1)
    reference = label_only(labels, subjects, allowed)
    total_variation = 0.5 * (relation - reference).abs().sum(1)
    within = torch.cdist(code, code)
    same_user_class = (~allowed) & (labels[:, None] == labels[None, :])
    stability = []
    for level in (0.05, 0.10):
        noisy = F.normalize(code + level * code.std() * torch.randn(code.shape, generator=rng_noise), dim=-1)
        noisy_order = (noisy @ noisy.T).masked_fill(~allowed, -2.0).argsort(dim=1, descending=True)
        stability.append(float((noisy_order[:, 0] == order[:, 0]).float().mean()))
    return {'top1_cross_user_purity': float(top1.float().mean()),
            'top3_cross_user_purity': float(top3.mean()),
            'relation_entropy': float(entropy.mean()),
            'max_entropy': float(np.log(int(allowed.sum(1).float().mean()))),
            'total_variation_to_label_only': float(total_variation.mean()),
            'positive_mass': float((relation * positive).sum(1).mean()),
            'label_only_positive_mass': 1.0,
            'cross_user_same_class_distance': float(within[positive].mean()),
            'cross_user_other_class_distance': float(within[allowed & ~positive].mean()),
            'same_user_same_class_distance': float(within[same_user_class].mean()) if bool(same_user_class.any()) else None,
            'top1_stable_noise_5pct': stability[0], 'top1_stable_noise_10pct': stability[1],
            'code_variance': float(code.var(0).mean())}


def compare_sources(codes, labels, subjects):
    allowed = subjects[:, None] != subjects[None, :]
    out = {}
    keys = list(codes)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, rb = adjacency(codes[a], allowed), adjacency(codes[b], allowed)
            out[f'{a}_vs_{b}_total_variation'] = float(0.5 * (ra - rb).abs().sum(1).mean())
            out[f'{a}_vs_{b}_top1_agreement'] = float(
                ((codes[a] @ codes[a].T).masked_fill(~allowed, -2.).argmax(1)
                 == (codes[b] @ codes[b].T).masked_fill(~allowed, -2.).argmax(1)).float().mean())
    return out


def run(folds, groups_per_fold=200):
    y, subj = P.labels()
    report = {}
    for fold in folds:
        tr, va, te = P.fold_split(fold, y, subj)
        codes = {}
        for source in ('multi', 'target_single', 'teacher_single'):
            code, _ = teacher_targets.relation_targets(fold, source, tr, device='cpu')
            codes[source] = code
        groups = sampling.relation_epochs(y, subj, tr, 1, P.SAMPLER_SEED + fold)
        groups = groups.reshape(-1, 32)[:groups_per_fold]
        per_source = {s: [] for s in codes}
        pairwise = []
        noise = torch.Generator().manual_seed(P.AUXILIARY_SEED + fold)
        for group in groups:
            labels = torch.tensor(y[group].astype('int64'))
            subjects = torch.tensor(subj[group].astype('int64'))
            sliced = {s: codes[s][group] for s in codes}
            for source, code in sliced.items():
                per_source[source].append(group_report(code, labels, subjects, noise))
            pairwise.append(compare_sources(sliced, labels, subjects))
        report[fold] = {'groups': len(groups),
                        'sources': {s: {k: float(np.mean([r[k] for r in rows]))
                                        for k in rows[0] if rows[0][k] is not None}
                                    for s, rows in per_source.items()},
                        'pairwise': {k: float(np.mean([r[k] for r in pairwise])) for k in pairwise[0]}}
        print('fold', fold, json.dumps(report[fold]['sources']['multi'], indent=None), flush=True)
    across = {}
    for source in ('multi', 'target_single', 'teacher_single'):
        keys = report[folds[0]]['sources'][source]
        across[source] = {k: float(np.mean([report[f]['sources'][source][k] for f in folds])) for k in keys}
    across['pairwise'] = {k: float(np.mean([report[f]['pairwise'][k] for f in folds]))
                          for k in report[folds[0]]['pairwise']}
    out = {'per_fold': report, 'across_folds': across, 'tau': TAU,
           'caveat': 'the teacher was trained on these rows; purity here is not '
                     'unseen-user generalisation, only a degeneracy check'}
    dump(DIAG / 'relation_prediagnostic.json', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    ap.add_argument('--groups', type=int, default=200)
    args = ap.parse_args()
    result = run(args.folds, args.groups)
    print(json.dumps(result['across_folds'], indent=2))
    print('RELATION_DIAGNOSTIC_OK')
