"""Per-coordinate skill distribution by branch: is ANY hidden-view coordinate
predictable from the deployed radar, or is the whole branch below the
class-centre baseline?"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocol as P
import predictability as C
from diagnostics.c0_probe import BRANCHES, ridge_skill


def run(folds=P.FOLDS):
    y, subj = P.labels()
    features = C.build_probe_features()
    per_fold = {}
    for fold in folds:
        target, _ = C.teacher_coordinates(fold)
        tr, va, te = P.fold_split(fold, y, subj)
        users = sorted(set(subj[tr].tolist()))
        skills, _ = ridge_skill(features, target, users, y, subj, tr)
        mean_skill = skills.mean(0)
        per_fold[fold] = {name: {
            'mean': float(mean_skill[cols].mean()),
            'max': float(mean_skill[cols].max()),
            'fraction_positive': float((mean_skill[cols] > 0).mean()),
            'fraction_above_0.2': float((mean_skill[cols] > 0.2).mean()),
        } for name, cols in BRANCHES.items()}
        print('fold', fold, json.dumps({k: round(v['mean'], 3) for k, v in per_fold[fold].items()}),
              flush=True)
    across = {name: {key: float(np.mean([per_fold[f][name][key] for f in folds]))
                     for key in per_fold[folds[0]][name]} for name in BRANCHES}
    P.dump(P.DIAG / 'c0_branch_detail.json', {'per_fold': per_fold, 'across_folds': across})
    print(json.dumps(across, indent=2))
    print('C0_BRANCH_DETAIL_OK')


if __name__ == '__main__':
    run()
