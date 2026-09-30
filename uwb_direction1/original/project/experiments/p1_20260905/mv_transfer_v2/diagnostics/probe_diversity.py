"""Is the selected set uninformative, or merely redundant? A dimension- and
diversity-matched PCA control plus the internal correlation of each set."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocol as P
import predictability as C


def internal_correlation(block):
    z = (block - block.mean(0)) / (block.std(0) + 1e-9)
    c = np.abs(np.corrcoef(z.T))
    off = ~np.eye(len(c), dtype=bool)
    return float(c[off].mean())


def run(folds=P.FOLDS):
    residual = json.loads((P.DIAG / 'predictability_residual.json').read_text(encoding='utf-8'))
    y, subj = P.labels()
    out = {}
    for fold in folds:
        target, _ = C.teacher_coordinates(fold)
        tr, va, te = P.fold_split(fold, y, subj)
        chosen = np.asarray(residual['per_fold'][str(fold)]['selected_coordinates'])
        rng = np.random.default_rng(fold)
        random_set = rng.choice(target.shape[1], size=len(chosen), replace=False)
        scores = {'selected': [], 'random': [], 'pca': []}
        for user, fit, probe in P.inner_splits(tr, subj):
            pca = PCA(n_components=len(chosen), random_state=0).fit(target[fit])
            for name in scores:
                if name == 'pca':
                    a, b = pca.transform(target[fit]), pca.transform(target[probe])
                else:
                    columns = chosen if name == 'selected' else random_set
                    a, b = target[fit][:, columns], target[probe][:, columns]
                mu, sd = a.mean(0), a.std(0) + 1e-6
                model = LogisticRegression(max_iter=400).fit((a - mu) / sd, y[fit])
                scores[name].append(float(model.score((b - mu) / sd, y[probe])))
        out[fold] = {k: float(np.mean(v)) for k, v in scores.items()}
        out[fold]['internal_correlation_selected'] = internal_correlation(target[tr][:, chosen])
        out[fold]['internal_correlation_random'] = internal_correlation(target[tr][:, random_set])
        print(fold, json.dumps(out[fold]), flush=True)
    across = {k: float(np.mean([out[f][k] for f in folds])) for k in out[folds[0]]}
    P.dump(P.DIAG / 'probe_diversity.json', {'per_fold': out, 'across_folds': across,
                                             'dimensions': 32})
    print(json.dumps(across, indent=2))
    print('PROBE_DIVERSITY_OK')


if __name__ == '__main__':
    run()
