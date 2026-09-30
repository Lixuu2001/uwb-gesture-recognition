"""Independent checkpoint/metric/selection replay for the four-arm head study."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in sorted(set(y.tolist()))]))


def forward(weights, x):
    values = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            features = torch.from_numpy(x[start:start+256]).flatten(1)
            joined = torch.cat((features, torch.ones(len(features), 3)), 1)
            hidden = F.gelu(F.linear(joined, weights['net.0.weight'], weights['net.0.bias']))
            values.append(F.linear(hidden, weights['net.3.weight'], weights['net.3.bias']))
    return torch.cat(values).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(3)
    summary = json.loads((args.run/'RESULTS.json').read_text())
    protocol = json.loads((args.run/'protocol.json').read_text())
    canonical_protocol = args.run/'protocol.json'
    assert summary['state'] == 'complete'
    y = np.load(args.data/'cache/y_clutterremoved.npy')
    users = np.load(args.data/'cache/subj_clutterremoved.npy')
    good = ~(((users == 2) & np.isin(y,[10,11])) | ((users == 6) & (y == 3)))
    report = {'state':'verified','runs':[],'artifact_hashes':{},'paired':{},'bootstrap_scope':'descriptive paired user bootstrap; overlapping training folds; exploratory users'}
    for held in range(1,9):
        path = args.data/f'experiments/p1_20260928/multiview_s2026/fold{held}/predictions.npz'
        with np.load(path) as cache:
            ids, features, roles = cache['ids'], cache['features'], cache['role']
        expected_test = np.flatnonzero(good & (users == held))
        expected_val = np.flatnonzero(good & (users == held%8+1))
        reference_history = None
        for arm in protocol['arms']:
            folder = args.run/f'fold{held}'/arm
            r = json.loads((folder/'result.json').read_text())
            for filename, expected_sha in r['files'].items():
                assert sha(folder/filename) == expected_sha
                report['artifact_hashes'][str(folder/filename)] = expected_sha
            history = r['history']
            best = min(history, key=lambda row:row['validation_ce'])
            assert best['epoch'] == r['selected_epoch'] and best['validation_ce'] == r['selected_val_ce']
            assert r['identity']['frozen_features_sha256'] == sha(path)
            hashes = {row['epoch']:row['source_batch_ids_sha256'] for row in history if row['epoch']>0}
            if reference_history is None:
                reference_history = hashes
            else:
                assert all(hashes[e]==reference_history[e] for e in hashes.keys() & reference_history.keys())
            checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            assert checkpoint['identity'] == r['identity']
            with np.load(folder/'predictions.npz') as p:
                np.testing.assert_array_equal(p['ids'],expected_test)
                np.testing.assert_array_equal(p['validation_ids'],expected_val)
                np.testing.assert_array_equal(p['y'],y[expected_test])
                # Forward assembled independently from the four saved tensors.
                logits = forward(checkpoint['model'],features[roles==2])
                val_logits = forward(checkpoint['model'],features[roles==1])
                np.testing.assert_array_equal(logits,p['logits'])
                np.testing.assert_array_equal(val_logits,p['validation_logits'])
                score = macro(p['y'],p['logits'].argmax(1))
                assert np.isclose(score,r['metrics']['macro_accuracy'],atol=1e-12,rtol=0)
                val_ce = F.cross_entropy(torch.from_numpy(val_logits),torch.from_numpy(y[expected_val])).item()
                assert abs(val_ce-r['selected_val_ce'])<1e-7
                cm=np.zeros((12,12),dtype=int)
                for a,b in zip(p['y'],p['logits'].argmax(1)):cm[a,b]+=1
                np.testing.assert_array_equal(cm,r['metrics']['confusion'])
            report['runs'].append({'held':held,'arm':arm,'macro_accuracy':score,
                                   'prediction_replay':'bitwise_equal','earliest_minimum_validation_ce':True})
    for arm in protocol['arms']:
        scores=[r['macro_accuracy'] for r in report['runs'] if r['arm']==arm]
        np.testing.assert_allclose(scores,summary['summary'][arm]['per_user'],rtol=0,atol=1e-12)
    rng=np.random.default_rng(20260930)
    indices=rng.integers(0,8,size=(10000,8))
    for a,b in [('full_ce','subset_ce'),('full_supcon','full_ce'),('full_crossuser','full_ce'),
                ('full_crossuser','full_supcon'),('full_crossuser','historical_LTR')]:
        delta=100*(np.array(summary['summary'][a]['per_user'])-summary['summary'][b]['per_user'])
        report['paired'][a+' minus '+b]={'difference_pp':float(delta.mean()),'improved_users':int((delta>0).sum()),
                'descriptive_95_interval_pp':np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
    report['run_count']=len(report['runs'])
    report['model_and_prediction_artifacts']=len(report['artifact_hashes'])
    report['auditor_sha256']=sha(Path(__file__))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['state','run_count','model_and_prediction_artifacts','paired']}))


if __name__ == '__main__':
    main()
