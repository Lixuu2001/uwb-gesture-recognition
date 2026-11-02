"""Permission, leakage and consistency audit over finished runs.

Answers three questions a results table cannot: did the deployed model ever see
a hidden radar, do arms that are being compared actually share their population
and initialisation, and does the reported number come out of the saved
predictions.
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

import protocol as P
from protocol import BASE, RUNS, DIAG, dump

FORBIDDEN_PREFIXES = ('teacher', 'relation', 'process', 'predictor', 'aux')


def completed_runs():
    out = {}
    for status_path in sorted(RUNS.glob('*/status.json')):
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state.get('status') == 'complete':
            out[status_path.parent.name] = (status_path.parent, state)
    return out


def audit_metrics(run, state):
    with np.load(run / 'predictions.npz') as a:
        y, prediction, logits = a['y'], a['pred'], a['logits']
    issues = []
    if not np.array_equal(prediction, logits.argmax(-1)):
        issues.append('stored prediction is not the argmax of the stored logits')
    accuracy = float((prediction == y).mean())
    macro = float(f1_score(y, prediction, labels=np.unique(y), average='macro', zero_division=0))
    if abs(accuracy - state['accuracy']) > 1e-12:
        issues.append(f'accuracy mismatch {accuracy} vs {state["accuracy"]}')
    if abs(macro - state['macro_f1']) > 1e-12:
        issues.append(f'macro-F1 mismatch {macro} vs {state["macro_f1"]}')
    history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    if history[-1]['epoch'] != state['stopped_epoch']:
        issues.append('metrics log does not end at the recorded stopping epoch')
    best = min(history, key=lambda r: r['val_ce'])
    if best['epoch'] != state['best_epoch']:
        issues.append('selected epoch is not the minimum validation cross-entropy')
    return {'accuracy': accuracy, 'macro_f1': macro, 'issues': issues}


def audit_split(run, state):
    y, subj = P.labels()
    tr, va, te = P.fold_split(state['held_subject'], y, subj)
    with np.load(run / 'split.npz') as a:
        saved = (a['train_ids'], a['val_ids'], a['test_ids'])
    issues = []
    for name, expected, got in zip(('train', 'val', 'test'), (tr, va, te), saved):
        if not np.array_equal(expected, got):
            issues.append(f'{name} ids do not match the frozen protocol split')
    if set(saved[0]) & set(saved[2]) or set(saved[1]) & set(saved[2]):
        issues.append('test rows appear in training or validation')
    if set(subj[saved[2]].tolist()) != {state['held_subject']}:
        issues.append('test rows are not exclusively the held-out subject')
    provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    for key, ids in (('student_train_ids_hash', tr), ('validation_ids_hash', va),
                     ('test_ids_hash', te)):
        if provenance[key] != P.hash_ids(ids):
            issues.append(f'{key} in the ledger does not match the protocol')
    return {'issues': issues, 'provenance': provenance}


def audit_deployment(run):
    state = torch.load(run / 'deployment_state.pt', map_location='cpu', weights_only=False)
    bad = [k for k in state['model'] if k.split('.')[0] in FORBIDDEN_PREFIXES]
    return {'exported_modules': sorted({k.split('.')[0] for k in state['model']}),
            'issues': [f'deployed state exports {bad}'] if bad else []}


def audit_hidden_views(run, ids=128):
    """Shuffle the two radars the student is not allowed to use and re-predict.

    A single differing logit would mean the deployed path reads a hidden view.
    """
    from train import Classifier, setup
    from model_adapter import classification_features
    checkpoint = torch.load(run / 'deployment_state.pt', map_location='cpu', weights_only=False)
    setup(P.SPLIT_SEED)
    model = Classifier([P.DEPLOY_VIEW], 'concat', 'mean').cuda().eval()
    model.load_state_dict(checkpoint['model'])
    with np.load(run / 'predictions.npz') as a:
        test_ids = a['ids'][:ids]
    amplitude = np.load(BASE / 'prepared/off.npy', mmap_mode='r')
    phase = np.load(BASE / 'prepared/only.npy', mmap_mode='r')
    triplet = np.concatenate([np.array(amplitude[test_ids], copy=True),
                              np.array(phase[test_ids], copy=True)], -1)   # (n,3,90,378)
    rng = np.random.default_rng(0)
    corrupted = triplet.copy()
    for view in (0, 1):
        corrupted[:, view] = corrupted[rng.permutation(len(corrupted)), view]
    corrupted[:, 0] += 7.0

    def forward(array):
        out = []
        for start in range(0, len(array), 32):
            block = torch.tensor(array[start:start + 32, P.DEPLOY_VIEW:P.DEPLOY_VIEW + 1],
                                 device='cuda', dtype=torch.float16)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = classification_features(model, block)
            out.append(logits.float().cpu().numpy())
        return np.concatenate(out)

    clean, dirty = forward(triplet), forward(corrupted)
    difference = float(np.abs(clean - dirty).max())
    del model
    torch.cuda.empty_cache()
    return {'max_abs_logit_change_when_hidden_views_are_destroyed': difference,
            'issues': [] if difference == 0.0 else ['deployed logits moved when L/T changed']}


def audit_arm_matching(runs):
    """Arms compared against each other must share population and initialisation."""
    groups = {}
    for name, (run, state) in runs.items():
        provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
        key = (state['held_subject'], state['model_seed'])
        groups.setdefault(key, []).append((state['method_tag'], provenance))
    issues = []
    matched = {}
    for (fold, seed), members in sorted(groups.items()):
        reference_tag, reference = members[0]
        for tag, provenance in members[1:]:
            for field in ('student_train_ids_hash', 'validation_ids_hash', 'test_ids_hash',
                          'initial_classifier_hash', 'label_statistics_source'):
                if provenance[field] != reference[field]:
                    issues.append(f'h{fold} seed{seed}: {tag} differs from {reference_tag} in {field}')
        samplers = {}
        for tag, provenance in members:
            samplers.setdefault(provenance['sampler_manifest_hash'], []).append(tag)
        matched[f'h{fold}_seed{seed}'] = {'arms': sorted(t for t, _ in members),
                                          'sampler_groups': {k[:12]: v for k, v in samplers.items()}}
    return {'issues': issues, 'matched': matched}


def run(check_hidden_views=True):
    runs = completed_runs()
    report = {'n_complete': len(runs), 'runs': {}, 'issues': []}
    for name, (path, state) in runs.items():
        entry = {'metrics': audit_metrics(path, state), 'split': audit_split(path, state),
                 'deployment': audit_deployment(path)}
        if check_hidden_views:
            entry['hidden_views'] = audit_hidden_views(path)
        entry['split'].pop('provenance')
        report['runs'][name] = entry
        for section in entry.values():
            report['issues'].extend(f'{name}: {i}' for i in section['issues'])
    report['arm_matching'] = audit_arm_matching(runs)
    report['issues'].extend(report['arm_matching']['issues'])
    report['clean'] = not report['issues']
    dump(DIAG / 'audit.json', report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-hidden-views', action='store_true')
    args = ap.parse_args()
    result = run(not args.skip_hidden_views)
    print(json.dumps({'n_complete': result['n_complete'], 'clean': result['clean'],
                      'issues': result['issues'][:20]}, indent=2))
    print('AUDIT_OK' if result['clean'] else 'AUDIT_FOUND_ISSUES')
