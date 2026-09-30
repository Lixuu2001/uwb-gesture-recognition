"""Aggregate finished runs, recompute every reported number from the predictions.

Nothing here reads a checkpoint or a metrics line to decide a result: accuracy
and macro-F1 come back out of predictions.npz, so a stale metric can never be
ranked. Runs that are not complete are listed, never quietly skipped.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

import protocol as P
from protocol import RUNS, dump

BOOTSTRAP_SEED = 20260906
BOOTSTRAP_DRAWS = 10000


def collect():
    rows = []
    for status_path in sorted(RUNS.glob('*/status.json')):
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state.get('status') != 'complete':
            continue
        run = status_path.parent
        with np.load(run / 'predictions.npz') as a:
            y, prediction = a['y'], a['pred']
            np.testing.assert_array_equal(prediction, a['logits'].argmax(-1))
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        rows.append({'name': run.name, 'method_tag': state['method_tag'],
                     'fold': state['held_subject'], 'model_seed': state['model_seed'],
                     'accuracy': state['accuracy'], 'macro_f1': state['macro_f1'],
                     'recomputed_accuracy': float((prediction == y).mean()),
                     'recomputed_macro_f1': float(f1_score(y, prediction, labels=np.unique(y),
                                                           average='macro', zero_division=0)),
                     'n_test': int(len(y)), 'best_epoch': state['best_epoch'],
                     'stopped_epoch': state['stopped_epoch'],
                     'elapsed_seconds': state['elapsed_seconds'],
                     'relation': config.get('relation'), 'sampler': config.get('sampler'),
                     'process': config.get('process'),
                     'parameters_deployed': state['parameters_deployed']})
    return rows


def incomplete():
    out = []
    for status_path in sorted(RUNS.glob('*/status.json')):
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state.get('status') != 'complete':
            out.append({'name': status_path.parent.name, 'status': state.get('status'),
                        'error': state.get('error')})
    return out


def per_fold(rows):
    """arm -> fold -> seed -> accuracy, plus the per-arm equal-weight mean."""
    table = {}
    for row in rows:
        table.setdefault(row['method_tag'], {}).setdefault(row['fold'], {})[row['model_seed']] = row
    return table


def coverage(folds_table, seeds, folds=P.FOLDS):
    """True only when every requested fold carries every requested seed.

    Partial seed coverage is the trap this guards: averaging seeds inside a fold
    while some folds have one seed and others two produces a headline number
    that belongs to no protocol at all.
    """
    return all(fold in folds_table and set(seeds) <= set(folds_table[fold]) for fold in folds)


def bootstrap_delta(candidate, baseline, seeds, folds=P.FOLDS):
    """Paired user bootstrap over a fixed seed set present in both arms.

    Seeds are averaged inside a user before resampling, because two seeds of the
    same person are not two independent people.
    """
    if not (coverage(candidate, seeds, folds) and coverage(baseline, seeds, folds)):
        return None
    deltas = []
    for fold in folds:
        deltas.append(np.mean([candidate[fold][s]['recomputed_accuracy']
                               - baseline[fold][s]['recomputed_accuracy'] for s in seeds]))
    deltas = np.asarray(deltas)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = deltas[rng.integers(0, len(deltas), size=(BOOTSTRAP_DRAWS, len(deltas)))].mean(1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {'mean_delta_pp': float(deltas.mean() * 100),
            'ci95_pp': [float(low * 100), float(high * 100)],
            'positive_users': int((deltas > 0).sum()), 'n_users': int(len(deltas)),
            'per_user_delta_pp': (deltas * 100).round(4).tolist(),
            'seeds': list(seeds)}


def transfer_ledger(candidate, baseline, seeds, folds=P.FOLDS):
    """Newly correct, newly wrong and net change, so a mean is never the whole story."""
    gained = lost = 0
    for fold in folds:
        for seed in list(seeds)[:1]:                  # one seed keeps sample counts meaningful
            a = np.load(RUNS / candidate[fold][seed]['name'] / 'predictions.npz')
            b = np.load(RUNS / baseline[fold][seed]['name'] / 'predictions.npz')
            np.testing.assert_array_equal(a['ids'], b['ids'])
            right_a, right_b = a['pred'] == a['y'], b['pred'] == b['y']
            gained += int((right_a & ~right_b).sum())
            lost += int((~right_a & right_b).sum())
    return {'newly_correct': gained, 'newly_wrong': lost, 'net': gained - lost}


def arm_summary(table, arm, seeds):
    """Accuracy of one arm over one explicit seed set. Folds missing a requested
    seed are excluded and counted, never silently averaged over fewer seeds."""
    folds = table.get(arm, {})
    accuracies, f1s, used = [], [], []
    for fold in P.FOLDS:
        if fold not in folds or not set(seeds) <= set(folds[fold]):
            continue
        accuracies.append(np.mean([folds[fold][s]['recomputed_accuracy'] for s in seeds]))
        f1s.append(np.mean([folds[fold][s]['recomputed_macro_f1'] for s in seeds]))
        used.append(fold)
    if not accuracies:
        return None
    return {'folds_complete': len(accuracies), 'folds': used, 'seeds': list(seeds),
            'accuracy_pct': float(np.mean(accuracies) * 100),
            'macro_f1_pct': float(np.mean(f1s) * 100),
            'std_subject_pp': float(np.std(accuracies, ddof=0) * 100),
            'worst_user_pct': float(np.min(accuracies) * 100)}


COMPARISONS = [('A-GS', 'B-GS'),   # sampler effect at identical loss
               ('A-SC', 'A-GS'), ('A-US', 'A-SC'), ('A-MV', 'A-SC'), ('A-MV', 'A-US'),
               ('A-TS', 'A-US'), ('A-MV', 'A-TS'), ('A-SHUF', 'A-MV'), ('A-RKD', 'A-GS'),
               ('B-SELF', 'B-GS'), ('B-MV', 'B-GS'), ('B-MV', 'B-SELF'), ('B-Q', 'B-MV'),
               ('B-AMP', 'B-MV'), ('B-SHUF', 'B-MV'), ('B-TIME', 'B-MV'), ('B-CDF', 'B-MV'),
               ('AB-MV', 'A-MV'), ('AB-MV', 'B-MV'), ('AB-USMV', 'AB-MV'), ('AB-MVSELF', 'AB-MV')]


def build(write=True):
    rows = collect()
    table = per_fold(rows)
    all_seeds = sorted({row['model_seed'] for row in rows})
    seed_sets = [[s] for s in all_seeds] + ([all_seeds] if len(all_seeds) > 1 else [])
    arms = {}
    for arm in sorted(table):
        for seeds in seed_sets:
            summary = arm_summary(table, arm, seeds)
            if summary is not None:
                arms[f'{arm}@{"+".join(map(str, seeds))}'] = summary
    comparisons = []
    for candidate, baseline in COMPARISONS:
        if candidate not in table or baseline not in table:
            continue
        for seeds in seed_sets:
            stats = bootstrap_delta(table[candidate], table[baseline], seeds)
            if stats is None:
                continue
            stats.update({'candidate': candidate, 'baseline': baseline,
                          'ledger': transfer_ledger(table[candidate], table[baseline], seeds),
                          'gate_pass': bool(stats['mean_delta_pp'] >= 2.0
                                            and stats['positive_users'] >= 6)})
            comparisons.append(stats)
    report = {'arms': arms, 'comparisons': comparisons, 'incomplete_runs': incomplete(),
              'seeds_present': all_seeds,
              'gate': 'development gate: at least +2.00 pp and at least 6 of 8 users; '
                      'this is an investment threshold, not a significance test. '
                      'Every row names the exact seed set it was computed over; a '
                      'comparison is emitted only when both arms cover all eight folds '
                      'for that seed set.'}
    if write:
        dump(RUNS.parent / 'diagnostics/summary.json', report)
        (RUNS.parent / 'diagnostics/SUMMARY.md').write_text(render(report), encoding='utf-8')
    return report


def render(report):
    lines = ['# mv_transfer_v2 结果汇总', '',
             '所有数字由 predictions.npz 重新计算。每一行显式标注它是在哪个 seed 集合上算的；',
             '只有两个臂在该 seed 集合上都齐 8 折，才输出该比较。', '',
             '|arm@seeds|折数|accuracy%|macro-F1%|被试标准差pp|最差被试%|',
             '|---|---:|---:|---:|---:|---:|']
    for arm, s in report['arms'].items():
        lines.append(f"|{arm}|{s['folds_complete']}/8|{s['accuracy_pct']:.2f}|{s['macro_f1_pct']:.2f}|"
                     f"{s['std_subject_pp']:.2f}|{s['worst_user_pct']:.2f}|")
    lines += ['', '|候选|基线|seeds|Δ pp|95%用户bootstrap CI|改善用户|新增正确|变错|净|门槛|',
              '|---|---|---|---:|---|---:|---:|---:|---:|---:|']
    for c in report['comparisons']:
        ci = f"[{c['ci95_pp'][0]:+.2f},{c['ci95_pp'][1]:+.2f}]"
        led = c['ledger']
        seeds = '+'.join(map(str, c['seeds']))
        lines.append(f"|{c['candidate']}|{c['baseline']}|{seeds}|{c['mean_delta_pp']:+.2f}|{ci}|"
                     f"{c['positive_users']}/{c['n_users']}|{led['newly_correct']}|{led['newly_wrong']}|"
                     f"{led['net']:+d}|{c['gate_pass']}|")
    if report['incomplete_runs']:
        lines += ['', '## 未完成任务', '']
        lines += [f"- {r['name']}: {r['status']} {r.get('error') or ''}" for r in report['incomplete_runs']]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    report = build()
    if not args.quiet:
        print(render(report))
