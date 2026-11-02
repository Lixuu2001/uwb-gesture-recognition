"""Summarize paired reliability-gated completion results."""
import json
from pathlib import Path
import numpy as np
from pipeline import dump
from summarize import uncertainty

B = Path(__file__).resolve().parent
OUT = B / 'gated_reconstruction_results'
E2E = B / 'e2e_reconstruction_results'


def load_groups():
    groups = {}
    for p in OUT.glob('*.json'):
        r = json.loads(p.read_text(encoding='utf-8'))
        if 'mode' not in r or 'fold' not in r:
            continue
        groups.setdefault((r['mode'], r['view'], r['seed']), {})[r['fold']] = r
    return groups


def paired(candidate, control, cgroup, xgroup):
    if len(cgroup) != 8 or len(xgroup) != 8:
        return None
    delta = []
    for h in range(1, 9):
        a = np.load(Path(str(OUT / cgroup[h]['tag']) + '_predictions.npz'))
        bpath = E2E / (xgroup[h]['tag'] + '_predictions.npz') if xgroup[h]['tag'].startswith(('fixed_mask', 'capacity', 'joint')) else OUT / (xgroup[h]['tag'] + '_predictions.npz')
        b = np.load(bpath)
        for k in ['ids', 'y', 'train_ids', 'val_ids']:
            np.testing.assert_array_equal(a[k], b[k])
        assert cgroup[h]['teacher_sha'] == xgroup[h]['teacher_sha']
        delta.append(cgroup[h]['accuracy'] - xgroup[h]['accuracy'])
    d = uncertainty(delta)
    return {'candidate': candidate, 'control': control, **d,
            'deltas_pp': (np.array(delta) * 100).tolist(),
            'passes_gate': bool(d['mean_delta_pp'] >= 2 and d['positive_subjects'] >= 6)}


def main():
    groups = load_groups(); models = []; pairs = []
    for (mode, view, seed), rr in sorted(groups.items()):
        if not rr: continue
        a = list(rr.values())
        models.append({'mode': mode, 'view': view, 'seed': seed, 'folds': len(a),
                       'accuracy_pct': float(np.mean([r['accuracy'] for r in a]) * 100),
                       'macro_f1_pct': float(np.mean([r['macro_f1'] for r in a]) * 100),
                       'worst_subject_pct': min(r['accuracy'] for r in a) * 100,
                       'mean_nmse': float(np.mean([r['missing_nmse'] for r in a])),
                       'mean_gate': float(np.mean([r['test_gate_mean'] for r in a])),
                       'parameters': a[0]['parameters']})
    joint = groups.get(('gate_joint', 0, 2026), {})
    cap = groups.get(('gate_capacity', 0, 2026), {})
    if joint and cap:
        pairs.append(paired('gate_joint', 'gate_capacity', joint, cap))
    fixed = {}
    for p in E2E.glob('fixed_mask_L_s2026_h*.json'):
        r = json.loads(p.read_text(encoding='utf-8')); fixed[r['fold']] = r
    if joint and len(fixed) == 8:
        pairs.append(paired('gate_joint', 'e2e_fixed_mask', joint, fixed))
    dump(OUT / 'summary.json', {'models': models, 'pairs': pairs,
                               'note': 'LOSO 8 users; same teacher initialization and visible Left input. Gates are learned on training subjects only.'})
    lines = ['# 可靠性门控单视角重建：结果', '',
             '门控模型只接收 Left 原始输入；两个补全视角经过独立标量 sigmoid 门控，门控为0时退化为零填充。', '',
             '|方法|视角|seed|折数|accuracy%|macro-F1%|最差用户%|缺失NMSE|平均门控|参数|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in models:
        lines.append(f"|{r['mode']}|{r['view']}|{r['seed']}|{r['folds']}/8|{r['accuracy_pct']:.2f}|{r['macro_f1_pct']:.2f}|{r['worst_subject_pct']:.2f}|{r['mean_nmse']:.3f}|{r['mean_gate']:.3f}|{r['parameters']}|")
    lines += ['', '|候选|对照|Δpp|95%用户bootstrap CI|改善用户|研发门槛|', '|---|---|---:|---|---:|---|']
    for r in pairs:
        lo, hi = r['ci95_subject_bootstrap_pp']
        lines.append(f"|{r['candidate']}|{r['control']}|{r['mean_delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{r['positive_subjects']}/8|{r['passes_gate']}|")
    lines += ['', '判定门槛：候选必须同时超过同结构 gate_capacity 和端到端 fixed_mask，平均至少+2个百分点且至少6/8用户改善。',
              '这一步只检验当前冻结教师特征目标、MSE辅助和标量可靠性门控，不等价于证明所有形式的视角重建都不可行。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('GATED_SUMMARY', len(models), len(pairs))


if __name__ == '__main__': main()
