"""Paired audit and gate for cross-user OOF privileged distillation."""
import json
import numpy as np
from pipeline import BASE, RESULTS, dump, digest, split
from followup_analysis import load, check_pair
from summarize import uncertainty
from oof_privileged import OUT, TAGS


def main():
    y = np.load(BASE/'prepared/y.npy'); subjects = np.load(BASE/'prepared/subj.npy')
    tags = [TAGS['multi'], TAGS['single'], 'sv_R_label_global', 'sv_R_label_uniform',
            'sv_R_concat_kd', 'sv_R_concat']
    cache = {(tag, fold): load(tag, 2026, fold) for tag in tags for fold in range(1, 9)}
    models = []
    for tag in tags:
        rows = [cache[tag, fold][0] for fold in range(1, 9) if cache[tag, fold] is not None]
        item = {'tag': tag, 'folds': len(rows)}
        if len(rows) == 8:
            item.update(accuracy_pct=float(np.mean([row['accuracy'] for row in rows])*100),
                        macro_f1_pct=float(np.mean([row['macro_f1'] for row in rows])*100))
        models.append(item)
    baselines = [TAGS['single'], 'sv_R_label_global', 'sv_R_label_uniform',
                 'sv_R_concat_kd', 'sv_R_concat']
    pairs = []
    for baseline in baselines:
        candidate = TAGS['multi']
        if any(cache[tag, fold] is None for tag in (candidate, baseline) for fold in range(1, 9)):
            continue
        deltas = []
        for fold in range(1, 9):
            cm, ca = cache[candidate, fold]; bm, ba = cache[baseline, fold]
            check_pair(ca, ba)
            assert cm['initial_head_hash'] == bm['initial_head_hash']
            assert cm['parameters'] == bm['parameters']
            proxy_path = OUT/f'proxy_h{fold}.npz'
            assert cm['config']['proxy_sha'] == digest(proxy_path)
            proxy_meta = json.loads(proxy_path.with_suffix('.json').read_text())
            assert proxy_meta['warnings'] == 0 and not proxy_meta['validation_or_test_used']
            tr, va, te = split(y, subjects, fold, 2026)
            with np.load(proxy_path) as archive:
                for key, ids in [('train_ids', tr), ('val_ids', va), ('test_ids', te)]:
                    np.testing.assert_array_equal(archive[key], ids)
                np.testing.assert_allclose(archive['multi'].sum(1), 1, atol=1e-7)
                np.testing.assert_allclose(archive['single'].sum(1), 1, atol=1e-7)
            deltas.append(cm['accuracy']-bm['accuracy'])
        stat = uncertainty(deltas)
        stat.update(candidate=candidate, baseline=baseline,
                    gate_pass=stat['mean_delta_pp'] >= 2 and stat['positive_subjects'] >= 6)
        pairs.append(stat)
    main_pairs = {row['baseline']: row for row in pairs}
    gate_complete = TAGS['single'] in main_pairs and 'sv_R_label_global' in main_pairs
    gate_pass = gate_complete and main_pairs[TAGS['single']]['gate_pass'] and \
                main_pairs['sv_R_label_global']['gate_pass']
    status = 'complete' if len(pairs) == len(baselines) else 'pending'
    dump(OUT/'summary.json', {'models': models, 'pairs': pairs, 'status': status,
                              'passes_extension_gate': gate_pass,
                              'gate': 'multi must beat both single OOF and global smoothing by >=2 pp and >=6/8 users'})
    lines = ['# 跨用户折外三视角特权蒸馏', '',
             '每个训练样本的软标签来自未见过该样本所属用户的代理；部署学生只读取Right幅度+相位。', '',
             '|配置|折数|accuracy%|macro-F1%|', '|---|---:|---:|---:|']
    for row in models:
        acc = f"{row['accuracy_pct']:.2f}" if 'accuracy_pct' in row else '待完成'
        f1 = f"{row['macro_f1_pct']:.2f}" if 'macro_f1_pct' in row else '待完成'
        lines.append(f"|{row['tag']}|{row['folds']}/8|{acc}|{f1}|")
    lines += ['', '|候选|基线|Δ pp|95%用户bootstrap CI|改善用户|单项门槛|',
              '|---|---|---:|---|---:|---:|']
    for row in pairs:
        lo, hi = row['ci95_subject_bootstrap_pp']
        lines.append(f"|{row['candidate']}|{row['baseline']}|{row['mean_delta_pp']:+.2f}|"
                     f"[{lo:+.2f},{hi:+.2f}]|{row['positive_subjects']}/8|{row['gate_pass']}|")
    lines += ['', f"联合扩展门槛通过：{gate_pass}。",
              '只有同时超过单视角OOF和全局平滑控制才归因于三视角折外信息。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('OOF_PRIVILEGED_SUMMARY', status, 'gate', gate_pass,
          [(row['tag'], row['folds']) for row in models])


if __name__ == '__main__':
    main()
