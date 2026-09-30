"""Audit adaptive versus matched-global label smoothing."""
import json
import numpy as np
from pipeline import BASE, RESULTS, dump, digest, split
from followup_analysis import load, check_pair
from summarize import uncertainty

OUT = BASE/'label_smoothing_results'
ADAPTIVE = 'sv_R_label_uniform'
GLOBAL = 'sv_R_label_global'


def main():
    y = np.load(BASE/'prepared/y.npy'); subjects = np.load(BASE/'prepared/subj.npy')
    tags = [ADAPTIVE, GLOBAL, 'sv_R_concat', 'sv_R_concat_kd']
    cache = {(tag, fold): load(tag, 2026, fold) for tag in tags for fold in range(1, 9)}
    models = []
    for tag in tags:
        rows = [cache[tag, fold][0] for fold in range(1, 9) if cache[tag, fold] is not None]
        item = {'tag': tag, 'folds': len(rows)}
        if len(rows) == 8:
            item.update(accuracy_pct=float(np.mean([row['accuracy'] for row in rows])*100),
                        macro_f1_pct=float(np.mean([row['macro_f1'] for row in rows])*100))
        models.append(item)
    pairs = []
    for candidate, baseline in [(ADAPTIVE, GLOBAL), (GLOBAL, 'sv_R_concat'),
                                (ADAPTIVE, 'sv_R_concat')]:
        if any(cache[tag, fold] is None for tag in (candidate, baseline) for fold in range(1, 9)):
            continue
        deltas = []
        for fold in range(1, 9):
            cm, ca = cache[candidate, fold]; bm, ba = cache[baseline, fold]
            check_pair(ca, ba)
            assert cm['initial_head_hash'] == bm['initial_head_hash']
            assert cm['parameters'] == bm['parameters']
            if GLOBAL in (candidate, baseline):
                gm = cache[GLOBAL, fold][0]
                assert gm['teacher'] is None and gm['config']['teacher'] == ''
                tr, va, te = split(y, subjects, fold, 2026)
                matrix_path = OUT/f'global_matrix_h{fold}.npz'
                assert gm['config']['global_projection_matrix_sha'] == digest(matrix_path)
                with np.load(matrix_path) as archive:
                    for key, ids in [('train_ids', tr), ('val_ids', va), ('test_ids', te)]:
                        np.testing.assert_array_equal(archive[key], ids)
                    matrix = archive['global_matrix']
                    counts = archive['class_counts']
                source_path = BASE/f'confusion_kd_results/matrix_h{fold}.npz'
                with np.load(source_path) as archive:
                    adaptive = archive['uniform']
                adaptive_mass = np.average(1-np.diag(adaptive), weights=counts)
                global_mass = np.average(1-np.diag(matrix), weights=counts)
                np.testing.assert_allclose(adaptive_mass, global_mass, atol=1e-7)
                np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-6)
            deltas.append(cm['accuracy']-bm['accuracy'])
        stat = uncertainty(deltas)
        stat.update(candidate=candidate, baseline=baseline,
                    gate_pass=stat['mean_delta_pp'] >= 2 and stat['positive_subjects'] >= 6)
        pairs.append(stat)
    status = 'complete' if len(pairs) == 3 else 'pending'
    dump(OUT/'summary.json', {'models': models, 'pairs': pairs, 'status': status,
                              'gate': 'adaptive-global >= +2.00 pp and >=6/8 positive users'})
    lines = ['# 类别自适应平滑必要性消融', '',
             '全局控制与自适应版本使用相同网络、初始化、训练循环和训练样本平均平滑质量。', '',
             '|配置|折数|accuracy%|macro-F1%|', '|---|---:|---:|---:|']
    for row in models:
        acc = f"{row['accuracy_pct']:.2f}" if 'accuracy_pct' in row else '待完成'
        f1 = f"{row['macro_f1_pct']:.2f}" if 'macro_f1_pct' in row else '待完成'
        lines.append(f"|{row['tag']}|{row['folds']}/8|{acc}|{f1}|")
    lines += ['', '|候选|基线|Δ pp|95%用户bootstrap CI|改善用户|通过门槛|',
              '|---|---|---:|---|---:|---:|']
    for row in pairs:
        lo, hi = row['ci95_subject_bootstrap_pp']
        lines.append(f"|{row['candidate']}|{row['baseline']}|{row['mean_delta_pp']:+.2f}|"
                     f"[{lo:+.2f},{hi:+.2f}]|{row['positive_subjects']}/8|{row['gate_pass']}|")
    lines += ['', '主判据：adaptive相对global至少+2.00个百分点且至少6/8用户改善。',
              '区间包含零不证明统计等效；失败时只报告未发现类别自适应的增量证据。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('LABEL_SMOOTHING_SUMMARY', status, [(row['tag'], row['folds']) for row in models])


if __name__ == '__main__':
    main()
