"""Audited paired projections, ordinary KD, and CE comparisons."""
import json
import numpy as np
from pipeline import BASE, RESULTS, dump
from followup_analysis import load, check_pair, migration_counts
from summarize import uncertainty

OUT = BASE / 'confusion_kd_results'
STRUCTURE = 'sv_R_project_structure'
UNIFORM = 'sv_R_project_uniform'


def main():
    tags = [STRUCTURE, UNIFORM, 'sv_R_concat_kd', 'sv_R_concat']
    cache = {(t, h): load(t, 2026, h) for t in tags for h in range(1, 9)}
    models = []; pairs = []
    for tag in tags:
        rows = [cache[tag, h][0] for h in range(1, 9) if cache[tag, h] is not None]
        r = {'tag': tag, 'folds': len(rows)}
        if len(rows) == 8:
            r.update(accuracy_pct=float(np.mean([x['accuracy'] for x in rows]) * 100),
                     macro_f1_pct=float(np.mean([x['macro_f1'] for x in rows]) * 100),
                     worst_user_pct=min(x['accuracy'] for x in rows) * 100)
        models.append(r)
    for baseline in (UNIFORM, 'sv_R_concat_kd', 'sv_R_concat'):
        if any(cache[t, h] is None for t in (STRUCTURE, baseline) for h in range(1, 9)): continue
        delta = []; accounting = []
        for h in range(1, 9):
            cm, ca = cache[STRUCTURE, h]; bm, ba = cache[baseline, h]
            check_pair(ca, ba)
            assert cm['initial_head_hash'] == bm['initial_head_hash']
            assert cm['parameters'] == bm['parameters']
            if bm['teacher']:
                assert cm['teacher']['logits_sha256'] == bm['teacher']['logits_sha256']
            if baseline == UNIFORM:
                for key in ('projection_matrix_sha', 'proxy_feature_sha', 'wrapper_sha'):
                    assert cm['config'][key] == bm['config'][key]
            with np.load(cm['teacher']['source'] + '_all_logits.npz') as teacher:
                check_pair(ca, {'ids': ca['ids'], 'y': teacher['y'][ca['ids']],
                               'train_ids': teacher['train_ids'], 'val_ids': teacher['val_ids']})
                counts = migration_counts(ca['y'], ba['pred'], teacher['logits'][ca['ids']].argmax(1), ca['pred'])
            accounting.append({'fold': h, **counts})
            delta.append(cm['accuracy'] - bm['accuracy'])
        u = uncertainty(delta)
        pairs.append({'candidate': STRUCTURE, 'baseline': baseline, **u,
                      'deltas_pp': (np.array(delta) * 100).tolist(), 'accounting': accounting,
                      'passes_gate': u['mean_delta_pp'] >= 2 and u['positive_subjects'] >= 6})
    passes = len(pairs) == 3 and all(p['passes_gate'] for p in pairs if p['baseline'] != 'sv_R_concat')
    dump(OUT / 'summary.json', {'models': models, 'pairs': pairs, 'passes_extension_gate': passes,
         'decision': 'pending' if len(pairs) != 3 else 'expand' if passes else 'do_not_expand',
         'scope': 'Exploratory training-user OOF proxy projection; no physical observability ground truth.'})
    lines = ['# 单视角混淆约束蒸馏：结果', '', '|配置|完成折|accuracy%|macro-F1%|最差用户%|', '|---|---:|---:|---:|---:|']
    for r in models:
        vals = [f'{r[k]:.2f}' if k in r else '待完成' for k in ('accuracy_pct', 'macro_f1_pct', 'worst_user_pct')]
        lines.append(f"|{r['tag']}|{r['folds']}/8|" + '|'.join(vals) + '|')
    lines += ['', '|候选|对照|Δ pp|95%用户bootstrap CI|改善用户|扩展门槛|', '|---|---|---:|---|---:|---|']
    for r in pairs:
        lo, hi = r['ci95_subject_bootstrap_pp']
        lines.append(f"|structure|{r['baseline']}|{r['mean_delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{r['positive_subjects']}/8|{r['passes_gate']}|")
    lines += ['', '需要同时胜过普通KD与uniform，平均至少+2 pp且至少6/8用户改善。',
              '只有8名独立用户，所有区间是探索性统计；未齐全不下结论。混淆矩阵只从原梯度训练样本的内部留一用户预测估计。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('CONFUSION_SUMMARY', [(r['tag'], r['folds']) for r in models], 'decision', 'pending' if len(pairs) != 3 else passes)


if __name__ == '__main__': main()
