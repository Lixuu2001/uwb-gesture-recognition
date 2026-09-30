"""Separate phase representation effects from incremental KD effects.

All labels here are post-training evaluation only. Nothing feeds this analysis
back into training, teacher selection, or sample weights.
"""
import json
import time
from pathlib import Path
import numpy as np
from pipeline import BASE, RESULTS, dump, digest, split
from summarize import uncertainty

OUT = BASE / 'followup_results'
CONTRASTS = [('sv_R_concat', 'sv_R_off'), ('sv_R_concat_kd', 'sv_R_concat'),
             ('sv_R_concat_kd', 'sv_R_off'), ('sv_R_concat', 'sv_R_phase_abs'),
             ('sv_R_phase_abs', 'sv_R_duplicate')]


def check_pair(a, b):
    for key in ('ids', 'y', 'train_ids', 'val_ids'):
        np.testing.assert_array_equal(a[key], b[key])


def seed_clustered_effect(by_seed):
    seeds = sorted(by_seed)
    values = np.stack([by_seed[s] for s in seeds])
    if values.shape != (len(seeds), 8): raise ValueError('Require eight aligned subject effects per seed')
    per_subject = values.mean(0)
    return {**uncertainty(per_subject), 'n_seeds': len(seeds), 'seeds': seeds,
            'per_subject_seed_mean_delta_pp': (per_subject * 100).tolist()}


def migration_counts(y, baseline, teacher, candidate):
    bc = baseline == y; tc = teacher == y; cc = candidate == y
    gain = int((~bc & cc).sum()); lost = int((bc & ~cc).sum())
    assert gain - lost == int(cc.sum()) - int(bc.sum())
    return {'n': len(y), 'baseline_correct': int(bc.sum()), 'candidate_correct': int(cc.sum()),
            'gained': gain, 'lost': lost, 'net_correct': gain - lost,
            'teacher_opportunities': int((~bc & tc).sum()),
            'teacher_opportunities_captured': int((~bc & tc & cc).sum())}


def load(tag, seed, fold):
    stem = RESULTS / f'{tag}_s{seed}_h{fold}'
    if not stem.with_suffix('.json').exists(): return None
    meta = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(str(stem) + '_predictions.npz') as archive:
        a = {k: archive[k] for k in archive.files}
    y = np.load(BASE / 'prepared/y.npy'); subjects = np.load(BASE / 'prepared/subj.npy')
    for key, ids in zip(('train_ids', 'val_ids', 'ids'), split(y, subjects, fold, seed)):
        np.testing.assert_array_equal(a[key], ids)
    np.testing.assert_array_equal(a['y'], y[a['ids']])
    np.testing.assert_array_equal(a['pred'], a['logits'].argmax(1))
    assert np.isfinite(a['logits']).all()
    assert abs(meta['accuracy'] - float(np.mean(a['pred'] == a['y']))) < 1e-12
    assert meta['seed'] == seed and meta['held_subject'] == fold
    return meta, a


def main():
    OUT.mkdir(exist_ok=True)
    tags = sorted({t for pair in CONTRASTS for t in pair} | {'teacher_concat'})
    cache = {(tag, seed, h): load(tag, seed, h) for tag in tags for seed in (2026, 2027) for h in range(1, 9)}
    models = []; pairs = []; combined = []; accounts = []
    for tag in tags:
        for seed in (2026, 2027):
            rows = [cache[tag, seed, h] for h in range(1, 9) if cache[tag, seed, h] is not None]
            r = {'tag': tag, 'seed': seed, 'folds': len(rows)}
            if len(rows) == 8:
                r.update(accuracy_pct=float(np.mean([m['accuracy'] for m, a in rows]) * 100),
                         macro_f1_pct=float(np.mean([m['macro_f1'] for m, a in rows]) * 100),
                         parameters=rows[0][0]['parameters'])
            models.append(r)
    for candidate, baseline in CONTRASTS:
        effects = {}
        for seed in (2026, 2027):
            if any(cache[t, seed, h] is None for t in (candidate, baseline) for h in range(1, 9)): continue
            dd = []
            for h in range(1, 9):
                cm, ca = cache[candidate, seed, h]; bm, ba = cache[baseline, seed, h]
                check_pair(ca, ba)
                if cm['mode'] == bm['mode']:
                    assert cm['initial_head_hash'] == bm['initial_head_hash']
                dd.append(cm['accuracy'] - bm['accuracy'])
                if cm.get('teacher'):
                    teacher_path = Path(cm['teacher']['source'] + '_all_logits.npz')
                    assert digest(teacher_path) == cm['teacher']['logits_sha256']
                    with np.load(teacher_path) as ta:
                        np.testing.assert_array_equal(ta['train_ids'], ca['train_ids'])
                        np.testing.assert_array_equal(ta['val_ids'], ca['val_ids'])
                        np.testing.assert_array_equal(ta['y'][ca['ids']], ca['y'])
                        tp = ta['logits'][ca['ids']].argmax(1)
                    counts = migration_counts(ca['y'], ba['pred'], tp, ca['pred'])
                    accounts.append({'candidate': candidate, 'baseline': baseline, 'seed': seed,
                                     'fold': h, 'teacher': str(teacher_path), **counts})
            effects[seed] = dd
            u = uncertainty(dd)
            pairs.append({'candidate': candidate, 'baseline': baseline, 'seed': seed, **u,
                          'passes_expansion_gate': u['mean_delta_pp'] >= 2 and u['positive_subjects'] >= 6})
        if len(effects) == 2:
            combined.append({'candidate': candidate, 'baseline': baseline, **seed_clustered_effect(effects)})
    totals = []
    for candidate, baseline in CONTRASTS:
        for seed in (2026, 2027):
            rows = [r for r in accounts if (r['candidate'], r['baseline'], r['seed']) == (candidate, baseline, seed)]
            if len(rows) != 8: continue
            keys = ('n', 'baseline_correct', 'candidate_correct', 'gained', 'lost', 'net_correct',
                    'teacher_opportunities', 'teacher_opportunities_captured')
            totals.append({'candidate': candidate, 'baseline': baseline, 'seed': seed,
                           **{k: sum(r[k] for r in rows) for k in keys}})
    result = {'updated': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'models': models,
              'pairs': pairs, 'seed_clustered': combined, 'accounting_totals': totals,
              'accounting_by_subject': accounts,
              'scope': 'Post-hoc repeated experiments on the same eight subjects; seed changes both initialization and train/validation split. No independent confirmation.'}
    dump(OUT / 'analysis.json', result)
    lines = ['# 第一部分续研：已完成结果与缺口', '', '只对完整8折报告均值；未完成实验仅报告折数。', '',
             '|方法|seed|完成折|accuracy%|macro-F1%|', '|---|---:|---:|---:|---:|']
    for r in models:
        acc = f"{r['accuracy_pct']:.2f}" if 'accuracy_pct' in r else '待完成'
        f1 = f"{r['macro_f1_pct']:.2f}" if 'macro_f1_pct' in r else '待完成'
        lines.append(f"|{r['tag']}|{r['seed']}|{r['folds']}/8|{acc}|{f1}|")
    lines += ['', '|候选|基线|seed|增量pp|用户bootstrap 95%区间|改善用户|', '|---|---|---:|---:|---|---:|']
    for r in pairs:
        lo, hi = r['ci95_subject_bootstrap_pp']
        lines.append(f"|{r['candidate']}|{r['baseline']}|{r['seed']}|{r['mean_delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{r['positive_subjects']}/8|")
    lines += ['', '## 两seed按用户聚合（仍只有8名独立用户）', '']
    for r in combined:
        lo, hi = r['ci95_subject_bootstrap_pp']
        lines.append(f"- {r['candidate']} − {r['baseline']}：{r['mean_delta_pp']:+.2f} pp；CI [{lo:+.2f},{hi:+.2f}]；{r['positive_subjects']}/8用户的seed均值改善。")
    lines += ['', '## KD错误迁移账本', '', '|candidate|baseline|seed|新增正确|损失正确|净正确数|教师机会|捕获机会|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in totals:
        lines.append(f"|{r['candidate']}|{r['baseline']}|{r['seed']}|{r['gained']}|{r['lost']}|{r['net_correct']}|{r['teacher_opportunities']}|{r['teacher_opportunities_captured']}|")
    lines += ['', 'concatKD−off包含相位和KD共同变化；只有concatKD−concat检验相位之上的KD增量。',
              '按样本汇总的净正确数与按用户等权平均的pp有不同权重，不能直接互换。',
              'Right是前轮多配置探索选出的视角；两seed使用相同测试用户，不构成独立外部验证。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('FOLLOWUP_ANALYSIS', len(pairs), 'complete contrasts;', len(combined), 'two-seed contrasts')
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == '__main__': main()
