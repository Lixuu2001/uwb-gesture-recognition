"""Aggregate all results, enforce paired comparisons, and expose failures."""
import itertools,json,csv
from pathlib import Path
import numpy as np
from pipeline import BASE,RESULTS,dump

def uncertainty(d):
    """Subject is the resampling unit; seeds never count as extra subjects."""
    d=np.asarray(d);rng=np.random.default_rng(7401)
    b=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
    signs=np.array(list(itertools.product([-1,1],repeat=len(d))))
    p=float((np.abs((signs*d).mean(1))>=abs(d.mean())-1e-12).mean())
    return {'mean_delta_pp':float(d.mean()*100),'ci95_subject_bootstrap_pp':(np.quantile(b,[.025,.975])*100).tolist(),
            'positive_subjects':int((d>0).sum()),'n_subjects':len(d),'sign_flip_p_unadjusted':p}

def main():
    groups={};all_rows=[]
    for p in RESULTS.glob('*_s*_h*.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        if 'accuracy' not in r:continue
        groups.setdefault((r['tag'],r['seed']),{})[r['held_subject']]=r;all_rows.append(r)
    summary=[];pairs=[]
    for (tag,seed),rows in sorted(groups.items()):
        rr=list(rows.values());a=np.array([r['accuracy'] for r in rr])
        summary.append({'tag':tag,'seed':seed,'folds':len(a),'accuracy_pct':100*a.mean(),
                        'std_subject_pp':100*a.std(ddof=1) if len(a)>1 else 0,
                        'macro_f1_pct':100*np.mean([r['macro_f1'] for r in rr]),
                        'parameters':rr[0]['parameters'],'mean_best_epoch':np.mean([r['best_epoch'] for r in rr]),
                        'total_seconds':sum(r['elapsed_seconds'] for r in rr)})
        if not tag.startswith('sv_') or tag.endswith('_off'):continue
        baseline='_'.join(tag.split('_')[:2])+'_off';bb=groups.get((baseline,seed),{})
        if len(rows)!=8 or len(bb)!=8:continue
        for held in range(1,9):
            p0=RESULTS/f'{baseline}_s{seed}_h{held}_predictions.npz';p1=RESULTS/f'{tag}_s{seed}_h{held}_predictions.npz'
            a0=np.load(p0);a1=np.load(p1)
            for key in ['ids','train_ids','val_ids','y']:np.testing.assert_array_equal(a0[key],a1[key])
            if rows[held]['mode']==bb[held]['mode']:
                assert rows[held]['initial_head_hash']==bb[held]['initial_head_hash'],(tag,held,'init differs')
        dd=[rows[h]['accuracy']-bb[h]['accuracy'] for h in range(1,9)]
        pairs.append({'candidate':tag,'baseline':baseline,'seed':seed,**uncertainty(dd),'deltas_pp':[100*d for d in dd]})
    dump(RESULTS/'summary.json',{'models':summary,'comparisons':pairs,
        'statistical_note':'Exploratory multiple comparisons; unadjusted p values, 8 independent subjects only. Bootstrap intervals do not cure small N or selection.'})
    if summary:
        with open(RESULTS/'model_summary.csv','w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    text=['# 第一部分实验结果（自动聚合）','',
          '质量审计后 9300 条、12 类、8 被试；细节见 PLAN.md 和 AUDIT.md。',
          '完整八折才能报告该配置的 LOSO 结果。以下自动列出已完成折数，未完成配置不与完整配置排名。','',
          '| 配置 | seed | 完成折数 | accuracy % | 被试标准差 | macro-F1 % | 参数量 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['tag']} | {r['seed']} | {r['folds']}/8 | {r['accuracy_pct']:.2f} | {r['std_subject_pp']:.2f} | {r['macro_f1_pct']:.2f} | {r['parameters']:,} |")
    text+=['','## 配对变化','', '| 候选 | 对照 | seed | 平均 Δ pp | 用户 bootstrap 95% CI | 改善用户 |','|---|---|---:|---:|---|---:|']
    for r in pairs:
        lo,hi=r['ci95_subject_bootstrap_pp'];text.append(f"| {r['candidate']} | {r['baseline']} | {r['seed']} | {r['mean_delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['positive_subjects']}/8 |")
    text+=['','区间以被试为单位，八名用户样本较少；多配置探索存在选择效应，不宣称所有比较均为确认性显著结论。']
    (RESULTS/'RESULTS_LIVE.md').write_text('\n'.join(text),encoding='utf-8')
    print('SUMMARIZED',len(summary),'models',len(pairs),'paired comparisons')

if __name__=='__main__':main()
