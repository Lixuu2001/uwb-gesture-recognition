"""Within-protocol paired reconstruction comparisons, not old-head comparisons."""
import json
import numpy as np
from pipeline import dump
from reconstruction import OUT
from summarize import uncertainty

def main():
    groups={}
    for p in OUT.glob('*_s*_h*.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        groups.setdefault((r['mode'],r['view'],r['seed']),{})[r['fold']]=r
    models=[];pairs=[]
    for (mode,v,seed),rows in sorted(groups.items()):
        rr=list(rows.values());a=np.array([x['accuracy'] for x in rr])
        models.append({'mode':mode,'view':'LTR' if mode=='real3' else 'LTR'[v],'seed':seed,'folds':len(rr),
                       'accuracy_pct':float(a.mean()*100),'subject_sd_pp':float(a.std(ddof=1)*100) if len(a)>1 else None,
                       'macro_f1_pct':float(np.mean([x['macro_f1'] for x in rr])*100),
                       'missing_nmse':float(np.mean([x['missing_nmse'] for x in rr])),
                       'parameters':rr[0]['parameters']})
        if mode not in ('capacity','joint','pure_recon') or len(rows)!=8:continue
        for ctrl in (['fixed_mask','capacity'] if mode=='joint' else ['fixed_mask']):
            bb=groups.get((ctrl,v,seed),{})
            if len(bb)!=8:continue
            for h in range(1,9):
                a0=np.load(OUT/(bb[h]['tag']+'_predictions.npz'));a1=np.load(OUT/(rows[h]['tag']+'_predictions.npz'))
                for k in ['ids','train_ids','val_ids','y']:np.testing.assert_array_equal(a0[k],a1[k])
                if mode=='joint' and ctrl=='capacity':assert rows[h]['initial_hash']==bb[h]['initial_hash']
            delta=[rows[h]['accuracy']-bb[h]['accuracy'] for h in range(1,9)]
            pairs.append({'candidate':mode,'control':ctrl,'view':'LTR'[v],'seed':seed,**uncertainty(delta),'deltas_pp':(np.array(delta)*100).tolist()})
    dump(OUT/'summary.json',{'models':models,'pairs':pairs,'note':'Frozen fold-local encoder; seed repetitions do not add independent users. All comparisons exploratory, latent reconstruction only.'})
    lines=['# 单视角重建三视角特征：自动汇总','','同一冻结编码器、相同LOSO索引、独立新读出头；不可与旧端到端数值直接相减。',
           'NMSE=预测缺失两路MSE/使用训练均值补全的MSE；小于1说明优于均值填充。重建有效必须同时看分类和容量对照。','',
           '|方法|视角|seed|折数|accuracy %|macroF1 %|缺失NMSE|参数量(不含冻结编码器)|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in models:lines.append(f"|{r['mode']}|{r['view']}|{r['seed']}|{r['folds']}/8|{r['accuracy_pct']:.2f}|{r['macro_f1_pct']:.2f}|{r['missing_nmse']:.3f}|{r['parameters']}|")
    lines+=['','|候选|对照|视角|seed|Δpp|95%被试bootstrap CI|改善人数|','|---|---|---|---:|---:|---|---:|']
    for r in pairs:
        lo,hi=r['ci95_subject_bootstrap_pp']
        lines.append(f"|{r['candidate']}|{r['control']}|{r['view']}|{r['seed']}|{r['mean_delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{r['positive_subjects']}/8|")
    lines+=['','pure_recon 的分类头从真实三路固定，和joint的适应性分类头不同；其对fixed_mask的差值属于整管线比较。',
            '容量对照capacity与joint唯一训练损失差异是额外0.1 MSE；checkpoint均按验证CE。较低重建误差不是恢复三维真值的证明。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('SUMMARY',len(models),'configurations',len(pairs),'complete paired comparisons')

if __name__=='__main__':main()
