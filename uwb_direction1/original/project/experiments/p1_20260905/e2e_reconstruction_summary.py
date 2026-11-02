"""Paired E2E comparisons, retaining subject as the unit of inference."""
import json
import numpy as np
from e2e_reconstruction import OUT
from pipeline import dump
from summarize import uncertainty

def main():
    groups={}
    for p in OUT.glob('*_s*_h*.json'):
        r=json.loads(p.read_text());groups.setdefault((r['mode'],r['view'],r['seed']),{})[r['fold']]=r
    models=[];pairs=[]
    for (mode,v,seed),rr in sorted(groups.items()):
        a=list(rr.values())
        models.append({'mode':mode,'view':'LTR'[v],'seed':seed,'folds':len(a),'accuracy_pct':float(np.mean([r['accuracy'] for r in a])*100),
                       'macro_f1_pct':float(np.mean([r['macro_f1'] for r in a])*100),'worst_subject_pct':min(r['accuracy'] for r in a)*100,
                       'mean_nmse':float(np.mean([r['missing_nmse'] for r in a])),'parameters':a[0]['parameters']})
        if mode!='joint' or len(rr)!=8:continue
        for ctrl in ['fixed_mask','capacity']:
            bb=groups.get((ctrl,v,seed),{})
            if len(bb)!=8:continue
            for h in range(1,9):
                a0=np.load(OUT/(bb[h]['tag']+'_predictions.npz'));a1=np.load(OUT/(rr[h]['tag']+'_predictions.npz'))
                for k in ['ids','y','train_ids','val_ids']:np.testing.assert_array_equal(a0[k],a1[k])
                for k in ['initial_head_hash','initial_encoder_hash','teacher_sha']:assert rr[h][k]==bb[h][k],(h,k)
                if ctrl=='capacity':assert rr[h]['initial_hash']==bb[h]['initial_hash']
            delta=[rr[h]['accuracy']-bb[h]['accuracy'] for h in range(1,9)]
            d=uncertainty(delta)
            pairs.append({'candidate':mode,'control':ctrl,'view':'LTR'[v],'seed':seed,**d,'deltas_pp':(np.array(delta)*100).tolist(),
                          'passes_gate':bool(d['mean_delta_pp']>=2 and d['positive_subjects']>=6)})
    dump(OUT/'summary.json',{'models':models,'pairs':pairs,'note':'LOSO 8 users; same pretrained teacher encoder, now fine-tuned. Exploratory within-protocol effects only.'})
    lines=['# 端到端单视角重建：结果','','所有臂同折三视角教师初始化、同train/val/test；只有目标单视角输入。尚未8折齐全的配置不用于判定。','',
           '|方法|视角|seed|完成折|accuracy%|macroF1%|最差用户%|NMSE|参数|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in models:lines.append(f"|{r['mode']}|{r['view']}|{r['seed']}|{r['folds']}/8|{r['accuracy_pct']:.2f}|{r['macro_f1_pct']:.2f}|{r['worst_subject_pct']:.2f}|{r['mean_nmse']:.3f}|{r['parameters']}|")
    lines+=['','|候选|对照|Δpp|95%用户bootstrap CI|改善用户|研发门槛|','|---|---|---:|---|---:|---|']
    for r in pairs:
        lo,hi=r['ci95_subject_bootstrap_pp'];lines.append(f"|{r['candidate']}|{r['control']}|{r['mean_delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{r['positive_subjects']}/8|{r['passes_gate']}|")
    lines+=['','研发门槛需要同时胜过fixed_mask和capacity，不因NMSE更低就宣布分类有效。',
            '预训练权重/池化/学习率与旧实验不同；只对上述同条件对照的差值作机制判断。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('E2E_SUMMARY',len(models),len(pairs))

if __name__=='__main__':main()
