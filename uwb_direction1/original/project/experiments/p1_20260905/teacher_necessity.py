"""Teacher necessity contrasts and read-only training-target gap audit."""
import json
import numpy as np
from scipy.special import softmax
from pipeline import BASE, RESULTS, dump, split
from followup_analysis import load, check_pair
from summarize import uncertainty

OUT=BASE/'teacher_necessity_results'
LABEL='sv_R_label_uniform'
UNIFORM='sv_R_project_uniform'


def training_gap():
    y=np.load(BASE/'prepared/y.npy');s=np.load(BASE/'prepared/subj.npy');rows=[]
    for h in range(1,9):
        tr,va,te=split(y,s,h,2026)
        with np.load(RESULTS/f'teacher_concat_s2026_h{h}_all_logits.npz') as t:
            np.testing.assert_array_equal(t['train_ids'],tr);np.testing.assert_array_equal(t['val_ids'],va)
            p=softmax(t['logits'][tr].astype('float64')/2,axis=1)
        with np.load(BASE/f'confusion_kd_results/matrix_h{h}.npz') as a:
            np.testing.assert_array_equal(a['train_ids'],tr);u=a['uniform']
        label=np.eye(12)[y[tr]];qt=.5*(p+p@u);qy=.5*(label+label@u)
        rows.append({'fold':h,'n_train':len(tr),'teacher_train_accuracy':float((p.argmax(1)==y[tr]).mean()),
                     'mean_teacher_true_class_probability_T2':float(p[np.arange(len(tr)),y[tr]].mean()),
                     'mean_raw_target_L1':float(np.abs(p-label).sum(1).mean()),
                     'mean_projected_target_L1':float(np.abs(qt-qy).sum(1).mean())})
    dump(OUT/'training_target_gap.json',{'rows':rows,'scope':'Training rows only; overlapping outer-fold train sets are not independent samples. Target similarity does not prove trained-model equivalence.'})
    return rows


def main():
    OUT.mkdir(exist_ok=True);gaps=training_gap()
    tags=[LABEL,UNIFORM,'sv_R_project_structure','sv_R_concat_kd','sv_R_concat']
    cache={(t,h):load(t,2026,h) for t in tags for h in range(1,9)}
    models=[];pairs=[]
    for tag in tags:
        rr=[cache[tag,h][0] for h in range(1,9) if cache[tag,h] is not None]
        r={'tag':tag,'folds':len(rr)}
        if len(rr)==8:r.update(accuracy_pct=float(np.mean([x['accuracy'] for x in rr])*100),macro_f1_pct=float(np.mean([x['macro_f1'] for x in rr])*100))
        models.append(r)
    for candidate,baseline in [(UNIFORM,LABEL),(LABEL,'sv_R_concat'),(LABEL,'sv_R_concat_kd')]:
        if any(cache[t,h] is None for t in (candidate,baseline) for h in range(1,9)):continue
        ds=[];accounts=[]
        for h in range(1,9):
            cm,ca=cache[candidate,h];bm,ba=cache[baseline,h];check_pair(ca,ba)
            assert cm['initial_head_hash']==bm['initial_head_hash'] and cm['parameters']==bm['parameters']
            lm=cache[LABEL,h][0]
            assert lm['teacher'] is None and not lm['config']['teacher']
            if candidate==UNIFORM:assert cm['config']['projection_matrix_sha']==lm['config']['projection_matrix_sha']
            cc=ca['pred']==ca['y'];bc=ba['pred']==ba['y']
            gain=int((cc&~bc).sum());lost=int((~cc&bc).sum())
            assert gain-lost==int(cc.sum()-bc.sum())
            accounts.append({'fold':h,'gained':gain,'lost':lost,'net':gain-lost})
            ds.append(cm['accuracy']-bm['accuracy'])
        pairs.append({'candidate':candidate,'baseline':baseline,**uncertainty(ds),'deltas_pp':(np.array(ds)*100).tolist(),'accounting':accounts})
    dump(OUT/'summary.json',{'models':models,'pairs':pairs,'training_gap':gaps,'status':'complete' if len(pairs)==3 else 'pending'})
    lines=['# 三视角教师必要性消融','','仅比较seed2026的匹配Right学生。去教师不改变网络、混淆矩阵、温度、权重或初始化。','',
           '|配置|折数|accuracy%|macro-F1%|','|---|---:|---:|---:|']
    for r in models:
        vals=[f'{r[k]:.2f}' if k in r else '待完成' for k in ['accuracy_pct','macro_f1_pct']]
        lines.append(f"|{r['tag']}|{r['folds']}/8|"+'|'.join(vals)+'|')
    lines+=['','|候选|基线|Δ pp|95%用户bootstrap CI|改善用户|','|---|---|---:|---|---:|']
    for r in pairs:
        lo,hi=r['ci95_subject_bootstrap_pp']
        lines.append(f"|{r['candidate']}|{r['baseline']}|{r['mean_delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|{r['positive_subjects']}/8|")
    lines+=['','## 原训练目标差异（按8外折均值，仅诊断）','',
        f"- 教师训练集分类准确率：{np.mean([r['teacher_train_accuracy'] for r in gaps])*100:.3f}%。",
        f"- T=2时真类平均概率：{np.mean([r['mean_teacher_true_class_probability_T2'] for r in gaps]):.4f}。",
        f"- 投影后教师与标签目标的平均L1距离：{np.mean([r['mean_projected_target_L1'] for r in gaps]):.4f}（范围0–2）。",
        '- 此诊断不代替完整训练，也不把重复外折的训练样本当独立样本。',
        '- 区间包含零不证明统计等效；需报告未发现增量或存在不确定性，不能直接说教师永远无用。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('TEACHER_NECESSITY',[(r['tag'],r['folds']) for r in models],len(pairs),'contrasts')


if __name__=='__main__':main()
