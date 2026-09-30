"""Post-hoc accounting of transferable successes and negative transfer.

Uses test labels only for evaluation, never to fit a gate or choose a teacher.
"""
import json
from pathlib import Path
import numpy as np
from pipeline import RESULTS,dump

def accounting(y,b,t,k):
    bc=b==y;tc=t==y;kc=k==y
    opportunity=tc&~bc
    gained=(~bc)&kc;lost=bc&~kc
    assert int(gained.sum()-lost.sum())==int(kc.sum()-bc.sum())
    return {'n':len(y),'baseline_correct':int(bc.sum()),'teacher_correct':int(tc.sum()),
            'kd_correct':int(kc.sum()),'teacher_opportunities':int(opportunity.sum()),
            'captured_opportunities':int((opportunity&kc).sum()),
            'new_correct':int(gained.sum()),'lost_correct':int(lost.sum()),
            'teacher_wrong_and_student_damaged':int((~tc&bc&~kc).sum()),
            'both_wrong_before_but_kd_correct':int((~tc&~bc&kc).sum()),
            'net_delta_pp':float((kc.mean()-bc.mean())*100)}

def main():
    rows=[]
    for p in RESULTS.glob('sv_*_kd_s*_h*.json'):
        meta=json.loads(p.read_text(encoding='utf-8'));stem=p.stem
        # The KD tag may be ``sv_R_concat_kd`` (phase+amplitude) or
        # ``sv_R_kd`` (amplitude only).  In both cases the matched baseline is
        # the same-view ``sv_<view>_off`` arm; replacing the literal suffix
        # would incorrectly look for ``sv_R_concat_off``.
        parts=meta['tag'].split('_')
        if len(parts) < 3 or parts[0] != 'sv' or parts[-1] != 'kd':
            raise ValueError(f'unexpected KD tag: {meta["tag"]}')
        baseline_tag='_'.join(parts[:2]+['off'])
        bstem=stem.replace(meta['tag']+'_s', baseline_tag+'_s', 1)
        base=RESULTS/(bstem+'_predictions.npz');kd=RESULTS/(stem+'_predictions.npz')
        b=np.load(base);k=np.load(kd)
        np.testing.assert_array_equal(b['ids'],k['ids']);np.testing.assert_array_equal(b['y'],k['y'])
        teacher=np.load(meta['teacher']['source']+'_all_logits.npz')
        t=teacher['logits'][b['ids']].argmax(1)
        row={'tag':meta['tag'],'seed':meta['seed'],'held_subject':meta['held_subject'],
             'teacher_source':meta['teacher']['source'],**accounting(b['y'],b['pred'],t,k['pred'])}
        row['by_class']={int(c):accounting(b['y'][b['y']==c],b['pred'][b['y']==c],t[b['y']==c],k['pred'][b['y']==c]) for c in np.unique(b['y'])}
        rows.append(row)
    dump(RESULTS/'transfer_accounting.json',{'interpretation':'Teacher opportunity is teacher correct AND baseline wrong. Capturing it is empirical knowledge transfer, not proof of recovery of unobservable physical state.','rows':rows})
    print('TRANSFER_ANALYSIS',len(rows),'folds')

if __name__=='__main__':main()
