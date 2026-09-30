"""Post-hoc diagnostic only; never select configurations from test metrics."""
import numpy as np
import torch
from reconstruction import OUT, Completion, extract, evaluate
from pipeline import dump,split
from train import setup

def main():
    setup(2026);rows=[]
    for h in range(1,9):
        f,y,s,sha=extract(h);tr,va,te=split(y,s,h,2026)
        ck=torch.load(OUT/f'real3_LTR_s2026_h{h}.pt',map_location='cuda',weights_only=False)
        x=(f.float()-ck['mu'].cuda())/ck['sd'].cuda()
        head={k[5:]:v for k,v in ck['state'].items() if k.startswith('head.')}
        for v in range(3):
            m=Completion(992,v,'fixed_mask').cuda().eval();m.head.load_state_dict(head)
            z,_,_=evaluate(m,x,y,te)
            rows.append({'held_subject':h,'view':'LTR'[v],'unadapted_teacher_zero_accuracy':float(np.mean(z.argmax(1)==y[te]))})
        del f,x,ck,m
    dump(OUT/'unadapted_teacher_diagnostic.json',{'note':'Read-only test diagnostic. This naive baseline was trained only on three real views; do not use its gain as evidence against fixed_mask trained on the deployment view.','rows':rows})
    for v in 'LTR':print(v,'naive zero teacher',round(np.mean([r['unadapted_teacher_zero_accuracy'] for r in rows if r['view']==v])*100,3))

if __name__=='__main__':main()
