"""Validation-MSE-selected linear reconstruction on fixed fold-local features."""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score
from reconstruction import OUT,extract,fit_scale,Completion
from pipeline import dump,split,digest
from train import setup

@torch.no_grad()
def main():
    setup(2026);rows=[]
    for h in range(1,9):
        ft,y,s,sha=extract(h);tr,va,te=split(y,s,h,2026);mu,sd=fit_scale(ft,tr);z=(ft.float()-mu)/sd
        ck=torch.load(OUT/f'real3_LTR_s2026_h{h}.pt',map_location='cuda',weights_only=False)
        head=Completion(992,0,'real3').cuda().eval();head.load_state_dict(ck['state'])
        for v in range(3):
            dest=OUT/f'ridge_{"LTR"[v]}_s2026_h{h}'
            if dest.with_suffix('.json').exists():rows.append(json.loads(dest.with_suffix('.json').read_text()));continue
            miss=[j for j in range(3) if j!=v];x=z[:,v].double();t=z[:,miss].flatten(1).double()
            xx=x[tr].T@x[tr];xt=x[tr].T@t[tr];eye=torch.eye(992,device='cuda',dtype=torch.float64)
            best=float('inf');bestw=None;bestalpha=None;val_scores=[]
            for a in [.001,.01,.1,1.]:
                w=torch.linalg.solve(xx+len(tr)*a*eye,xt);valmse=float((x[va]@w-t[va]).square().mean())
                val_scores.append({'alpha':a,'val_mse':valmse})
                if valmse<best:best=valmse;bestw=w;bestalpha=a
            predft=(x[te]@bestw).float().reshape(-1,2,992);parts=z[te].clone();parts[:,miss]=predft
            logits=head(parts)[0].cpu().numpy();pred=logits.argmax(1)
            mse=float((predft-z[te][:,miss]).square().mean());nmse=mse/float(z[te][:,miss].square().mean())
            torch.save({'weight':bestw.float().cpu(),'mu':mu.cpu(),'sd':sd.cpu(),'view':v,'teacher_sha':sha},dest.with_suffix('.pt'))
            np.savez_compressed(str(dest)+'_predictions.npz',ids=te,y=y[te],pred=pred,logits=logits,train_ids=tr,val_ids=va)
            r={'tag':dest.stem,'fold':h,'view':v,'mode':'ridge','seed':2026,'accuracy':float(np.mean(pred==y[te])),
               'macro_f1':float(f1_score(y[te],pred,labels=np.unique(y[te]),average='macro',zero_division=0)),
               'missing_nmse':nmse,'parameters':992*1984+765196,'alpha':bestalpha,'val_scores':val_scores,'teacher_sha':sha,'code_sha':digest(__file__)}
            dump(dest.with_suffix('.json'),r);rows.append(r)
            print('RIDGE',h,'LTR'[v],'acc',round(r['accuracy']*100,3),'nmse',round(nmse,3),flush=True)
        del ft,z,ck,head,x,t,xx,xt,bestw;torch.cuda.empty_cache()
    dump(OUT/'ridge_summary.json',{'note':'Frozen features and frozen three-view head; alpha selected only by validation MSE.','rows':rows})
    for v in range(3):
        rr=[r for r in rows if r['view']==v]
        print('RIDGE_MEAN','LTR'[v],'acc',np.mean([r['accuracy'] for r in rr])*100,'nmse',np.mean([r['missing_nmse'] for r in rr]),flush=True)

if __name__=='__main__':main()
