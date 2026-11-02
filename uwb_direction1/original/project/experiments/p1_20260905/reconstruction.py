"""Fold-local frozen-encoder 1->3 latent completion; no test target input."""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, confusion_matrix
from pipeline import BASE, RESULTS, dump, digest, split
from train import Classifier, setup, digest_state

OUT=BASE/'reconstruction_results'

def fit_scale(x,ids):
    a=x[ids].float()
    return a.mean(0,keepdim=True),a.std(0,unbiased=False,keepdim=True).clamp_min(.05)

def reconstruction_loss(parts,targets,view):
    missing=[v for v in range(3) if v!=view]
    return F.mse_loss(parts[:,missing].float(),targets[:,missing].detach().float())

class Completion(nn.Module):
    def __init__(self,dim,view,mode):
        super().__init__();self.dim=dim;self.view=view;self.mode=mode
        self.head=nn.Sequential(nn.Linear(3*dim,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,12))
        self.decoder=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,2*dim)) if mode in ('joint','capacity','pure_recon') else None
    def forward(self,x):
        if self.mode=='real3':
            if x.ndim!=3 or x.shape[1:]!=(3,self.dim):raise ValueError('real3 expects all views')
            parts=x
        else:
            if x.ndim!=2 or x.shape[1]!=self.dim:raise ValueError('deployment accepts ONLY one view')
            missing=x.new_zeros((len(x),2,self.dim)) if self.decoder is None else self.decoder(x).reshape(-1,2,self.dim)
            rows=[];j=0
            for v in range(3):
                if v==self.view:rows.append(x)
                else:rows.append(missing[:,j]);j+=1
            parts=torch.stack(rows,1)
        return self.head(parts.flatten(1)),parts

@torch.no_grad()
def compact(enc,x):
    _,t,s=enc(x)
    b,v,_,d=t.shape
    return torch.cat([t.reshape(b,v,10,9,d).mean(3).flatten(2),s.reshape(b,v,21,9,d).mean(3).flatten(2)],-1).float()

def extract(fold):
    dest=OUT/f'features_h{fold}.npz'
    teacher=RESULTS/f'teacher_concat_s2026_h{fold}.pt'
    sha=digest(teacher)
    y=np.load(BASE/'prepared/y.npy');s=np.load(BASE/'prepared/subj.npy')
    tr,va,te=split(y,s,fold,2026)
    archive=np.load(RESULTS/f'teacher_concat_s2026_h{fold}_all_logits.npz')
    np.testing.assert_array_equal(archive['train_ids'],tr);np.testing.assert_array_equal(archive['val_ids'],va)
    if dest.exists():
        a=np.load(dest);assert str(a['teacher_sha'])==sha
        for k,ids in [('train_ids',tr),('val_ids',va),('test_ids',te)]:np.testing.assert_array_equal(a[k],ids)
        return torch.tensor(a['features'],device='cuda'),y,s,sha
    setup(2026);model=Classifier([0,1,2],'off','concat').cuda().eval()
    ck=torch.load(teacher,map_location='cpu',weights_only=False)
    assert ck['fold']==fold and ck['args']['seed']==2026
    model.load_state_dict(ck['state']);del ck
    assert model.encoder.temporal.xmv==0 and model.encoder.spatial.xmv==0
    model.requires_grad_(False)
    x=np.load(BASE/'prepared/off.npy',mmap_mode='r')
    out=np.zeros((len(y),3,992),np.float16)
    # Compute only quality-retained rows; each feature is a function of its view only.
    ids=np.sort(np.concatenate([tr,va,te]))
    with torch.no_grad():
        probe=torch.tensor(np.array(x[ids[:2]],copy=True),device='cuda').float()
        full=compact(model.encoder,probe)
        for v in range(3):
            part=compact(model.encoder,probe[:,v:v+1])
            torch.testing.assert_close(full[:,v],part[:,0],rtol=1e-4,atol=1e-4)
        for start in range(0,len(ids),64):
            j=ids[start:start+64];xx=torch.tensor(np.array(x[j],copy=True),device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):z=compact(model.encoder,xx)
            out[j]=z.cpu().numpy()
    np.savez_compressed(dest,features=out,teacher_sha=sha,train_ids=tr,val_ids=va,test_ids=te)
    del model;gc.collect();torch.cuda.empty_cache()
    print('EXTRACTED',fold,flush=True)
    return torch.tensor(out,device='cuda'),y,s,sha

@torch.no_grad()
def evaluate(m,x,yy,ids,batch=256):
    m.eval();logits=[];err=torch.zeros(3,device=x.device);den=err.clone();n=0
    for i in range(0,len(ids),batch):
        ix=ids[i:i+batch];xx=x[ix]
        z,parts=m(xx if m.mode=='real3' else xx[:,m.view])
        logits.append(z.cpu().numpy())
        err+=(parts-xx).square().sum((0,2));den+=xx.square().sum((0,2));n+=len(ix)
    logits=np.concatenate(logits)
    return logits,err.cpu().numpy()/(n*x.shape[-1]),den.cpu().numpy()/(n*x.shape[-1])

def train_arm(fold,view,mode,features,y,s,sha,seed,epochs):
    tag=f'{mode}_{"LTR" if mode=="real3" else "LTR"[view]}_s{seed}_h{fold}'
    dest=OUT/tag
    if dest.with_suffix('.json').exists():print('SKIP',tag,flush=True);return
    tr,va,te=split(y,s,fold,2026) # seed changes readout RNG only: fixed pretrained encoder/splits
    mu,sd=fit_scale(features,tr);x=(features.float()-mu)/sd;yy=torch.tensor(y,device='cuda')
    setup(seed);m=Completion(992,view,mode).cuda()
    init=digest_state(m)
    if mode=='pure_recon':
        ck=torch.load(OUT/f'real3_LTR_s{seed}_h{fold}.pt',map_location='cuda',weights_only=False)
        m.head.load_state_dict({k[5:]:v for k,v in ck['state'].items() if k.startswith('head.')})
        m.head.requires_grad_(False);del ck
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    order=torch.Generator().manual_seed(seed+fold*1000)
    torch.manual_seed(seed+100000);torch.cuda.manual_seed_all(seed+100000)
    best=float('inf');best_ep=0;history=[];tic=time.time()
    for ep in range(1,epochs+1):
        m.train();m.head.eval() if mode=='pure_recon' else None
        ids=tr[torch.randperm(len(tr),generator=order).numpy()]
        total=0.
        for i in range(0,len(ids),256):
            j=ids[i:i+256];xx=x[j];opt.zero_grad(set_to_none=True)
            z,parts=m(xx if mode=='real3' else xx[:,view])
            rec=reconstruction_loss(parts,xx,view)
            loss=rec if mode=='pure_recon' else F.cross_entropy(z,yy[j])+(0.1*rec if mode=='joint' else 0)
            loss.backward();opt.step();total+=loss.item()*len(j)
        vl,ve,vd=evaluate(m,x,y,va)
        ce=F.cross_entropy(torch.tensor(vl),torch.tensor(y[va])).item()
        score=float(np.mean(ve[[v for v in range(3) if v!=view]])) if mode=='pure_recon' else ce
        history.append({'epoch':ep,'loss':total/len(tr),'val_ce':ce,'val_selection':score})
        if score<best:
            best=score;best_ep=ep
            torch.save({'state':m.state_dict(),'mode':mode,'view':view,'mu':mu.cpu(),'sd':sd.cpu(),'teacher_sha':sha},dest.with_suffix('.pt'))
        if ep>=12 and ep-best_ep>=8:break
    ck=torch.load(dest.with_suffix('.pt'),map_location='cuda',weights_only=False);m.load_state_dict(ck['state']);del ck
    logits,err,den=evaluate(m,x,y,te);pred=logits.argmax(1)
    val_logits,_,_=evaluate(m,x,y,va)
    np.savez_compressed(str(dest)+'_predictions.npz',ids=te,y=y[te],pred=pred,logits=logits,train_ids=tr,val_ids=va,val_logits=val_logits)
    missing=[v for v in range(3) if v!=view]
    res={'tag':tag,'mode':mode,'view':view,'fold':fold,'seed':seed,'encoder_seed':2026,'accuracy':float(np.mean(pred==y[te])),
         'macro_f1':float(f1_score(y[te],pred,labels=np.unique(y[te]),average='macro',zero_division=0)),
         'confusion':confusion_matrix(y[te],pred,labels=np.arange(12)).tolist(),
         'mse_by_view':err.tolist(),'zero_baseline_mse_by_view':den.tolist(),
         'missing_nmse':float(err[missing].sum()/den[missing].sum()),'best_epoch':best_ep,'best_validation':best,
         'parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
         'initial_hash':init,'history':history,'n_test':len(te),'seconds':time.time()-tic,'teacher_sha':sha,'code_sha':digest(__file__)}
    dump(dest.with_suffix('.json'),res)
    print('COMPLETE',tag,'acc',round(res['accuracy']*100,3),'nmse',round(res['missing_nmse'],3),'seconds',round(res['seconds'],1),flush=True)
    del m,opt,x;gc.collect();torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',nargs='+',type=int,default=list(range(1,9)))
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--views',nargs='+',type=int,default=[0,1,2]);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);setup(a.seed)
    import ctypes;ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    for h in a.folds:
        ft,y,s,sha=extract(h)
        train_arm(h,0,'real3',ft,y,s,sha,a.seed,a.epochs)
        for v in a.views:
            for mode in ('fixed_mask','capacity','joint','pure_recon'):
                train_arm(h,v,mode,ft,y,s,sha,a.seed,a.epochs)
        del ft;gc.collect();torch.cuda.empty_cache()
    print('RECONSTRUCTION_COMPLETE',flush=True)

if __name__=='__main__':main()
