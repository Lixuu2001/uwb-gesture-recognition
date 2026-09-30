"""Matched closed-set training, frozen independent KD, auditable LOSO outputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, copy, gc, json, time, random, warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pipeline import BASE, RESULTS, dump, split, digest
from config import Config
from uwb_osr.model import SharedDualPathEncoder, ClosedSetHead, reduce_tokens
warnings.filterwarnings('ignore',message='enable_nested_tensor')

def setup(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False

class Classifier(nn.Module):
    def __init__(self,views,mode,readout='mean',aux=0):
        super().__init__()
        self.cfg=Config(views=['Left','Top','Right'][:len(views)],pool='flatten',
                        use_fusion=False,doppler='concat' if mode in ('concat','duplicate') else 'off')
        self.encoder=SharedDualPathEncoder(self.cfg)
        self.readout=readout
        self.head=ClosedSetHead(self.encoder.feat_dim*(len(views) if readout=='concat' else 1),12)
        self.aux=nn.Linear(self.encoder.feat_dim,aux*90) if aux else None
        self.aux_count=aux
    def forward(self,x):
        _,t,s=self.encoder(x)
        f=torch.cat([reduce_tokens(t,'flatten',2),reduce_tokens(s,'flatten',2)],-1)
        z=f.flatten(1) if self.readout=='concat' else f.mean(1)
        a=None if self.aux is None else self.aux(f.mean(1)).reshape(-1,self.aux_count,90)
        return self.head(z),a

def kd_loss(student,teacher,temp=2):
    return F.kl_div(F.log_softmax(student.float()/temp,-1),
                    F.softmax(teacher.detach().float()/temp,-1),reduction='batchmean')*temp**2

def load_gpu(mode,views):
    # Copy one small chunk at a time; never materialize all 2GB maps in host RAM.
    first=np.load(BASE/'prepared'/f'{"signed" if mode=="signed" else "only" if mode=="only" else "off"}.npy',mmap_mode='r')
    mult=2 if mode in ('concat','duplicate') else 1
    out=torch.empty((len(first),len(views),90,189*mult),dtype=torch.float16,device='cuda')
    second=np.load(BASE/'prepared/only.npy',mmap_mode='r') if mode=='concat' else first
    for i in range(0,len(first),128):
        a=np.array(first[i:i+128,views],copy=True)
        if mult==2:a=np.concatenate([a,np.array(second[i:i+128,views],copy=True)],-1)
        out[i:i+len(a)].copy_(torch.from_numpy(a))
    return out

@torch.no_grad()
def predict(model,data,ids,batch=128):
    model.eval();out=[]
    for i in range(0,len(ids),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):z,_=model(data[ids[i:i+batch]])
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)

def run(args,data,fold):
    name=f'{args.tag}_s{args.seed}_h{fold}'
    RESULTS.mkdir(exist_ok=True)
    dest=RESULTS/name
    if dest.with_suffix('.json').exists():
        print('SKIP',name,flush=True);return
    y=np.load(BASE/'prepared/y.npy');subjects=np.load(BASE/'prepared/subj.npy')
    tr,va,te=split(y,subjects,fold,args.seed)
    setup(args.seed)
    aux_count=0 if args.aux=='none' else 1 if args.aux=='self' else 3
    model=Classifier(args.views,args.mode,args.readout,aux_count).cuda()
    initial_head_hash=digest_state(model.head)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    yy=torch.as_tensor(y,device='cuda')
    tlog=None; teacher_provenance=None
    if args.teacher:
        candidates=[RESULTS/f'{tag}_s{args.seed}_h{fold}' for tag in args.teacher.split(',')]
        source=min(candidates,key=lambda p:json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))['best_val_ce'])
        meta=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
        if meta['held_subject']!=fold or meta['seed']!=args.seed:raise ValueError('Teacher fold mismatch')
        teacher_file=Path(str(source)+'_all_logits.npz')
        archive=np.load(teacher_file)
        np.testing.assert_array_equal(archive['y'],y)
        np.testing.assert_array_equal(archive['train_ids'],tr)
        np.testing.assert_array_equal(archive['val_ids'],va)
        tlog=torch.tensor(archive['logits'],device='cuda')
        teacher_provenance={'source':str(source),'logits_sha256':digest(teacher_file)}
    motion=None;quality=None
    if aux_count:
        raw=np.load(BASE/'prepared/motion.npy')
        raw=raw[:,args.views] if aux_count==1 else raw
        # Standardization fitted on gradient-training indices only.
        mu=raw[tr].mean((0,2),keepdims=True);sd=raw[tr].std((0,2),keepdims=True)+1e-6
        motion=torch.tensor((raw-mu)/sd,device='cuda')
        quality=torch.tensor(np.load(BASE/'prepared/quality.npy'),device='cuda') if args.aux=='quality' else None
    best=float('inf');best_ep=0;history=[];start=time.time()
    tr_gpu=torch.tensor(tr,device='cuda')
    # Separate permutation generator makes order identical CE/KD/aux despite model RNG.
    order_rng=torch.Generator(device='cpu').manual_seed(args.seed+fold*1000)
    torch.manual_seed(args.seed+100000);torch.cuda.manual_seed_all(args.seed+100000)
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;correct=0;count=0;tic=time.time()
        ids=tr_gpu[torch.randperm(len(tr),generator=order_rng).to('cuda')]
        for begin in range(0,len(ids),args.batch):
            batch_ids=ids[begin:begin+args.batch];opt.zero_grad(set_to_none=True)
            # Microbatch only reduces activation memory; same effective batch for all arms.
            for mb in batch_ids.split(args.microbatch):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z,a=model(data[mb]);loss=F.cross_entropy(z,yy[mb])
                    if tlog is not None:loss=loss+args.kd_weight*kd_loss(z,tlog[mb])
                    if a is not None:
                        err=(a.float()-motion[mb]).square()
                        if quality is not None:err=err*quality[mb]
                        loss=loss+0.1*err.mean()
                (loss*len(mb)/len(batch_ids)).backward()
                total+=float(loss.detach())*len(mb);correct+=int((z.argmax(-1)==yy[mb]).sum());count+=len(mb)
            opt.step()
        vl=predict(model,data,va,args.microbatch)
        vloss=float(F.cross_entropy(torch.from_numpy(vl),torch.from_numpy(y[va])))
        vacc=float((vl.argmax(-1)==y[va]).mean())
        row={'epoch':epoch,'train_loss':total/count,'train_acc':correct/count,'val_ce':vloss,'val_acc':vacc,'seconds':time.time()-tic}
        history.append(row)
        if vloss<best:
            best=vloss;best_ep=epoch
            torch.save({'state':model.state_dict(),'args':vars(args),'fold':fold,'epoch':epoch},dest.with_suffix('.pt'))
        print(name,json.dumps(row),flush=True)
        if epoch>=12 and epoch-best_ep>=8:break
    ck=torch.load(dest.with_suffix('.pt'),map_location='cuda',weights_only=False)
    model.load_state_dict(ck['state']);del ck
    logits=predict(model,data,te,args.microbatch);pred=logits.argmax(-1)
    val_logits=predict(model,data,va,args.microbatch)
    np.savez_compressed(str(dest)+'_predictions.npz',ids=te,y=y[te],pred=pred,logits=logits,val_ids=va,val_y=y[va],val_logits=val_logits,train_ids=tr)
    if args.save_teacher:
        all_logits=predict(model,data,np.arange(len(y)),args.microbatch)
        np.savez_compressed(str(dest)+'_all_logits.npz',logits=all_logits,y=y,train_ids=tr,val_ids=va)
    result={'tag':args.tag,'held_subject':fold,'seed':args.seed,'mode':args.mode,'views':args.views,
            'readout':args.readout,'aux':args.aux,'teacher':teacher_provenance,'config':vars(args),
            'parameters':sum(p.numel() for p in model.parameters()),'initial_head_hash':initial_head_hash,
            'n_train':len(tr),'n_val':len(va),'n_test':len(te),'best_epoch':best_ep,'best_val_ce':best,
            'accuracy':accuracy_score(y[te],pred),'macro_f1':f1_score(y[te],pred,average='macro',labels=np.unique(y[te]),zero_division=0),
            'evaluated_classes':np.unique(y[te]).tolist(),
            'confusion':confusion_matrix(y[te],pred,labels=np.arange(12)).tolist(),
            'history':history,'elapsed_seconds':time.time()-start,
            'train_py_sha256':digest(__file__),'pipeline_sha256':digest(BASE/'pipeline.py')}
    dump(dest.with_suffix('.json'),result)
    print('COMPLETE',name,'acc',result['accuracy'],'F1',result['macro_f1'],'sec',result['elapsed_seconds'],flush=True)
    del model,opt

def digest_state(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['off','only','concat','signed','duplicate'],default='off')
    ap.add_argument('--views',type=int,nargs='+',default=[0]);ap.add_argument('--readout',choices=['mean','concat'],default='mean')
    ap.add_argument('--folds',type=int,nargs='+',default=list(range(1,9)))
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--batch',type=int,default=128);ap.add_argument('--microbatch',type=int,default=32)
    ap.add_argument('--teacher',default='');ap.add_argument('--save-teacher',action='store_true')
    ap.add_argument('--kd-weight',type=float,default=.5)
    ap.add_argument('--aux',choices=['none','self','joint','quality'],default='none')
    args=ap.parse_args();setup(args.seed)
    if os.name=='nt':
        import ctypes;ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    data=load_gpu(args.mode,args.views)
    print('START',vars(args),'gpu_data_GB',data.numel()*data.element_size()/1e9,flush=True)
    for fold in args.folds:
        run(args,data,fold);gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
