"""Teacher-free matched soft-target control. No multiview inputs or logits."""
import argparse
import gc
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, confusion_matrix
from pipeline import BASE, RESULTS, split, digest, dump
from train import Classifier, setup, load_gpu, predict, digest_state

TAG = 'sv_R_label_uniform'


def label_target(labels, matrix):
    onehot = F.one_hot(labels, num_classes=matrix.shape[0]).float()
    return .5 * (onehot + onehot @ matrix.detach().float())


def label_loss(student, labels, matrix, temp=2):
    return F.kl_div(F.log_softmax(student.float()/temp, dim=-1),
                    label_target(labels, matrix), reduction='batchmean') * temp**2


def run(fold, data):
    stem = RESULTS / f'{TAG}_s2026_h{fold}'
    if stem.with_suffix('.json').exists(): print('SKIP', fold, flush=True); return
    y = np.load(BASE/'prepared/y.npy'); subjects = np.load(BASE/'prepared/subj.npy')
    tr, va, te = split(y, subjects, fold, 2026)
    matrix_path = BASE / f'confusion_kd_results/matrix_h{fold}.npz'
    metadata = json.loads(matrix_path.with_suffix('.json').read_text())
    assert metadata['matrix_sha'] == digest(matrix_path)
    assert not metadata['validation_used_for_projection'] and metadata['warnings'] == 0
    with np.load(matrix_path) as a:
        for key, ids in [('train_ids',tr),('val_ids',va),('test_ids',te)]: np.testing.assert_array_equal(a[key],ids)
        matrix = torch.tensor(a['uniform'], dtype=torch.float32, device='cuda')
    setup(2026)
    model = Classifier([2], 'concat', 'mean').cuda()
    init = digest_state(model.head)
    config = {'tag':TAG,'seed':2026,'folds':[fold],'mode':'concat','views':[2],'readout':'mean',
              'aux':'none','teacher':'','epochs':40,'batch':128,'microbatch':32,'kd_weight':.5,
              'soft_target':'training_onehot_uniform_projection','temperature':2,'projection_mix':.5,
              'projection_matrix_sha':digest(matrix_path),'wrapper_sha':digest(__file__)}
    opt = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    yy = torch.as_tensor(y, device='cuda'); tr_gpu = torch.tensor(tr, device='cuda')
    order = torch.Generator(device='cpu').manual_seed(2026+fold*1000)
    torch.manual_seed(2026+100000); torch.cuda.manual_seed_all(2026+100000)
    best = float('inf'); best_ep=0; history=[]; start=time.time()
    for epoch in range(1,41):
        model.train(); total=0.; correct=0; count=0; tic=time.time()
        ids=tr_gpu[torch.randperm(len(tr),generator=order).to('cuda')]
        for begin in range(0,len(ids),128):
            batch=ids[begin:begin+128]; opt.zero_grad(set_to_none=True)
            for mb in batch.split(32):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z,_=model(data[mb])
                    loss=F.cross_entropy(z,yy[mb])+.5*label_loss(z,yy[mb],matrix)
                (loss*len(mb)/len(batch)).backward()
                total+=float(loss.detach())*len(mb); correct+=int((z.argmax(-1)==yy[mb]).sum()); count+=len(mb)
            opt.step()
        vl=predict(model,data,va,32)
        ce=float(F.cross_entropy(torch.from_numpy(vl),torch.from_numpy(y[va])))
        row={'epoch':epoch,'train_loss':total/count,'train_acc':correct/count,'val_ce':ce,
             'val_acc':float((vl.argmax(-1)==y[va]).mean()),'seconds':time.time()-tic}
        history.append(row)
        if ce<best:
            best=ce; best_ep=epoch
            torch.save({'state':model.state_dict(),'args':config,'fold':fold,'epoch':epoch},stem.with_suffix('.pt'))
        print(stem.name,json.dumps(row),flush=True)
        if epoch>=12 and epoch-best_ep>=8:break
    ck=torch.load(stem.with_suffix('.pt'),map_location='cuda',weights_only=False)
    model.load_state_dict(ck['state']);del ck
    logits=predict(model,data,te,32); pred=logits.argmax(-1)
    vl=predict(model,data,va,32)
    np.savez_compressed(str(stem)+'_predictions.npz',ids=te,y=y[te],pred=pred,logits=logits,
                        train_ids=tr,val_ids=va,val_y=y[va],val_logits=vl)
    r={'tag':TAG,'seed':2026,'held_subject':fold,'mode':'concat','views':[2],'readout':'mean',
       'aux':'none','teacher':None,'config':config,'parameters':sum(p.numel() for p in model.parameters()),
       'initial_head_hash':init,'n_train':len(tr),'n_val':len(va),'n_test':len(te),
       'best_epoch':best_ep,'best_val_ce':best,'accuracy':float((pred==y[te]).mean()),
       'macro_f1':float(f1_score(y[te],pred,labels=np.unique(y[te]),average='macro',zero_division=0)),
       'evaluated_classes':np.unique(y[te]).tolist(),'confusion':confusion_matrix(y[te],pred,labels=np.arange(12)).tolist(),
       'history':history,'elapsed_seconds':time.time()-start,'train_py_sha256':digest(__file__),
       'original_loop_source_sha':digest(BASE/'train.py'),'pipeline_sha256':digest(BASE/'pipeline.py')}
    dump(stem.with_suffix('.json'),r)
    print('COMPLETE',stem.name,r['accuracy'],flush=True)
    del model,opt,matrix;gc.collect();torch.cuda.empty_cache()


def smoke():
    setup(2026);model=Classifier([2],'concat').cuda()
    z,_=model(torch.randn(4,1,90,378,device='cuda'))
    labels=torch.tensor([0,3,5,11],device='cuda'); matrix=torch.eye(12,device='cuda')*.7+.025
    loss=F.cross_entropy(z,labels)+.5*label_loss(z,labels,matrix)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.temporal.proj.weight.grad.abs().sum()>0
    print('LABEL_UNIFORM_SMOKE_OK',float(loss.detach()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folds',type=int,nargs='+',default=list(range(1,9)))
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args();setup(2026)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if args.smoke:smoke()
    else:
        data=load_gpu('concat',[2])
        for fold in args.folds:run(fold,data)
