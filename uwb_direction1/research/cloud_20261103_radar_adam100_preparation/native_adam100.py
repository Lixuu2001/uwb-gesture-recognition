"""UNEXECUTED draft of a separately registered, published-recipe transfer trial.

Never changes or resumes the current SGDM40 study. Instantiate from the original
ImageNet checkpoint for any future trial. Torch Adam beta1=.9 is a declared
interpretation: the paper's simultaneous reference to Adam and Nesterov is
ambiguous; this is not a claim of exact author-code equivalence.
"""
import sys,hashlib
sys.path.insert(0,'/workspace/uwb-shufflenetv2-x1-source-20261103')
from native_shufflenet import np,torch,preprocess,make as full_native_make
SEED=20261029
EPOCHS=100
BATCH=20
def make():
    m=full_native_make()
    assert sum(p.numel() for p in m.parameters())==1265904
    assert all(p.requires_grad and p.device.type=='cpu' for p in m.parameters())
    assert torch.get_num_threads()==4
    return m
def optimizer(m):
    return torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),
        eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False,fused=False,
        maximize=False)
def validate_roles(training_users,validation_users):
    a=np.asarray(training_users);b=np.asarray(validation_users)
    assert a.shape==(4700,) and b.shape==(1000,)
    assert set(np.unique(a).tolist())=={3,4,5,6}
    assert all(int((a==u).sum())==n for u,n in [(3,1200),(4,1200),(5,1200),(6,1100)])
    assert np.all(b==2)
def step(m,opt,raw,labels):
    assert m.training and all(p.requires_grad for p in m.parameters())
    opt.zero_grad(set_to_none=True)
    logits=m(preprocess(raw))
    loss=torch.nn.functional.cross_entropy(logits,torch.as_tensor(labels,dtype=torch.long))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    return float(loss.detach())
def evaluate(m,validation_batch,labels):
    m.eval();chunks=[]
    with torch.inference_mode():
        for start in range(0,1000,BATCH):
            chunks.append(m(preprocess(validation_batch(start,start+BATCH))).numpy())
    scores=np.concatenate(chunks)
    assert scores.shape==(1000,12) and np.isfinite(scores).all()
    assert labels.shape==(1000,) and set(np.unique(labels).tolist())==set(range(10))
    pred=scores.argmax(1)
    macro=float(np.mean([np.mean(pred[labels==k]==k) for k in range(10)]))
    return scores,macro
def train100(training_batch,training_labels,validation_batch,validation_labels,
             training_users,validation_users,on_step=None,on_epoch=None):
    """No data loading and no diagnostic argument. Hooks must save synchronously.

    A future frozen study must verify sample IDs/data hashes and register its
    resource supervisor, full independent replay and private recovery separately.
    Validation is logged, not used to replace the paper's fixed epoch100 endpoint.
    """
    validate_roles(training_users,validation_users)
    y=np.asarray(training_labels);v=np.asarray(validation_labels)
    assert y.shape==(4700,) and set(np.unique(y).tolist())==set(range(12))
    assert v.shape==(1000,) and set(np.unique(v).tolist())==set(range(10))
    assert callable(training_batch) and callable(validation_batch)
    m=make();opt=optimizer(m);g=torch.Generator().manual_seed(SEED);updates=0
    for epoch in range(1,EPOCHS+1):
        m.train();order=torch.randperm(4700,generator=g).numpy()
        for offset in range(0,4700,BATCH):
            positions=order[offset:offset+BATCH]
            assert len(positions)==20
            loss=step(m,opt,training_batch(positions),y[positions]);updates+=1
            if on_step is not None:
                on_step(dict(epoch=epoch,offset=offset,update=updates,loss=loss,
                    order_sha256=hashlib.sha256(positions.tobytes()).hexdigest()))
        scores,macro=evaluate(m,validation_batch,v)
        state=dict(model=m.state_dict(),optimizer=opt.state_dict(),
                   order_rng=g.get_state(),torch_rng=torch.get_rng_state(),
                   epoch=epoch,updates=updates)
        if on_epoch is not None:
            on_epoch(dict(epoch=epoch,updates=updates,validation_macro=macro,
                selection='fixed_final_epoch100_not_best_validation',
                diagnostic_labels_read=False,users7_8_labels_used=False),
                scores,state)
    assert updates==23500
    return state
