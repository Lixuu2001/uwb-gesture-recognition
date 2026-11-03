"""Full official MobileNetV2 x1 V1 transfer; no UWB fitting on import.

Canonical class AST and helpers are retained exactly. Only API usage logging is
stubbed. Paper Adam/Nesterov ambiguity and UWB adapter remain disclosed.
"""
from pathlib import Path
import ast,hashlib,json,sys,warnings,collections,itertools
from typing import Any,Callable,List,Optional,Tuple,Union
from collections.abc import Sequence
sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-source-20261102')
from native_resnet18 import np,torch,nn,preprocess
from torch import Tensor
B=Path(__file__).resolve().parent
_log_api_usage_once=lambda *a,**k:None
def load_exact(name,names):
    raw=(B/'local_author_sources'/name).read_bytes()
    expected=json.loads((B/'AUTHOR_BLOBS.json').read_text())[name]
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected
    nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert [n.name for n in nodes]==names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(B/'local_author_sources'/name),'exec'),globals())
load_exact('utils.py',['_make_ntuple'])
load_exact('_utils.py',['_make_divisible'])
load_exact('misc.py',['ConvNormActivation','Conv2dNormActivation'])
load_exact('mobilenetv2.py',['InvertedResidual','MobileNetV2'])
SEED=20261029
EPOCHS=100
BATCH=20
def make(keep_original_head=False):
    meta=json.loads((B/'WEIGHTS_SOURCE.json').read_text())
    p=B/'weights/mobilenet_v2-b0353104.pth'
    assert p.stat().st_size==meta['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    torch.manual_seed(SEED);m=MobileNetV2()
    assert sum(p.numel() for p in m.parameters())==3504872
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
    if keep_original_head:m.eval()
    else:
        torch.manual_seed(SEED);m.classifier[1]=nn.Linear(1280,12)
        m.train();assert sum(p.numel() for p in m.parameters())==2239244
    m.requires_grad_(True)
    assert all(p.requires_grad and p.device.type=='cpu' for p in m.parameters())
    assert torch.get_num_threads()==4 and len(m.features)==19
    assert m.classifier[0].p==.2
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
