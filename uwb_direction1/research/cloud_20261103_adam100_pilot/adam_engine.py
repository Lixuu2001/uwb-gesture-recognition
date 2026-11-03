"""UNEXECUTED full Adam100 helpers, using the separate fixed native draft."""
import hashlib
from native_adam100 import np,torch,make,preprocess,train100
def semantic(value):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            h.update(str((tuple(x.shape),str(x.dtype))).encode())
            h.update(x.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x:walk(v)
        else:h.update(str(x).encode())
    walk(value);return h.hexdigest()
def raw(maps,ix):
    a=np.asarray(maps[ix])
    return np.ascontiguousarray(a.reshape(len(ix),90,3,189).transpose(0,2,1,3),dtype=np.float32)
def run(maps,loc,y,v,users,on_step,on_epoch):
    return train100(lambda pos:raw(maps,loc['train'][pos]),y,
        lambda start,end:raw(maps,loc['validation'][start:end]),v,
        users[loc['train']],users[loc['validation']],on_step,on_epoch)
def record_epoch(record,state):
    assert record['selection']=='fixed_final_epoch100_not_best_validation'
    return dict(record,model_sha256=semantic(state['model']),
        optimizer_sha256=semantic(state['optimizer']),
        order_rng_sha256=semantic(state['order_rng']),
        torch_rng_sha256=semantic(state['torch_rng']))
def atomic_checkpoint(path,state):
    tmp=path.with_suffix(path.suffix+'.tmp')
    assert not tmp.exists()
    torch.save(state,tmp);tmp.replace(path)
def evaluate(model,maps,positions):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(positions),20):
            out.append(model(preprocess(raw(maps,positions[start:start+20]))).numpy())
    a=np.concatenate(out)
    assert a.shape==(len(positions),12) and np.isfinite(a).all()
    return a
