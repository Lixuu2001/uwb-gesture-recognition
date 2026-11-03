"""Full synthetic batches only. No UWB data, labels or model selection."""
from pathlib import Path
import sys,time,json,hashlib
B=Path(__file__).resolve().parent
def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['timemil','shuffle']
    p=json.loads((B/'SOURCE6_METHOD_LOCK.json').read_text());arm=sys.argv[1]
    sys.path.insert(0,p['TimeMIL' if arm=='timemil' else 'ShuffleNet']['source'])
    if arm=='timemil':
        from adapter import torch,make_model,StableLookahead
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        m=make_model('timemil',p['seed']);opt=StableLookahead(m.parameters(),lr=.001,weight_decay=.0001)
    else:
        from native_adam100 import torch,np,make,optimizer,step
        m=make();opt=optimizer(m)
    rng=np.random.default_rng(p['seed']);times=[]
    for i in range(6):
        if arm=='timemil':
            x=torch.from_numpy(rng.normal(size=(64,90,567)).astype(np.float32));target=torch.nn.functional.one_hot(torch.arange(64)%12,12).float()
            start=time.monotonic();m.train();opt.zero_grad();logits=m(x,warmup=False);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step()
        else:
            x=rng.normal(size=(20,3,90,189)).astype(np.float32);labels=np.arange(20)%12
            start=time.monotonic();step(m,opt,x,labels)
        times.append(time.monotonic()-start)
        print(json.dumps(dict(arm=arm,update=i+1,seconds=times[-1],synthetic_only=True)),flush=True)
    result=dict(arm=arm,times=times,threads=torch.get_num_threads(),updates=6,synthetic_only=True,new_UWB_fit=False)
    (B/'profile01'/(arm+'_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
