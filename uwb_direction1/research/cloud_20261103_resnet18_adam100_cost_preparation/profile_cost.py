"""UNEXECUTED synthetic B20 Adam100 screen; never reads new UWB arrays/labels."""
from pathlib import Path
import sys,time,json,resource,shutil,gc,hashlib
from activation_guard import B,ready,sha
from cost_math import estimate
def semantic(value,torch):
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
def main():
    p=ready()
    out=B/'profile01';out.mkdir(exist_ok=False)
    (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(
        policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),
        full_B20_shape=[20,3,90,189],new_UWB_values_or_labels_fit=False),indent=2)+'\n')
    module=(B/p['native_module_directory']).resolve()
    sys.path.insert(0,str(module))
    from native_resnet18_adam100 import np,torch,make,optimizer,step,preprocess
    assert torch.get_num_threads()==4
    m=make();opt=optimizer(m);rng=np.random.default_rng(20261029)
    steps=[];evaluation=[];full_saves=[];full_loads=[];model_saves=[];model_loads=[]
    for i in range(6):
        raw=rng.normal(size=(20,3,90,189)).astype(np.float32);labels=np.arange(20)%12
        t=time.monotonic();loss=step(m,opt,raw,labels);steps.append(time.monotonic()-t)
        print(json.dumps(dict(stage='full_native_B20_Adam_update',index=i,
                             seconds=steps[-1],synthetic_loss=loss)),flush=True)
    m.eval();raw=rng.normal(size=(20,3,90,189)).astype(np.float32)
    for _ in range(3):
        t=time.monotonic()
        with torch.inference_mode():scores=m(preprocess(raw))
        assert scores.shape==(20,12) and torch.isfinite(scores).all()
        evaluation.append(time.monotonic()-t)
    sampler=torch.Generator().manual_seed(20261029)
    full=dict(model=m.state_dict(),optimizer=opt.state_dict(),order_rng=sampler.get_state(),
              torch_rng=torch.get_rng_state(),epoch=0,updates=6)
    model=dict(model=m.state_dict(),epoch=0)
    records=[];full_sizes=[];model_sizes=[]
    for kind,payload,saves,loads,sizes in [
            ('full',full,full_saves,full_loads,full_sizes),
            ('model',model,model_saves,model_loads,model_sizes)]:
        expected=semantic(payload,torch)
        for i in range(3):
            path=out/('disposable_synthetic_'+kind+'_'+str(i)+'.pt')
            assert not path.exists()
            t=time.monotonic();torch.save(payload,path);saves.append(time.monotonic()-t)
            sizes.append(path.stat().st_size)
            t=time.monotonic();loaded=torch.load(path,map_location='cpu',weights_only=True)
            loads.append(time.monotonic()-t)
            assert semantic(loaded,torch)==expected
            record=dict(kind=kind,name=path.name,bytes=path.stat().st_size,
                        sha256=sha(path),all_saved_state_exact=True)
            records.append(record);del loaded;gc.collect()
            # Only this explicitly disposable, size/SHA-recorded synthetic copy.
            path.unlink()
    rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    free=shutil.disk_usage(B).free
    result=estimate(p,steps,evaluation,full_saves,full_loads,model_saves,model_loads,
                    max(full_sizes),max(model_sizes),rss,free)
    result.update(state='complete_synthetic_full_ResNet18_Adam100_B20_cost',
        full_B20_update_seconds=steps,evaluation_B20_seconds=evaluation,
        full_checkpoint_save_seconds=full_saves,full_checkpoint_load_seconds=full_loads,
        model_only_save_seconds=model_saves,model_only_load_seconds=model_loads,
        checkpoint_records=records,peak_RSS_KiB=rss,workspace_free_bytes=free,
        actual_100epochs_not_run=True,new_UWB_values_or_labels_fit=False)
    (out/'COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
