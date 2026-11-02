import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import sys,json,time,resource,signal
from guard import B,ready,sha
from cost_math import evaluate
def write(p,o):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2)+'\n');q.replace(p)
def main():
    policy=ready();out=B/'profile01';out.mkdir(exist_ok=False)
    write(out/'DECLARATION_BEFORE_TIMING.json',dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,full_input_shape=[32,90,567],UWB_values_labels_fit=False,checkpoint_IO_not_authorized_by_screen=True))
    signal.alarm(1190);sys.path.insert(0,'/workspace/uwb-moderntcn-source-probe-20261102')
    from native_moderntcn import np,torch,make,assert_structure,step
    wall_start=time.monotonic();cpu_start=time.process_time();write(out/'STATE.json',dict(state='allocating_full_unmodified_author_waveGesture_model',UWB_values_labels_fit=False))
    m,c=make();assert_structure(m,c);n=sum(p.numel() for p in m.parameters());assert n==431894540 and len(list(m.parameters()))==44
    opt=torch.optim.Adam(m.parameters(),lr=c.learning_rate)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,steps_per_epoch=147,pct_start=c.pct_start,epochs=100,max_lr=c.learning_rate)
    assert abs(opt.param_groups[0]['lr']-.00004)<1e-15
    generator=torch.Generator().manual_seed(20261030);x=torch.randn(32,90,567,generator=generator);y=torch.arange(32)%12
    m.train();write(out/'STATE.json',dict(state='warming_one_full_native_Adam_update',parameters=n,input_shape=[32,90,567],UWB_values_labels_fit=False))
    loss=step(m,opt,x,y);assert np.isfinite(loss)
    tw=[];tc=[];ew=[];ec=[]
    for i in range(3):
        write(out/'STATE.json',dict(state='timing_full_native_update',index=i,UWB_values_labels_fit=False))
        w=time.monotonic();cpu=time.process_time();loss=step(m,opt,x,y);tw.append(time.monotonic()-w);tc.append(time.process_time()-cpu);assert np.isfinite(loss)
    m.eval()
    with torch.no_grad():
        for i in range(3):
            write(out/'STATE.json',dict(state='timing_full_native_eval',index=i,UWB_values_labels_fit=False))
            w=time.monotonic();cpu=time.process_time();z=m(x,None,None,None);ew.append(time.monotonic()-w);ec.append(time.process_time()-cpu);assert torch.isfinite(z).all() and tuple(z.shape)==(32,12)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r=evaluate(policy,tw,tc,ew,ec,rss)
    r.update(profile_completed_successfully=True,full_parameters=n,trainable_tensors=44,steady_training_wall_seconds=tw,steady_training_CPU_seconds=tc,evaluation_wall_seconds=ew,evaluation_CPU_seconds=ec,max_RSS_KiB=rss,profile_wall_seconds=time.monotonic()-wall_start,profile_CPU_seconds=time.process_time()-cpu_start,CPU_threads=torch.get_num_threads(),native_initial_lr=.00004,native_scheduler_not_stepped=True,synthetic_only=True,full_numerical_optimizer_updates=4,full_checkpoint_files_written=0)
    write(out/'COST_SCREEN.json',r);write(out/'STATE.json',dict(state=r['state'],UWB_values_labels_fit=False));print(json.dumps(r));signal.alarm(0)
if __name__=='__main__':main()
