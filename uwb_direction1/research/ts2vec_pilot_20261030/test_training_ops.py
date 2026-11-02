"""Synthetic original-fit parity, exact resumable raw/average/optimizer/RNG state."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import io,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from training_ops import make_model,source_check,STEP,optimizer,checkpoint,restore,batch_indices
B=Path(__file__).resolve().parent
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
def main():
 assert not (B/'TRAINING_OPS_CHECK.json').exists();started=time.monotonic();source_check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 data=np.random.default_rng(402).normal(size=(32,90,567)).astype(np.float32)
 reference=make_model();expected_losses=[];reference.after_iter_callback=lambda m,l:expected_losses.append(l);reference.fit(data.copy(),n_iters=3)
 reference_torch=torch.get_rng_state();reference_numpy=np.random.get_state();reference_python=random.getstate()
 candidate=make_model();opt=optimizer(candidate);dataset=TensorDataset(torch.from_numpy(data.copy()).to(torch.float));loader=DataLoader(dataset,batch_size=16,shuffle=True,drop_last=True);actual_losses=[]
 while True:
  interrupted=False
  for batch in loader:
   if candidate.n_iters>=3:interrupted=True;break
   loss=STEP(candidate,batch[0],opt);actual_losses.append(float(loss.detach()));candidate.n_iters+=1
  if interrupted:break
  candidate.n_epochs+=1
 assert actual_losses==expected_losses and candidate.n_epochs==reference.n_epochs and candidate.n_iters==reference.n_iters
 equal(candidate._net.state_dict(),reference._net.state_dict());equal(candidate.net.state_dict(),reference.net.state_dict());equal(torch.get_rng_state(),reference_torch);equal(np.random.get_state(),reference_numpy);equal(random.getstate(),reference_python)
 stream=io.BytesIO();torch.save(checkpoint(candidate,opt,3,dict(epoch=1,next_offset=48),dict(synthetic=True)),stream);stream.seek(0);saved=torch.load(stream,weights_only=False)
 fixed=torch.from_numpy(data[:16].copy());loss1=[]
 for _ in range(2):loss1.append(float(STEP(candidate,fixed.clone(),opt).detach()));candidate.n_iters+=1
 end_torch=torch.get_rng_state();end_numpy=np.random.get_state();end_python=random.getstate()
 resumed=make_model();other=optimizer(resumed);step,sampler=restore(resumed,other,saved);assert step==3
 loss2=[]
 for _ in range(2):loss2.append(float(STEP(resumed,fixed.clone(),other).detach()));resumed.n_iters+=1
 assert loss1==loss2;equal(candidate._net.state_dict(),resumed._net.state_dict());equal(candidate.net.state_dict(),resumed.net.state_dict());equal(opt.state_dict(),other.state_dict());equal(end_torch,torch.get_rng_state());equal(end_numpy,np.random.get_state());equal(end_python,random.getstate())
 for step in (1,293,294,586,587,600):
  ix,meta=batch_indices(step);assert len(ix)==len(np.unique(ix))==16 and min(ix)>=0 and max(ix)<4700
  epoch=(step-1)//293+1;start=(step-1)%293*16;expected=torch.randperm(4700,generator=torch.Generator().manual_seed(20261029+epoch)).numpy()[start:start+16];np.testing.assert_array_equal(ix,expected)
 result=dict(state='pass',synthetic_only=True,real_UWB_values_or_labels_read=False,real_training_started=False,original_author_fit_three_steps_model_average_losses_all_rng_exact=True,extracted_AST_training_math_unchanged=True,checkpoint_resume_next_two_steps_raw_average_optimizer_all_rng_bitwise_exact=True,initial_inclusive_average_preserved=True,sampler_boundary_steps_verified=[1,293,294,586,587,600],new_sampler_scope='Prospective dedicated epochorderseed+epoch differs from originalglobalDataLoader trajectory; fixed source-data protocol still required.',wall_seconds=time.monotonic()-started)
 (B/'TRAINING_OPS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
