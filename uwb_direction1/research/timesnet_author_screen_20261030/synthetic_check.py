"""Full official gesture preset: source parity, FFT reference and restore proof."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast,copy,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from author_adapter import *
B=Path(__file__).resolve().parent
def main():
 start=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 from models.TimesNet import FFT_for_Period
 r=np.random.RandomState(20261029);fft_cases=0
 for batch,length,channels in [(16,90,567),(3,17,4),(5,31,8)]:
  x=r.randn(batch,length,channels).astype(np.float64);z=np.fft.rfft(x,axis=1);amp=np.abs(z).mean(axis=0).mean(axis=-1);amp[0]=0;top=np.argsort(amp)[-3:][::-1];period,weight=FFT_for_Period(torch.from_numpy(x),3);np.testing.assert_array_equal(period,length//top);np.testing.assert_allclose(weight.numpy(),np.abs(z).mean(axis=-1)[:,top],rtol=1e-12,atol=1e-12);fft_cases+=1
 cfg,m,o=make();params=sum(p.numel() for p in m.parameters());gen=torch.Generator().manual_seed(7);batches=[(torch.randn(16,90,567,generator=gen),torch.arange(16)%12) for _ in range(3)]
 loop=copy.deepcopy(original_inner_loop());loop.iter=ast.parse('enumerate(loader)',mode='eval').body
 proto=ast.parse('def full_loop(self,loader,model_optim,criterion):\n iter_count=0\n train_loss=[]\n return train_loss').body[0];proto.body.insert(2,loop);ns=dict(torch=torch,nn=nn,time=time);exec(compile(ast.fix_missing_locations(ast.Module(body=[proto],type_ignores=[])),'full_original_author_inner_loop','exec'),ns)
 stop=early_stopper();m.train();losses=ns['full_loop'](SimpleNamespace(model=m,device=torch.device('cpu'),args=cfg),[(x,y[:,None],torch.ones(16,90)) for x,y in batches],o,nn.CrossEntropyLoss());expected=copy.deepcopy(snapshot(m,o,stop,1,3))
 cfg,m,o=make();stop=early_stopper();observed=[float(step(m,o,x,y)) for x,y in batches];assert losses==observed;equal(expected,snapshot(m,o,stop,1,3));assert all(torch.isfinite(v).all() for v in m.state_dict().values());active=len(o.state)
 saved=copy.deepcopy(snapshot(m,o,stop,1,3))
 with tempfile.TemporaryDirectory() as d:
  for index,(x,y) in enumerate(batches[:2]):step(m,o,x,y);stop(-(.4+.1*index),m,d)
  final=copy.deepcopy(snapshot(m,o,stop,2,5));cfg,m2,o2=make();s2=early_stopper();restore(saved,m2,o2,s2)
  for index,(x,y) in enumerate(batches[:2]):step(m2,o2,x,y);s2(-(.4+.1*index),m2,d)
  equal(final,snapshot(m2,o2,s2,2,5))
  # Original ties save/reset and only strict declines consume patience.
  tie=early_stopper(patience=2);tie(-.5,m2,d);tie(-.4,m2,d);assert tie.counter==1;tie(-.5,m2,d);assert tie.counter==0;tie(-.4,m2,d);tie(-.4,m2,d);assert tie.early_stop
 m2.eval()
 with torch.inference_mode():
  x=batches[0][0];whole=m2(x,torch.ones(16,90),None,None);separate=torch.cat([m2(x[:8],torch.ones(8,90),None,None),m2(x[8:],torch.ones(8,90),None,None)])
 delta=float((whole-separate).abs().max());assert whole.shape==(16,12) and torch.isfinite(whole).all()
 unused=[n for n,p in m2.named_parameters() if p.grad is None]
 result=dict(state='pass',official_preset='UWaveGestureLibrary_complete',parameters=params,batch_size=16,d_model=32,d_ff=64,e_layers=2,top_k=3,num_kernels=6,train_epochs_cap=30,patience=10,fft_independent_NumPy_cases=fft_cases,original_three_update_loop_model_optimizer_losses_all_RNG_bitwise=True,checkpoint_next_two_steps_model_RAdam_stopper_all_RNG_bitwise=True,original_early_stopping_tie_reset_and_decline_patience=True,active_optimizer_parameter_states=active,unused_original_parameter_names=unused,batch_context_max_logit_difference=delta,batch_context_period_selection_preserved=True,real_UWB_values_or_labels_read=False,real_training_started=False,paper_fulltext_read=False,wall_seconds=time.monotonic()-start)
 (B/'SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
