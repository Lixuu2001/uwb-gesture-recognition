"""Synthetic exact-source step, augmentation, freezing and full resume tests."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import copy,json,random,time,warnings
from pathlib import Path
import numpy as np
import torch
from ops import *
B=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('ignore',message='Implicit dimension choice')
 start=time.monotonic();r=np.random.RandomState(17);small=r.randn(257,3,90).astype(np.float32)
 src=(SCREEN/'vendor/TS-TCC/dataloader/augmentations.py').read_text();assert src.count('np.random.permutation(splits)')==src.count('ret[i] = pat[0,warp]')==1
 src=src.replace('np.random.permutation(splits)','[splits[j] for j in np.random.permutation(len(splits))]').replace('ret[i] = pat[0,warp]','ret[i] = pat[:,warp]');ns={};exec(compile(src,'documented_multichannel_numpy2_reference','exec'),ns)
 cfg=source_check()();np.random.seed(20261029);aw,ast=ns['DataTransform'](torch.from_numpy(small),cfg);rng=np.random.get_state()
 weak,strong,receipt=augmented(small);np.testing.assert_array_equal(weak,np.asarray(aw).astype(np.float32));np.testing.assert_array_equal(strong,np.asarray(ast).astype(np.float32));equal(rng,receipt['final_numpy_rng']);assert not np.array_equal(strong[:,0],strong[:,1])
 gen=torch.Generator().manual_seed(7);x=torch.randn(128,567,90,generator=gen);a=torch.randn(x.shape,generator=gen);b=torch.randn(x.shape,generator=gen);y=torch.arange(128)%12
 outcomes={};pretrained=copy.deepcopy(make()[1].state_dict())
 for phase in ('ssl','linear','finetune','scratch'):
  args=dict(phase=phase,pretrained=pretrained if phase in ('linear','finetune') else None)
  parts=make(**args);cfg,m,t,o,to,s=parts;before=copy.deepcopy(m.state_dict())
  mode={'ssl':'self_supervised','linear':'train_linear','finetune':'fine_tune','scratch':'supervised'}[phase]
  from trainer.trainer import model_train
  loss,_=model_train(m,t,o,to,torch.nn.CrossEntropyLoss(),[(x,y,a if phase=='ssl' else x,b if phase=='ssl' else x)],cfg,torch.device('cpu'),mode)
  expected=copy.deepcopy(save_state(m,t,o,to,s,phase,1,0,1,{}));firstloss=float(loss)
  cfg,m,t,o,to,s=make(**args);observed,_=step(m,t,o,to,cfg,x,y,a,b,phase);assert float(observed)==firstloss;equal(expected,save_state(m,t,o,to,s,phase,1,0,1,{}))
  saved=copy.deepcopy(save_state(m,t,o,to,s,phase,1,0,1,{}))
  for index in range(2):
   step(m,t,o,to,cfg,x,y,a,b,phase)
   if s is not None:s.step(.5+index*.1)
  final=copy.deepcopy(save_state(m,t,o,to,s,phase,1,2,3,{}))
  cfg,m,t,o,to,s=restore(saved,make(**args))
  for index in range(2):
   step(m,t,o,to,cfg,x,y,a,b,phase)
   if s is not None:s.step(.5+index*.1)
  equal(final,save_state(m,t,o,to,s,phase,1,2,3,{}))
  if phase=='linear':
   for name,p in m.named_parameters():
    if 'logits' not in name:assert not p.requires_grad and torch.equal(p,before[name])
   assert int(m.conv_block1[1].num_batches_tracked)==int(before['conv_block1.1.num_batches_tracked'])+3
  outcomes[phase]=dict(author_single_step_bitwise_parity=True,checkpoint_next_two_steps_all_models_optimizers_scheduler_RNG_bitwise=True,optimizer_states=len(o.state),temporal_optimizer_states=len(to.state))
 for epoch,batch in [(1,0),(1,35),(2,0),(40,35)]:
  ii=indices(epoch,batch);assert len(ii)==len(np.unique(ii))==128 and min(ii)>=0 and max(ii)<4700
 result=dict(state='pass',augmented_full_reference_float32_and_final_RNG_bitwise=True,chunk_boundary_records=257,all_channels_preserved=True,phases=outcomes,full_default_batch=128,elapsed_seconds=time.monotonic()-start,real_UWB_values_or_labels_read=False,real_training_started=False)
 (B/'OPS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
