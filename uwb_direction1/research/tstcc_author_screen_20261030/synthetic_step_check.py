"""Full default HAR batch128 synthetic author SSL step; no real data."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json,torch,warnings
from author_adapter import make_models,author_one_batch
def check():
 p=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_num_interop_threads(1)
 start=time.monotonic();cfg,m,t,o,to=make_models();gen=torch.Generator().manual_seed(7);a=torch.randn(128,567,90,generator=gen);b=torch.randn(128,567,90,generator=gen)
 with warnings.catch_warnings(record=True) as w:
  warnings.simplefilter('always');before={k:v.clone() for k,v in m.state_dict().items()};loss,_=author_one_batch(m,t,o,to,cfg,a,b)
 assert torch.isfinite(loss);changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
 with torch.no_grad():logits,features=m(a)
 assert logits.shape==(128,12) and features.shape==(128,128,13)
 assert all(torch.isfinite(v).all() for v in m.state_dict().values());assert len(o.state)>0 and len(to.state)>0
 result=dict(state='pass_single_fullHARdefault_batch_author_SSL_step',model_parameters=sum(v.numel() for v in m.parameters()),temporal_parameters=sum(v.numel() for v in t.parameters()),batch_size=cfg.batch_size,epochs=cfg.num_epoch,shape_only_adaptations=dict(input_channels=567,classes=12,features_len=13),full_latent_shape=list(features.shape),loss=float(loss),changed_model_state_tensors=len(changed),encoder_optimizer_states=len(o.state),temporal_optimizer_states=len(to.state),elapsed_seconds=time.monotonic()-start,warnings=[{'category':x.category.__name__,'message':str(x.message)} for x in w],real_UWB_values_or_labels_read=False,real_experiment_started=False,author_augmentation_used=False,augmentation_semantics_still_pending=True,checkpoint_resume_test_passed=False,full_cost_gate_passed=False)
 (p/'SYNTHETIC_STEP_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':check()
