"""Single formal shape-only full author multivariate-model CPU budget check."""
import sys,tempfile,json,time,resource
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-oscnn-engine-20261031')
from source_engine import torch,np,build
B=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();e=build(567,90,12,20261029);g=torch.Generator().manual_seed(9);x=torch.randn(16,567,90,generator=g);y=torch.arange(16)%12;training=[];evaluation=[]
 for i in range(8):
  t=time.monotonic();loss,z=e.update(x,y);assert torch.isfinite(loss);training.append(time.monotonic()-t)
 e.model.eval()
 for i in range(4):
  t=time.monotonic();z=e.model(x);assert z.shape==(16,12) and torch.isfinite(z).all();evaluation.append(time.monotonic()-t)
 estimates={str(ep):2*(ep*294*max(training)+((ep//50)*(294+63)+63+75)*max(evaluation))+120 for ep in [2000,500,100]};chosen=next((ep for ep in [2000,500,100] if estimates[str(ep)]<=43200),None)
 result=dict(state='pass' if chosen else 'resource_gate_fail',chosen_epochs=chosen,source_constructor_epochs=2000,source_demo_epochs=500,explicit_epoch_budget_adaptation=(chosen==100),variant='author_multivariate_OS_CNN',author_model_unmodified=True,parameters_including_fixed_masks=sum(p.numel() for p in e.model.parameters()),trainable_parameters=sum(p.numel() for p in e.model.parameters() if p.requires_grad),active_Adam_states=len(e.optimizer.state),batch_size=16,training_batches_per_epoch=294,last_training_batch=12,training_records_per_epoch=4700,source_monitor_period=50,training_seconds=training,eval_seconds=evaluation,eval_profile_grad_enabled=True,estimated_seconds_by_epoch_count=estimates,estimate_formula='2*(epochs*294*maxtrain+((epochs//50)*(294+63)+63+75)*maxeval)+120',budget_seconds=43200,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.monotonic()-start,real_UWB_values_labels_or_fit=False)
 (B/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
