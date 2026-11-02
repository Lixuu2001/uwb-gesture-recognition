"""Full author fit control flow versus engine; pure synthetic, no UWB/cost gate."""
import ast,copy,contextlib,importlib,io,json,random,tempfile,time
from pathlib import Path
from source_engine import *
B=Path(__file__).resolve().parent
def full_source_function():
 mod=importlib.import_module('Classifiers.OS_CNN.OS_CNN_easy_use')
 tree=ast.parse((SOURCE/'vendor/oscnn/Classifiers/OS_CNN/OS_CNN_easy_use.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit'));fn.name='fit_with_return_state'
 # Only append a return AFTER the unchanged source fit body to inspect its locals.
 fn.body.append(ast.Return(ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Load()) for n in ['torch_OS_CNN','optimizer','scheduler','layer_parameter_list']],ctx=ast.Load())))
 ns=dict(mod.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'pinned_author_fit_with_posthoc_return_only','exec'),ns)
 return mod,ns['fit_with_return_state']
def raw_snapshot(m,o,s,layers):return copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),training=m.training,layers=layers,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.RandomState(9);x=g.randn(32,567,90).astype('float32');y=(np.arange(32)%12).astype('int64');vx=g.randn(19,567,90).astype('float32');vy=(np.arange(19)%12).astype('int64');seed=19
 mod,fit=full_source_function()
 with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
  torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  holder=mod.OS_CNN_easy_use(a+'/', 'synthetic', 'cpu',paramenter_number_of_layer_list=[8*128*567,5*128*256+2*256*128],max_epoch=2,batch_size=16,print_result_every_x_epoch=1)
  with contextlib.redirect_stdout(io.StringIO()):m,o,s,layers=fit(holder,x,y,vx,vy)
  original=raw_snapshot(m,o,s,layers);checkpoint=torch.load(holder.model_save_path,weights_only=False)
  engine=build(567,90,12,seed);from torch.utils.data import TensorDataset,DataLoader
  train=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=3,shuffle=True);val=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy)),batch_size=3,shuffle=False)
  for ep in [1,2]:engine.epoch(train,val,ep,1,Path(b)/'periodic.pt')
  torch.save(engine.model.state_dict(),Path(b)/'final.pt');actual=engine.snapshot({},{});stripped={k:v for k,v in actual.items() if k not in ['identity','sampler']};equal(original,stripped);equal(checkpoint,torch.load(Path(b)/'final.pt',weights_only=False))
 # Independent CE equation, scheduler state and post-eval restoration across epochs.
 engine=build(567,90,12,seed);x2=torch.from_numpy(x[:5]);y2=torch.from_numpy(y[:5]);loss,z=engine.update(x2,y2);z=z.numpy().astype('float64');shift=z-z.max(1,keepdims=True);nll=np.log(np.exp(shift).sum(1))-shift[np.arange(5),y2.numpy()];assert abs(float(loss.detach())-float(nll.mean()))<2e-6
 batches=[(torch.from_numpy(x[i:i+5]),torch.from_numpy(y[i:i+5])) for i in range(0,32,5)];validation=[(torch.from_numpy(vx),torch.from_numpy(vy))]
 with tempfile.TemporaryDirectory() as d:
  engine.epoch(batches,validation,1,1,Path(d)/'periodic.pt');saved=engine.snapshot(dict(synthetic=True),dict(epoch=1));engine.epoch(batches,validation,2,1,Path(d)/'periodic.pt');final=engine.snapshot(dict(synthetic=True),dict(epoch=2));other=build(567,90,12,seed);other.restore(saved);equal(saved,other.snapshot(dict(synthetic=True),dict(epoch=1)));other.epoch(batches,validation,2,1,Path(d)/'restored.pt');equal(final,other.snapshot(dict(synthetic=True),dict(epoch=2)))
 # Isolate plateau state: source default relative threshold, strict >patience trigger.
 e=build(567,90,12,seed)
 for ep in range(154):e.scheduler.step(1.);assert abs(e.optimizer.param_groups[0]['lr']-max(.0001,.001*.5**(ep//51)))<1e-12
 result=dict(state='pass_synthetic_engine_only',author_model_and_fit_body_unmodified=True,posthoc_return_appended_after_source_fit_for_observation_only=True,full_source_two_epochs_22updates_periodic_eval_rawcheckpoint_model_Adam_scheduler_all_RNG_exact=True,full_source_uses_native_DataLoader_shuffle_in_this_test=True,source_data_loader_global_RNG_vs_future_explicit_sampler_not_claimed=True,independent_NumPy_CE=True,restore_next_epoch_including_eval_weight_mask_side_effect_all_states_RNG_exact=True,last_batch_loss_scheduler_preserved=True,periodic_checkpoint_and_final_last_not_validation_best_preserved=True,plateau_154_constant_steps_matches_independent51step_rule=True,real_UWB_configuration_selected=False,formal_CPU_cost_gate_run=False,real_UWB_values_labels_or_fit=False,wall_seconds=time.monotonic()-start)
 (B/'SYNTHETIC_ENGINE_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
