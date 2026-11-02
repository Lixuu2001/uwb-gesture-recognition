import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,tempfile,contextlib,io
from native_dlinear import np,torch,make,forward,step,capture,restore,equal,cli_defaults
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def reference(m,x):
 a=np.pad(x,((0,0),(12,12),(0,0)),mode='edge');trend=np.stack([a[:,i:i+25].mean(1) for i in range(90)],1);season=x-trend
 sd=m.state_dict();s=(season.transpose(0,2,1)@sd['Linear_Seasonal.weight'].numpy().T)+sd['Linear_Seasonal.bias'].numpy();t=(trend.transpose(0,2,1)@sd['Linear_Trend.weight'].numpy().T)+sd['Linear_Trend.bias'].numpy();z=(s+t).transpose(0,2,1).reshape(len(x),-1)
 return z@sd['projection.weight'].numpy().T+sd['projection.bias'].numpy()
def main():
 g=np.random.default_rng(271);x=g.normal(size=(3,90,567)).astype('float32');y=torch.tensor([0,1,2]);m,opt,early=make(20261029);assert sum(p.numel() for p in m.parameters())==628752 and len(list(m.parameters()))==6
 initial=capture(m,opt)
 with torch.no_grad():np.testing.assert_allclose(forward(m,torch.from_numpy(x)).numpy(),reference(m,x),rtol=2e-5,atol=2e-6)
 for _ in range(7):step(m,opt,torch.from_numpy(x),y)
 saved=capture(m,opt);assert len(saved['optimizer']['state'])==6 and all(int(s['step'])==7 for s in saved['optimizer']['state'].values())
 with torch.no_grad():np.testing.assert_allclose(forward(m,torch.from_numpy(x)).numpy(),reference(m,x),rtol=2e-5,atol=2e-6)
 with tempfile.TemporaryDirectory() as d:
  torch.save(saved,Path(d)/'full.pt');torch.save(m.state_dict(),Path(d)/'native.pth');fresh,fopt,_=make(20261029);restore(torch.load(Path(d)/'full.pt',weights_only=False),fresh,fopt);equal(capture(fresh,fopt),saved)
  a=step(fresh,fopt,torch.from_numpy(x),y);after=capture(fresh,fopt);restore(saved,m,opt);b=step(m,opt,torch.from_numpy(x),y);equal(a,b);equal(capture(m,opt),after)
  fresh.load_state_dict(torch.load(Path(d)/'native.pth',weights_only=True));equal(fresh.state_dict(),saved['model'])
  h=early(patience=3,verbose=False)
  with contextlib.redirect_stdout(io.StringIO()):
   h(-.5,m,d);h(-.5,m,d);assert h.counter==0 and not h.early_stop
   h(-.4,m,d);h(-.4,m,d);h(-.4,m,d);assert h.counter==3 and h.early_stop and h.best_score==.5
 restore(initial,m,opt);equal(capture(m,opt),initial)
 r=dict(state='pass_full_TSLib_DLinear_synthetic_only',parameters=628752,parameter_tensors=6,CLI_defaults=cli_defaults(),full_NumPy_endpoint_average_two_temporal_dense_and_head_reference=True,seven_RAdam_updates_including_rectification=True,fresh_full_state_restore_and_exact_next_update=True,native_weights_reload_exact=True,source_early_stop_ties_save_latest_and_patience3_exact=True,source_classification_RAdam_not_forecasting_Adam=True,source_grad_clip4_retained=True,classification_transfer_not_original_forecasting_replication=True,real_UWB_values_labels_fit=False,formal_cost_run=False,novelty_claim=False)
 Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
