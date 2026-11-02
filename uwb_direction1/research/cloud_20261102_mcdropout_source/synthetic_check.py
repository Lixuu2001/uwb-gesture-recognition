import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,copy,sys
from mc_dropout import torch,np,nn,infer,dropout_only
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
class Fixture(nn.Module):
 def __init__(self):
  super().__init__();self.bn=nn.BatchNorm1d(4);self.drop=nn.Dropout(.2);self.head=nn.Linear(4,12)
 def forward(self,x):return self.head(self.drop(self.bn(x.transpose(1,2)).mean(-1)))
def state(m):return copy.deepcopy(m.state_dict()),{n:v.training for n,v in m.named_modules()},torch.get_rng_state().clone()
def unchanged(m,s):
 for n,v in m.state_dict().items():assert torch.equal(v,s[0][n])
 assert {n:v.training for n,v in m.named_modules()}==s[1] and torch.equal(torch.get_rng_state(),s[2])
def check(m,x,seed):
 m.train();s=state(m);r=infer(m,x,100,seed,2);unchanged(m,s);again=infer(m,x,100,seed,2)
 np.testing.assert_array_equal(r['logits'],again['logits']);assert np.any(r['logits'][0]!=r['logits'][1]);np.testing.assert_array_equal(r['predictions'],r['mean_sigmoid'].argmax(1))
 ref=(1/(1+np.exp(-r['logits'].astype('float64')))).mean(0);np.testing.assert_allclose(r['mean_sigmoid'],ref,rtol=2e-6,atol=2e-7)
 with torch.random.fork_rng(devices=[]),dropout_only(m),torch.no_grad():
  torch.manual_seed(seed);z=[]
  for _ in range(100):z.append(np.concatenate([m(torch.from_numpy(x[i:i+2])).numpy() for i in range(0,len(x),2)]))
 np.testing.assert_array_equal(r['logits'],np.stack(z));unchanged(m,s)
 try:
  with dropout_only(m):raise RuntimeError('synthetic exception')
 except RuntimeError:pass
 unchanged(m,s);return r
def main():
 g=np.random.default_rng(77);torch.manual_seed(4);check(Fixture(),g.normal(size=(3,7,4)).astype('float32'),20261029)
 sys.path.insert(0,'/workspace/uwb-timemil-20261029');from adapter import make_model
 m=make_model('timemil',20261029);r=check(m,(.1*g.normal(size=(3,90,567))).astype('float32'),20261029)
 assert len(r['dropout_modules'])==3 and all(v['p']==.2 for v in r['dropout_modules'])
 result=dict(state='pass_MC100_full_random_TimeMIL_synthetic_only',samples=100,seed=20261029,full_model_parameters=sum(p.numel() for p in m.parameters()),active_dropout_modules=r['dropout_modules'],all_model_and_BN_buffers_immutable=True,mixed_modes_and_global_RNG_restored=True,repeated_run_exact=True,independent_explicit_stochastic_loop_exact=True,float64_mean_sigmoid_matches_independent_numpy=True,three_dropout_nodes_only=True,exception_mode_restore=True,BN_training_never_enabled=True,TimeMIL_trained_checkpoint_loaded=False,real_UWB_values_labels_fit=False,formal_cost_run=False,novelty_claim=False)
 Path(__file__).with_name('SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
