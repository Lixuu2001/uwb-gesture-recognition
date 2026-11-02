"""Synthetic native-loader RNG, LR10/20 and full-state restoration checks."""
from pathlib import Path
import copy,json,tempfile
from engine import np,torch,setup,snapshot,restore,epoch,predict,isolate,equal,capture
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 g=np.random.default_rng(62);x=g.normal(size=(3,90,567)).astype('float32');vx=g.normal(size=(2,90,567)).astype('float32');y=np.array([0,1,2]);vy=np.array([0,1]);ident={'scope':'synthetic'}
 m,opt,s=setup(7,3,20,2);initial=snapshot(m,opt,s,ident,0,0);states=[];lrs=[]
 from torch.utils.data import DataLoader,TensorDataset
 dl=DataLoader(TensorDataset(torch.from_numpy(x).transpose(1,2),torch.from_numpy(y)),batch_size=2,shuffle=False,num_workers=0,drop_last=False)
 xb,yb=next(iter(dl));m.train()
 with torch.no_grad():p=m(xb).numpy()
 expected_nll=float(-p[np.arange(len(yb)),yb.numpy()].mean());restore(initial,m,opt,s,ident);first_losses=[]
 with tempfile.TemporaryDirectory() as d:
  for ep in range(20):
   r=epoch(m,opt,s,x,y,vx,vy,ep,lambda row:first_losses.append(row['batch_loss']));states.append(snapshot(m,opt,s,ident,ep+1,r['total_steps'],1));lrs.append((r['learning_rate_before'],r['learning_rate_after']))
   if ep==9:torch.save(states[-1],Path(d)/'epoch10.pt')
  np.testing.assert_allclose(first_losses[0],expected_nll,rtol=1e-6,atol=1e-6)
  assert lrs[8]==(.002,.002) and lrs[9]==(.002,.0002) and lrs[10]==(.0002,.0002) and lrs[19]==(.0002,.00002)
  fresh,fo,fs=setup(7,3,20,2);saved=torch.load(Path(d)/'epoch10.pt',weights_only=False);restore(saved,fresh,fo,fs,ident);equal(snapshot(fresh,fo,fs,ident,10,20,1),states[9])
  for ep in range(10,20):
   r=epoch(fresh,fo,fs,x,y,vx,vy,ep);equal(snapshot(fresh,fo,fs,ident,ep+1,r['total_steps'],1),states[ep])
  current=snapshot(fresh,fo,fs,ident,20,40,1)
  for key in ['identity','settings','total_steps']:
   bad=copy.deepcopy(saved)
   if key=='total_steps':bad[key]+=1
   elif key=='identity':bad[key]['scope']='wrong'
   else:bad[key]['batch_size']=3
   try:restore(bad,fresh,fo,fs,ident)
   except ValueError:pass
   else:raise AssertionError('Bad checkpoint accepted')
   equal(snapshot(fresh,fo,fs,ident,20,40,1),current)
  restore(initial,m,opt,s,ident);equal(snapshot(m,opt,s,ident,0,0),initial)
  before=capture(fresh,fo);isolate(fresh,fo,lambda:torch.save(fresh.state_dict(),Path(d)/'native.pt'));equal(capture(fresh,fo),before)
  # Native inference DataLoader consumes global base-seed exactly once per iter.
  predict(fresh,vx,2);assert not torch.equal(before['torch_rng'],capture(fresh,fo)['torch_rng'])
 r=dict(state='pass_full_TCN_native_epoch_engine_synthetic_only',epochs=20,updates=40,source_LR_after10_and20_exact=True,epoch10_fresh_restore_next10epochs_exact=True,three_wrong_checkpoint_refusals_before_mutation=True,initial_restore_exact=True,native_save_full_RNG_isolation_exact=True,native_DataLoader_global_base_seed_consumption_verified=True,real_UWB_values_labels_fit=False)
 write(Path(__file__).with_name('ENGINE_CHECK.json'),r);print(json.dumps(r))
if __name__=='__main__':main()
