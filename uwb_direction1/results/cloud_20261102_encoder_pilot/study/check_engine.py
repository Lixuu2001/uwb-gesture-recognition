"""Synthetic full-model epoch state/next-update and independent CE checks."""
from pathlib import Path
import copy,json,tempfile
from engine import np,torch,setup,snapshot,restore,epoch,permutation,values,labels,equal,close
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 g=np.random.default_rng(62);x=g.normal(size=(5,90,567)).astype('float32');vx=g.normal(size=(3,90,567)).astype('float32');y=np.array([0,1,2,0,1]);vy=np.array([0,1,2]);identity={'scope':'synthetic_engine_only'}
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);h,m,cb,s=setup(p,7,5,3,2);initial=snapshot(h,m,s,identity,0,0,-1);order=permutation(5,0,7)
  with torch.no_grad():pred=m(x[order[:2]],training=True).detach().numpy()
  expected_ce=float(-np.sum(np.eye(12,dtype='float32')[y[order[:2]]]*np.log(np.clip(pred,1e-7,1-1e-7)),axis=1).mean());restore(initial,h,m,s,identity);seen=[];expected=[]
  for ep in range(3):
   r=epoch(h,m,cb,s,x,y,vx,vy,ep,permutation(5,ep,7),lambda row:seen.append(row))
   expected.append(snapshot(h,m,s,identity,ep+1,r['total_steps'],2))
   if ep==0:torch.save(expected[-1],p/'epoch1.pt')
  np.testing.assert_allclose(seen[0]['batch_loss_from_tracker_delta'],expected_ce,rtol=2e-5,atol=2e-6);assert all(r['batch_size'] in [1,2] for r in seen)
  close(cb);(p/'fresh').mkdir();h2,m2,cb2,s2=setup(p/'fresh',7,5,3,2);saved=torch.load(p/'epoch1.pt',weights_only=False);restore(saved,h2,m2,s2,identity);equal(snapshot(h2,m2,s2,identity,1,3,2),expected[0])
  for ep in [1,2]:
   r=epoch(h2,m2,cb2,s2,x,y,vx,vy,ep,permutation(5,ep,7));equal(snapshot(h2,m2,s2,identity,ep+1,r['total_steps'],2),expected[ep])
  current=snapshot(h2,m2,s2,identity,3,9,2)
  for key in ['identity','settings','total_steps']:
   bad=copy.deepcopy(saved)
   if key=='total_steps':bad[key]+=1
   elif key=='identity':bad[key]['scope']='changed'
   else:bad[key]['batch_size']=3
   try:restore(bad,h2,m2,s2,identity)
   except ValueError:pass
   else:raise AssertionError('Bad checkpoint accepted')
   equal(snapshot(h2,m2,s2,identity,3,9,2),current)
  close(cb2);(p/'freshinitial').mkdir();h3,m3,cb3,s3=setup(p/'freshinitial',7,5,3,2);restore(initial,h3,m3,s3,identity);equal(snapshot(h3,m3,s3,identity,0,0,-1),initial);close(cb3)
 before=torch.get_rng_state().clone();a=permutation(5,0,7);b=permutation(5,0,7);equal(before,torch.get_rng_state());np.testing.assert_array_equal(a,b)
 for a in [x.astype('float64'),x[:,:,:566],np.full_like(x,np.nan)]:
  try:values(a,5)
  except ValueError:pass
  else:raise AssertionError('Invalid input accepted')
 for a in [y.astype('float32'),np.array([0,1,2,3,12]),y[:4]]:
  try:labels(a,5)
  except ValueError:pass
  else:raise AssertionError('Invalid labels accepted')
 r={'state':'pass_complete_Encoder_epoch_engine_synthetic_only','epochs':3,'updates':9,'initial_complete_callback_state_restore_exact':True,'epoch1_fresh_restore_next2epochs_all_states_exact':True,'first_unweighted_training_CE_independent_reference':True,'three_identity_settings_cursor_rejections_before_state_change':True,'six_numeric_shape_dtype_label_rejections':True,'dedicated_sampler_repeat_and_global_Torch_RNG_isolation':True,'native_fit_DataAdapter_RNG_parity_claim':False,'real_UWB_values_labels_fit':False};write(Path(__file__).with_name('ENGINE_CHECK.json'),r);print(json.dumps(r))
if __name__=='__main__':main()
