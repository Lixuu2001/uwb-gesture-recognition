from pathlib import Path
import copy,json,tempfile,time
from engine import np,torch,setup,epoch,snapshot,restore,equal,predict
from run_pilot import train_arrays
from audit_complete import replay_arrays
from data import split
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.default_rng(574)
 ids=np.arange(133,dtype='int64')+100;ix=dict(train=np.arange(65),validation=np.arange(65,98),diagnostic=np.arange(98,133));ordered={k:ix[k] for k in ['validation','diagnostic']};maps=(.1*g.normal(size=(133,90,567))).astype('float32');x=maps[ix['train']].copy();vx=maps[ordered['validation']].copy();y=np.arange(65)%12;vy=np.arange(33)%12;c=dict(epochs=10,batch_size=32,seed=20261029);identity={'scope':'synthetic_only'}
 with tempfile.TemporaryDirectory() as d:
  out=Path(d);r=train_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy,lambda **kw:None);a=replay_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy);assert r['steps']==a['updates_replayed']==r['epochs']*3
  rows=(out/'iterations.jsonl').read_text().splitlines();bad=json.loads(rows[0]);bad['training_ids'][0]+=999;rows[0]=json.dumps(bad);(out/'iterations.jsonl').write_text('\n'.join(rows)+'\n')
  try:replay_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy)
  except AssertionError:pass
  else:raise AssertionError('Corrupted sample IDs accepted')
 with tempfile.TemporaryDirectory() as d:
  m,opt,h,s=setup(c['seed'],65,10,32);initial=snapshot(m,opt,h,s,identity,0,0);z=epoch(m,opt,h,s,x,y,vx,vy,0,d);saved=snapshot(m,opt,h,s,identity,1,3,2);actual=epoch(m,opt,h,s,x,y,vx,vy,1,d);after=snapshot(m,opt,h,s,identity,2,6,2)
  fresh,fo,fh,fs=setup(c['seed'],65,10,32);restore(saved,fresh,fo,fh,fs,identity);equal(snapshot(fresh,fo,fh,fs,identity,1,3,2),saved);again=epoch(fresh,fo,fh,fs,x,y,vx,vy,1,d);equal(actual,again);equal(snapshot(fresh,fo,fh,fs,identity,2,6,2),after)
  for key in ['identity','settings','total_steps']:
   bad=copy.deepcopy(saved)
   if key=='total_steps':bad[key]+=1
   elif key=='identity':bad[key]['scope']='wrong'
   else:bad[key]['batch_size']=64
   try:restore(bad,fresh,fo,fh,fs,identity)
   except ValueError:pass
   else:raise AssertionError('Bad checkpoint accepted')
   equal(snapshot(fresh,fo,fh,fs,identity,2,6,2),after)
  restore(initial,m,opt,h,s,identity);equal(snapshot(m,opt,h,s,identity,0,0),initial)
 users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);counts=dict(train=4700,validation=1000,diagnostic=1200);split(np.arange(6900),users,np,counts);users[-1]=8
 try:split(np.arange(6900),users,np,counts)
 except ValueError:pass
 else:raise AssertionError('User8 admitted')
 result=dict(state='pass_complete_DLinear_producer_full_replay_restore_integration_synthetic_only',max_epochs=10,actual_fixture_epochs=r['epochs'],actual_fixture_updates=r['steps'],all_initial_first_best_last_final_states_exact=True,native_initial_best_last_exact=True,all_batch_ID_loss_gradnorm_and_epoch_earlystop_double_validation_logits_exact=True,fresh_epoch_early_state_restore_next_epoch_exact=True,three_wrong_checkpoint_refusals_before_mutation=True,initial_restore_exact=True,corrupt_training_ID_rejected=True,user8_role_rejected=True,real_UWB_values_labels_fit=False,wall_seconds=time.monotonic()-start);write(Path(__file__).with_name('INTEGRATION_CHECK.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
