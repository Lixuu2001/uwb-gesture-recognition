from pathlib import Path
import json,tempfile,time
from engine import np,torch
from run_pilot import train_arrays
from audit_complete import replay_arrays
from data import split,evaluation_order
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
 g=np.random.RandomState(573);ids=np.arange(12,dtype='int64')+100;ix=dict(train=np.arange(5),validation=np.arange(5,8),diagnostic=np.arange(8,12));ordered={k:ix[k] for k in ['validation','diagnostic']};maps=(.1*g.randn(12,90,567)).astype('float32');x=maps[ix['train']].copy();vx=maps[ordered['validation']].copy();y=np.array([0,1,2,0,1]);vy=np.array([0,1,2]);c=dict(epochs=20,batch_size=2,seed=7);ident={'scope':'synthetic_integration'}
 with tempfile.TemporaryDirectory() as d:
  out=Path(d);r=train_arrays(c,out,ident,ids,ix,ordered,maps,x,y,vx,vy,lambda **kw:None);assert r['steps']==60 and r['training_appearances']==100
  a=replay_arrays(c,out,ident,ids,ix,ordered,maps,x,y,vx,vy);assert a['updates_replayed']==60
  rows=(out/'iterations.jsonl').read_text().splitlines();bad=json.loads(rows[0]);bad['training_ids'][0]+=999;rows[0]=json.dumps(bad);(out/'iterations.jsonl').write_text('\n'.join(rows)+'\n')
  try:replay_arrays(c,out,ident,ids,ix,ordered,maps,x,y,vx,vy)
  except AssertionError:pass
  else:raise AssertionError('Corrupted batch IDs accepted')
 users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);counts=dict(train=4700,validation=1000,diagnostic=1200);roles=split(np.arange(6900),users,np,counts);assert set(users[roles['train']])=={3,4,5,6}
 bad=users.copy();bad[-1]=8
 try:split(np.arange(6900),bad,np,counts)
 except ValueError:pass
 else:raise AssertionError('User8 accepted')
 np.testing.assert_array_equal(evaluation_order(1000,'validation',7,np),np.arange(1000))
 r=dict(state='pass_complete_TCN_producer_replay_integration_synthetic_only',epochs=20,updates=60,all_initial_first_epoch10_last_final_custom_states_exact=True,native_initial_last_weights_exact=True,all_batch_IDs_losses_epoch_metrics_LR_and_dual_logprobs_exact=True,corrupt_training_ID_rejected=True,user8_role_rejected=True,source_canonical_order_verified=True,real_UWB_values_labels_fit=False,wall_seconds=time.monotonic()-start)
 write(Path(__file__).with_name('INTEGRATION_CHECK.json'),r);print(json.dumps(r))
if __name__=='__main__':main()
