"""Full-size architecture producer/artifact/auditor integration on synthetic data."""
from pathlib import Path
import json,tempfile,time
from engine import np,torch
from run_pilot import train_arrays
from audit_complete import replay_arrays
from data import split,evaluation_order
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
 g=np.random.RandomState(573);ids=np.arange(12,dtype='int64')+100;ix={'train':np.arange(5),'validation':np.arange(5,8),'diagnostic':np.arange(8,12)};ordered={'validation':ix['validation'][[2,0,1]],'diagnostic':ix['diagnostic'][[1,3,0,2]]};maps=(.1*g.randn(12,90,567)).astype('float32');maps[0,:,1]=0
 x=maps[ix['train']].transpose(0,2,1).copy();vx=maps[ordered['validation']].transpose(0,2,1).copy();yt=np.array([0,1,2,0,1],dtype='int64');yv=np.array([0,1,2],dtype='int64');c={'epochs':3,'batch_size':2,'seed':7};identity={'scope':'synthetic_integration_only'}
 with tempfile.TemporaryDirectory() as d:
  out=Path(d);seen=[];result=train_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv,lambda **kw:seen.append(kw));assert result['steps']==9 and result['training_appearances']==15
  checked=replay_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv);assert checked['updates_replayed']==9
  # Changed training-ID evidence must be rejected without completing replay.
  rows=(out/'iterations.jsonl').read_text().splitlines();r=json.loads(rows[0]);r['training_ids'][0]+=999;rows[0]=json.dumps(r);(out/'iterations.jsonl').write_text('\n'.join(rows)+'\n')
  try:replay_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv)
  except AssertionError:pass
  else:raise AssertionError('Corrupted training IDs accepted')
 # Pure role tests use synthetic IDs/users, without touching real metadata.
 counts={'train':4700,'validation':1000,'diagnostic':1200};users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100]);source=np.arange(6900,dtype='int64');roles=split(source,users,np,counts);assert set(users[roles['train']])=={3,4,5,6}
 bad=users.copy();bad[-1]=8
 try:split(source,bad,np,counts)
 except ValueError:pass
 else:raise AssertionError('User8 accepted')
 before=np.random.get_state();o=evaluation_order(1000,'validation',7,np);assert sorted(o.tolist())==list(range(1000));after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);assert before[2:]==after[2:]
 result={'state':'pass_full_native_producer_and_complete_replay_integration_synthetic_only','epochs':3,'updates':9,'five_initial_first_best_last_final_custom_states_exact':True,'all_batch_metrics_losses_IDs_and_epochs_exact':True,'native_selected_last_weights_optimizer_exact':True,'both_partition_predictions_and_final_state_exact':True,'corrupt_training_ID_rejected':True,'source_roles_and_user8_rejection_synthetic_checked':True,'labelblind_evaluation_permutation_no_global_NumPy_rng_draw':True,'real_UWB_values_labels_fit':False,'wall_seconds':time.monotonic()-start}
 write(Path(__file__).resolve().parent/'INTEGRATION_CHECK.json',result);print(json.dumps(result))
if __name__=='__main__':main()
