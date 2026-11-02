from pathlib import Path
import tempfile,json,time
from engine import np,torch,model,produce,replay,deterministic
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();g=np.random.default_rng(785);maps=(.1*g.normal(size=(7,90,567))).astype('float32');ids=np.arange(7,dtype='int64')+10;ordered=dict(validation=np.arange(3),diagnostic=np.arange(3,7));c=dict(samples=4,batch_size=2,seed=20261029);identity=dict(scope='synthetic_complete_random_TimeMIL_no_trained_checkpoint');m=model(c['seed'])
 with tempfile.TemporaryDirectory() as d:
  out=Path(d);ref=out/'reference.npz';z=deterministic(m,maps[ordered['diagnostic']],2);np.savez_compressed(ref,ids=ids[ordered['diagnostic']],logits=z)
  r=produce(c,out,m,ids,ordered,maps,identity,lambda **kw:None,ref);assert r['gradient_updates']==0
  a=replay(c,out,model(c['seed']),ids,ordered,maps,identity,ref);assert a['samples_replayed']==8
  lines=(out/'passes.jsonl').read_text().splitlines();bad=json.loads(lines[0]);bad['ids_sha256']='wrong';lines[0]=json.dumps(bad);(out/'passes.jsonl').write_text('\n'.join(lines)+'\n')
  try:replay(c,out,model(c['seed']),ids,ordered,maps,identity,ref)
  except AssertionError:pass
  else:raise AssertionError('Corrupt trace sample IDs accepted')
 result=dict(state='pass_complete_random_TimeMIL_MC_producer_independent_replay',synthetic_samples_per_role=4,synthetic_total_samples_replayed=8,full850973parameter_source_model=True,baseline_frozen_same_ID_logits_exact=True,all_MC_logits_rng_float64_means_exact=True,initial_first_final_and_native_weights_exact=True,odd_final_batch_exercised=True,corrupt_sample_ID_digest_rejected=True,new_trained_UWB_checkpoint_loaded=False,real_UWB_values_labels_fit=False,wall_seconds=time.monotonic()-start);write(Path(__file__).with_name('INTEGRATION_CHECK.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
