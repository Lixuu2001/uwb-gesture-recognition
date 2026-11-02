from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def ready(marker=True):
 if marker:
  r=read(B/'REMOTE_PREREGISTRATION.json');assert r['state']=='real_pilot_remote_exact_readback_verified' and r['branch_head_verified'] and r['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
 for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
 for n,h in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==h,n
 c=read(B/'PROTOCOL.json');cost=read(Path(c['cost_workspace'])/'VERIFIED_COST_RESULT.json');assert cost['state']=='independently_verified_full_ResNet18_SGDM40_B15_cost' and cost['resource_gate_passed']
 return c
def load_inputs(c,labels=True):
 import sys
 sys.path.insert(0,c['source_preparation']);from native_finetune import np
 with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900 and set(users.tolist())==set(range(1,7))
 loc={k:np.flatnonzero(condition) for k,condition in [('train',users>=3),('validation',users==2),('diagnostic',users==1)]};assert {k:len(v) for k,v in loc.items()}==c['counts']
 maps=np.load(c['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
 subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users)
 y=None;v=None
 if labels:
  a=np.load(Path(c['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');y=np.array(a[ids[loc['train']]],copy=True);v=np.array(a[ids[loc['validation']]],copy=True);assert set(y.tolist())==set(range(12)) and set(v.tolist())==set(range(10))
 return ids,users,loc,maps,y,v
def complete():
 c=ready()
 for n in ['supervisor01','audit_supervisor01']:
  t=read(B/n/'TERMINATION.json');assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 a=read(B/'COMPLETE_AUDIT.json');assert a['state']=='pass' and not a['diagnostic_labels_read'] and not a['users7_8_labels_used']
 for key in ['full40_epoch12560_update_replay_exact','all40_validation_scores_selection_exact','best_full_model_and_final_model_optimizer_momentum_rng_exact','all2200_selected_scores_predictions_exact','all6900_raw_RMS_cache_FP64_verified']:assert a[key] is True,key
 for n,h in a['artifact_hashes'].items():assert sha(n)==h,n
 return c
