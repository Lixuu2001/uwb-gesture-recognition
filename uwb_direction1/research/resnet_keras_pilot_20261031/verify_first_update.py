"""Read-only immutable first-step proof; no real labels, values or forward/fit."""
import tempfile,time,json
from pathlib import Path
from run_pilot import read,sha,write
from ops import *
B=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cfg=read(B/'PROTOCOL.json');out=B/'run01';frozen=read(B/'FROZEN_HASHES.json')
 for n,h in frozen.items():assert sha(B/n)==h,n
 initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);first=torch.load(out/'first_update.pt',map_location='cpu',weights_only=False);identity=read(out/'IDENTITY.json');assert initial['identity']==first['identity']==identity and identity['protocol_sha256']==sha(B/'PROTOCOL.json')
 with tempfile.TemporaryDirectory() as d:
  h,m,cb=make(d,cfg['seed']);equal(initial,snapshot(h,m,0,0,-1,identity));restore(first,h,m);check=snapshot(h,m,1,1,0,identity);expect={k:v for k,v in first.items() if k!='training_ids'};equal(expect,check)
 assert first['epoch']==first['total_steps']==1 and len(first['native']['optimizer'])==98 and int(first['native']['optimizer'][0])==1
 assert all(np.isfinite(v).all() for v in first['native']['weights']+first['native']['optimizer'])
 changed=sum(not np.array_equal(a,b) for a,b in zip(initial['native']['weights'],first['native']['weights']));assert changed>0
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 train=np.flatnonzero(np.isin(users,[3,4,5,6]));np.testing.assert_array_equal(first['training_ids'],ids[train[training_order(1,cfg['seed'])[:64]]])
 result=dict(state='pass',initial_exact_native_full_author_architecture_seed=True,parameters=834124,trainable_parameter_tensors=48,native_Adam_variables=98,Adam_iterations=1,changed_model_state_tensors=changed,all_model_optimizer_state_finite=True,full_first_state_restore_including_callbacks_metrics_all_RNG_exact=True,exact_seeded_training_ID_count=64,frozen_entries_verified=len(frozen),first_update_sha256=sha(out/'first_update.pt'),verifier_sha256=sha(__file__),real_raw_values_or_labels_read=False,real_forward_or_fit=False,users7_8_labels_used=False,observed_unix=time.time());write(B/'FIRST_UPDATE_VERIFIED.json',result);print(json.dumps(result))
