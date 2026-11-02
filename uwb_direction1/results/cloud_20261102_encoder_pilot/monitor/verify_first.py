"""Validate immutable first state without gesture-label access or new fit."""
from pathlib import Path
import hashlib,json,sys
B=Path('/workspace/uwb-encoder-pilot-20261102');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from guards import read,sha
def main():
 c=read(B/'PROTOCOL.json');out=B/'run01'
 assert (out/'initial.pt').exists() and (out/'first_update.pt').exists()
 from engine import np,torch,permutation,equal
 initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);first=torch.load(out/'first_update.pt',map_location='cpu',weights_only=False)
 assert first['identity']['protocol_sha256']==sha(B/'PROTOCOL.json') and initial['identity']==first['identity']
 assert initial['total_steps']==0 and first['total_steps']==1 and first['epoch']==1 and first['batch']==0
 assert initial['settings']==first['settings'] and first['settings']['epochs']==100 and first['settings']['batch_size']==12 and first['settings']['seed']==20261029
 assert len(first['model_state']['native']['optimizer'])==44 and int(first['model_state']['native']['optimizer'][0])==1
 assert sum(v.size for v in first['model_state']['native']['weights'])==3613324
 for name in ['weights','optimizer']:
  assert all(np.isfinite(v).all() for v in first['model_state']['native'][name])
 with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
 train=np.flatnonzero(np.isin(users,[3,4,5,6]));expected=ids[train[permutation(4700,0,20261029)[:12]]]
 np.testing.assert_array_equal(first['training_ids'],expected);assert len(set(first['training_ids']))==12
 assert not torch.equal(initial['model_state']['native']['torch_rng'],first['model_state']['native']['torch_rng']);equal(initial['model_state']['layer_rng'],first['model_state']['layer_rng'])
 r={'state':'immutable_full_Encoder_initial_first_update_verified','protocol_sha256':sha(B/'PROTOCOL.json'),'initial_sha256':sha(out/'initial.pt'),'first_update_sha256':sha(out/'first_update.pt'),'first_training_batch_IDs_sha256':hashlib.sha256(np.asarray(expected,dtype='int64').tobytes()).hexdigest(),'first_batch_size':12,'first_training_users':[3,4,5,6],'first_optimizer_step':1,'model_parameters':3613324,'optimizer_variables':44,'full_global_and_layer_RNG_preserved':True,'native_dropout_global_Torch_RNG_changed_layer_RNG_retained':True,'diagnostic_labels_read':False,'users7_8_labels_used':False,'complete_modelset_verified':False}
 (HERE/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
