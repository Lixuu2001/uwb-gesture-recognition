"""Immutable TCN first state verification; no gesture labels or model fitting."""
from pathlib import Path
import sys,json,hashlib
B=Path('/workspace/uwb-tcn-pilot-20261102');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from guards import ready,sha
def main():
 c=ready();out=B/'run01'
 from engine import np,torch,make
 a=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);f=torch.load(out/'first_update.pt',map_location='cpu',weights_only=False)
 assert a['total_steps']==0 and f['total_steps']==1 and f['epoch']==1 and f['batch']==0
 assert a['identity']==f['identity'] and f['identity']['protocol_sha256']==sha(B/'PROTOCOL.json') and f['identity']['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
 assert a['settings']==f['settings'] and (f['settings']['epochs'],f['settings']['batch_size'],f['settings']['seed'])==(20,64,20261029)
 m,opt=make(c['seed']);m.load_state_dict(f['model_state']['model']);opt.load_state_dict(f['model_state']['optimizer'])
 assert sum(p.numel() for p in m.parameters())==180162 and len(list(m.parameters()))==52 and len(m.state_dict())==100 and len(opt.state)==52
 assert all(int(v['step'])==1 for v in opt.state.values()) and all(torch.isfinite(v).all() for v in m.state_dict().values())
 assert all(torch.isfinite(v).all() for d in opt.state.values() for v in d.values() if isinstance(v,torch.Tensor))
 with np.load(c['source_manifest']) as z:ids=z['ids'];users=z['users']
 expected=ids[np.flatnonzero(np.isin(users,[3,4,5,6]))[:64]];np.testing.assert_array_equal(f['training_ids'],expected);assert len(set(expected.tolist()))==64
 assert not torch.equal(a['model_state']['torch_rng'],f['model_state']['torch_rng'])
 r=dict(state='full_TCN_immutable_initial_first_update_verified',initial_sha256=sha(out/'initial.pt'),first_update_sha256=sha(out/'first_update.pt'),first_batch_IDs_sha256=hashlib.sha256(np.asarray(expected,dtype='int64').tobytes()).hexdigest(),batch_size=64,first_optimizer_step=1,parameter_count=180162,trainable_tensors=52,Adam_parameter_states=52,state_dict_keys=100,source_no_shuffle_canonical_first_batch=True,global_Torch_RNG_changed_with_native_loader_and_dropout=True,diagnostic_labels_read=False,users7_8_labels_used=False,complete_modelset_verified=False)
 (HERE/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
