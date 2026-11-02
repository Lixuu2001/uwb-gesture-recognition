"""Immutable completed pretrained-arm audit while scratch continues.

Only validation-user2 maps/labels are indexed. Held1 predictions are hashed as
opaque bytes; their logits and diagnostic labels are never decoded or scored.
No training driver import, fit, new selection, or science-file modification.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from audit_complete import BASE,PHASES,read,digest,require_phase,identities,role_indices,verify_stage,configure_torch,checked,metric
def require_ready(base):
 require_phase(base,*PHASES[1])
 state=read(base/'run01/STATE.json')
 if 'pretrained' not in state.get('completed_arms',[]): raise RuntimeError('not_ready: pretrained arm incomplete')
 assert state['users7_8_labels_used'] is False
 arm=base/'run01/pretrained';done=read(arm/'ARM_COMPLETE.json')
 assert done['arm']=='pretrained' and done['classification_epochs']==100 and done['pretraining_epochs']==50
 assert digest(arm/'classification/best.pt')==done['checkpoint_sha256']
 assert digest(arm/'predictions_unscored.npz')==done['prediction_sha256']
 return done
def audit(base,output):
 done=require_ready(base) # Before any array/checkpoint load or Torch import.
 assert not output.exists();started=time.monotonic();configure_torch()
 cfg,identity,hashes=identities(base);run=base/'run01'
 for n,h in read(base/'AUDIT_HASHES.json').items(): checked(base/n,h,hashes)
 for n,h in read(base/'PRETRAINED_PHASE_AUDIT_HASHES.json').items(): checked(base/n,h,hashes)
 for path in (run/'IDENTITY.json',run/'split.json',run/'pretrained/ARM_COMPLETE.json',run/'pretrained/predictions_unscored.npz',run/'pretrained/pretrain/best.pt'):
  hashes[str(path)]=digest(path)
 with np.load(cfg['source_manifest']) as z: ids=z['ids'];users=z['users']
 ix=role_indices(read(run/'split.json'),ids,users)
 details,best=verify_stage(base,PHASES[1],cfg,identity,ids,ix,hashes)
 checked(cfg['source_maps'],cfg['source_maps_sha256'],hashes)
 label_path=Path(cfg['data'])/'cache/y_clutterremoved.npy';checked(label_path,cfg['input_sha256']['y_clutterremoved.npy'],hashes)
 # Manifest has users1..6, but only user2 values are actually indexed here.
 source=np.load(cfg['source_maps'],mmap_mode='r');assert source.shape==(6900,90,567) and source.dtype==np.float32
 labels=np.load(label_path,mmap_mode='r');val_index=ix['validation'];assert set(users[val_index])=={2}
 y=np.array(labels[ids[val_index]],copy=True)
 import torch
 from torch.nn import functional as F
 sys.path.insert(0,cfg['prototype']);from author_core import make_model
 model,_=make_model(cfg['seed']);model.load_state_dict(best['model']);model.eval();parts=[];loss_total=0.
 with torch.inference_mode():
  for offset in range(0,len(val_index),16):
   index=val_index[offset:offset+16];logits=model(torch.from_numpy(np.array(source[index],copy=True)).transpose(1,2).contiguous());assert torch.isfinite(logits).all();parts.append(logits.numpy())
   target=torch.from_numpy(y[offset:offset+16]).long();logprob=F.log_softmax(logits,dim=-1)
   loss=(.9*(-logprob.gather(1,target[:,None]).squeeze(1))+.1*(-logprob.mean(-1))).mean();loss_total+=float(loss)*len(index)
 macro=metric(y,np.concatenate(parts).argmax(1));loss=loss_total/len(val_index)
 assert abs(macro-best['validation_macro'])<1e-12
 np.testing.assert_allclose(loss,best['validation_loss'],rtol=1e-5,atol=1e-6)
 assert digest(run/'pretrained/predictions_unscored.npz')==done['prediction_sha256']
 for p,h in hashes.items(): assert digest(p)==h,'changed during immutable arm audit: '+p
 result=dict(state='pass',scope='completed_pretrained_classifier_validation_only',phase=details,validation_forward_records=1000,validation_user=2,recorded_selection_and_validation_macro_verified=True,smoothed_CE_independently_verified=True,SSL_best_to_classifier_initial_exact=True,classifier_optimizer_reset_verified=True,diagnostic_prediction_bytes_sha256=done['prediction_sha256'],diagnostic_values_decoded=False,diagnostic_labels_read_by_auditor=False,diagnostic_forward_or_score=False,users7_8_labels_used=False,real_training_or_new_selection=False,artifact_hashes=hashes,auditor_sha256=digest(__file__),audit_wall_seconds=time.monotonic()-started,limits='Phase/validation audit only, not full pilot completion or held-user accuracy evidence. Nonselected epochs not replayed. RNG reload proves schema, not full trajectory. Original stochastic SSL validation losses not replayed. Full both-arm audit and final preservation still required.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',scope=result['scope'],selected_epoch=best['epoch'],validation_records=1000)))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();audit(BASE,args.out)
