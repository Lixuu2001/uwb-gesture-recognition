"""Reject corrupted IDs/roles/taxonomy before any optimizer update (synthetic)."""
import json,tempfile,copy
from unittest.mock import patch
from source_engine import *
B=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with tempfile.TemporaryDirectory() as d:
  e=Engine(output=d,length=90,channels=567,classes=12,epochs=3,batch_size=4,seed=19,identity=dict(synthetic_only=True));x=np.zeros((6,90,567),dtype='float32');y=np.arange(6,dtype='int64');ids=np.arange(6)+100;order=np.arange(6);vx=np.zeros((3,90,567),dtype='float32');vy=np.arange(3);vids=np.arange(3)+500;vo=np.arange(3)
  valid=dict(epoch=1,values=x,labels=y,ids=ids,order=order,validation=(vx,vy,vids,vo));cases=[]
  for changes in [dict(order=np.zeros(6,dtype='int64')),dict(ids=np.zeros(6,dtype='int64')),dict(validation=(vx,vy,ids[:3],vo)),dict(labels=np.array([0,1,2,3,4,12])),dict(validation=(vx,np.array([0,1,12]),vids,vo)),dict(validation=(vx[:2],vy,vids,vo)),dict(validation=(vx,vy,vids,np.array([0,0,2]))),dict(validation=(np.full_like(vx,np.nan),vy,vids,vo))]:
   bad=dict(valid);bad.update(changes);cases.append(bad)
  initial=e.checkpoint(epoch=0,batch=-1,total_steps=0,sampler={})
  with patch.object(e.model,'train_on_batch',side_effect=AssertionError('Must refuse before training')):
   for bad in cases:
    try:e.epoch(**bad)
    except ValueError:pass
    else:raise AssertionError('Invalid inputs accepted')
    equal(initial,e.checkpoint(epoch=0,batch=-1,total_steps=0,sampler={}))
  try:e.checkpoint(epoch=0,batch=-1,total_steps=1,sampler={})
  except ValueError:pass
  else:raise AssertionError('Wrong optimizer cursor accepted')
  e.close()
 result=dict(state='pass_synthetic_input_identity_guards',invalid_role_order_taxonomy_or_finite_cases_refused_before_update=len(cases),all_model_Adam_BN_callbacks_metrics_modulemodes_RNG_unchanged=True,wrong_checkpoint_Adam_iteration_refused=True,real_UWB_values_labels_or_fit=False)
 (B/'INPUT_GUARDS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
