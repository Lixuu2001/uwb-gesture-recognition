from pathlib import Path
import time,json
from common import B,ready,write,sha,load_inputs
def main():
 c=ready();out=B/'run01';out.mkdir(exist_ok=False);start=time.monotonic()
 from engine import np,torch,make,run_training,evaluate,atomic_checkpoint
 ids,users,loc,maps,y,v=load_inputs(c);np.savez(out/'split.npz',ids=ids,users=users,**loc)
 m=make();torch.save(m.fc.state_dict(),out/'initial12_head.pt');del m
 write(out/'IDENTITY.json',dict(protocol_sha256=sha(B/'PROTOCOL.json'),source_ids_sha256=c['source_manifest_sha256'],seed=20261029,training_users=[3,4,5,6],validation_user=2,diagnostic_user=1,validation_labels_for_checkpoint_selection=True,diagnostic_labels_read=False,users7_8_labels_used=False))
 with (out/'steps.jsonl').open('x',buffering=1) as log:
  def on_step(r):log.write(json.dumps(r,separators=(',',':'))+'\n')
  def on_epoch(r,scores,state,best,improved):
   epoch=r['epoch'];np.savez_compressed(out/f'validation_epoch{epoch:02d}.npz',ids=ids[loc['validation']],scores=scores,predictions=scores.argmax(1));atomic_checkpoint(out/'final_full.pt',state)
   if improved:atomic_checkpoint(out/'best_model.pt',dict(model=best,epoch=r['selected_epoch'],validation_macro=r['best_validation_macro']))
   write(out/f'epoch{epoch:02d}.json',r);write(out/'STATE.json',dict(state='training_full_40_epochs',epoch=epoch,wall_seconds=time.monotonic()-start,diagnostic_labels_read=False));print(json.dumps(r),flush=True)
  m,opt,best,epoch,acc,initial=run_training(maps,loc,y,v,on_step,on_epoch)
 m.load_state_dict(best,strict=True)
 for role in ['validation','diagnostic']:
  scores=evaluate(m,maps,loc[role]);np.savez_compressed(out/(role+'_unscored.npz'),ids=ids[loc[role]],scores=scores,predictions=scores.argmax(1))
 write(out/'TRAIN_SELECTION.json',dict(epochs=40,updates=12560,selected_epoch=epoch,best_validation_macro=acc,initial_full_model_state_sha256=initial,selection='maxvalidationmacro_earliesttie',all_layers_trainable=True,diagnostic_labels_read=False,users7_8_labels_used=False))
 write(out/'STATE.json',dict(state='producer_complete_diagnostic_unscored',wall_seconds=time.monotonic()-start,diagnostic_labels_read=False));write(out/'FILES.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='FILES.json'})
if __name__=='__main__':main()
