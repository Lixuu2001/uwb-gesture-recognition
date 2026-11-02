from pathlib import Path
import copy,io,json
from native_finetune import np,torch,make,optimizer,step,encoder_digest
B=Path(__file__).resolve().parent;rng=np.random.default_rng(20261029);raw=rng.normal(size=(15,3,90,189)).astype(np.float32);y=np.arange(15)%12;m=make();opt=optimizer(m);g=torch.Generator().manual_seed(20261029);loss0=step(m,opt,raw,y,g)
checkpoint=dict(model=m.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),torch_rng=torch.get_rng_state());buffer=io.BytesIO();torch.save(checkpoint,buffer);size=buffer.tell();buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True);q=make();qo=optimizer(q);q.load_state_dict(saved['model'],strict=True);qo.load_state_dict(saved['optimizer']);qg=torch.Generator();qg.set_state(saved['generator']);assert encoder_digest(q)==encoder_digest(m)
losses=[]
for i in range(2):
 x=rng.normal(size=(15,3,90,189)).astype(np.float32);labels=(np.arange(15)+i)%12;state=g.get_state().clone();torch.set_rng_state(saved['torch_rng']);left=step(m,opt,x,labels,g);torch.set_rng_state(saved['torch_rng']);right=step(q,qo,x,labels,qg);assert left==right
 for name,a in m.state_dict().items():torch.testing.assert_close(a,q.state_dict()[name],rtol=0,atol=0)
 sa=opt.state_dict()['state'];sb=qo.state_dict()['state'];assert sa.keys()==sb.keys()
 for key in sa:
  for name,a in sa[key].items():torch.testing.assert_close(a,sb[key][name],rtol=0,atol=0)
 torch.testing.assert_close(g.get_state(),qg.get_state(),rtol=0,atol=0);losses.append(left)
r=dict(state='passed_future_full_ResNet18_alllayer_SGDM_B15_checkpoint_next2_updates_exact_on_synthetic',trainable_parameters=11182668,batch_size=15,full_input_shape=[15,3,90,189],classes=12,initial_loss=loss0,next2_losses=losses,complete_model_BN_optimizer_momentum_and_augmentation_RNG_exact=True,checkpoint_bytes=size,checkpoint_IO_in_RAM_only=True,full40epochs_not_run=True,no_UWB_values_labels_fit=True,no_formal_cost_or_actual_training_started=True,current_frozen_encoder_C1_pipeline_unmodified=True,not_selected_before_current_study_closure=True,accuracy_or_originality_claim=False)
(B/'CHECKPOINT_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
