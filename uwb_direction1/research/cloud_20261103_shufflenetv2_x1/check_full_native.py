from pathlib import Path
import io,json
from native_shufflenet import np,torch,make,optimizer,step,preprocess,independent_driver,encoder_digest
B=Path(__file__).resolve().parent;rng=np.random.default_rng(20261029);x=rng.normal(size=(3,3,90,189)).astype(np.float32);original=make(True)
with torch.inference_mode():a=original(preprocess(x));b=independent_driver(original,preprocess(x));torch.testing.assert_close(a,b,rtol=0,atol=0)
assert a.shape==(3,1000) and [len(original.stage2),len(original.stage3),len(original.stage4)]==[4,8,4];del original
m=make();opt=optimizer(m);g=torch.Generator().manual_seed(20261029);raw=rng.normal(size=(15,3,90,189)).astype(np.float32);labels=np.arange(15)%12;loss0=step(m,opt,raw,labels,g);buffer=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),augmentation_rng=g.get_state(),torch_rng=torch.get_rng_state()),buffer);size=buffer.tell();buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True);q=make();qo=optimizer(q);q.load_state_dict(saved['model'],strict=True);qo.load_state_dict(saved['optimizer']);qg=torch.Generator();qg.set_state(saved['augmentation_rng']);assert encoder_digest(m)==encoder_digest(q)
losses=[]
for i in range(2):
 raw=rng.normal(size=(15,3,90,189)).astype(np.float32);labels=(np.arange(15)+i)%12;torch.set_rng_state(saved['torch_rng']);left=step(m,opt,raw,labels,g);torch.set_rng_state(saved['torch_rng']);right=step(q,qo,raw,labels,qg);assert left==right;losses.append(left)
 for n,t in m.state_dict().items():torch.testing.assert_close(t,q.state_dict()[n],rtol=0,atol=0)
 for key,state in opt.state_dict()['state'].items():
  for n,t in state.items():torch.testing.assert_close(t,qo.state_dict()['state'][key][n],rtol=0,atol=0)
 torch.testing.assert_close(g.get_state(),qg.get_state(),rtol=0,atol=0)
r=dict(state='passed_full_official_ShuffleNetV2_x1_original_logits_and_full_B15_checkpoint_next2updates_exact',original_parameters=2278604,gesture12_trainable_parameters=1265904,stages_repeats=[4,8,4],stage_channels=[24,116,232,464,1024],original1000_logits_manual_channel_index_driver_exact=True,all_layers_trainable=True,batch_size=15,synthetic_input=[15,3,90,189],initial_loss=loss0,next2_losses=losses,full_model_BN_optimizer_momentum_and_RNG_exact=True,checkpoint_bytes=size,checkpoint_IO_in_RAM_only=True,actual_UWB_fit_started=False,formal_CPU_cost_started=False,accuracy_or_originality_claim=False)
(B/'NATIVE_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
