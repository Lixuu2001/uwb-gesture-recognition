"""Per-role generated full-native checkpoint continuation; no real dataset opens."""
from pathlib import Path
import tempfile,copy
from common import B,read,write,sha
import engine as e

def main():
 assert not(B/'SYNTHETIC_RESULT.json').exists()
 m,o,g=e.make();n=16;ids=e.np.arange(n,dtype='int64');users=e.np.array(e.c['train_users'])[ids%4]
 values=e.np.random.default_rng(808).normal(size=(n,90,567)).astype('float32');y=ids%12
 saved=None;reference=None
 with tempfile.TemporaryDirectory() as folder:
  path=Path(folder)/'resume.pt'
  for epoch in range(1,9):
   m.train();pos=e.order(n,epoch)[:8];e.step(m,o,values[pos],y[pos],epoch,users[pos]);o.epoch_end(epoch,.5,.9-.05*epoch)
   state=copy.deepcopy(e.capture(m,o,g,epoch,epoch))
   if epoch==7:e.atomic_save(path,state);saved=copy.deepcopy(state)
   if epoch==8:reference=state
  m2,o2,g2=e.make();st=e.torch.load(path,map_location='cpu',weights_only=True);assert e.semantic(st)==e.semantic(saved);e.restore(st,m2,o2,g2)
  m2.train();pos=e.order(n,8)[:8];e.step(m2,o2,values[pos],y[pos],8,users[pos]);o2.epoch_end(8,.5,.5)
  assert e.semantic(e.capture(m2,o2,g2,8,8))==e.semantic(reference)
  a=copy.deepcopy(o.swad).finalize();b=copy.deepcopy(o2.swad).finalize();assert e.semantic(a.state_dict())==e.semantic(b.state_dict()) and a.start_step==b.start_step and a.end_step==b.end_step
 from loso_roles_metrics import roles
 allusers=e.np.repeat(e.np.arange(1,7),3);v,loc=roles(allusers,e.c['held']);assert v==e.c['validation_user'] and set(allusers[loc['train']])==set(e.c['train_users'])
 result=dict(state='pass',held=e.c['held'],checks=dict(full_native617612=True,all8_dense_steps=True,native_slow_step6_and_SWAD_full_queues_disk7_to_exact8=True,all_RNG_and_modes_restored=True,finalize_exact=True,source_roles_disjoint=True,generated_only_no_real_IDs_maps_or_labels=True),source_sha256={n:sha(Path(__file__).parent/n) for n in ['engine.py','synthetic.py']},new_UWB_fit_started=False)
 write(B/'SYNTHETIC_RESULT.json',result);print(result)
if __name__=='__main__':main()
