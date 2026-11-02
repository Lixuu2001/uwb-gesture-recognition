"""Unmodified author model and explicit source epoch semantics; no UWB config.

No defaults select a real experiment seed, epoch budget, batch or checkpoint gate.
The future real entry must preregister those separately after the current audit.
"""
import sys,copy,random
from pathlib import Path
SOURCE=Path('/workspace/uwb-oscnn-source-probe-20261031');sys.path.insert(0,str(SOURCE))
from author_source_probe import torch,np,check_source,OS_CNN,generate_layer_parameter_list,equal
def build(input_channels,length,class_count,seed):
 check_source();torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 budget=[8*128*input_channels,5*128*256+2*256*128]
 layers=generate_layer_parameter_list(1,min(length//4,89),budget,in_channel=input_channels)
 m=OS_CNN(layers,class_count,False)
 o=torch.optim.Adam(m.parameters(),lr=.001)
 s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,'min',factor=.5,patience=50,min_lr=.0001)
 return Engine(m,o,s,layers)
class Engine:
 def __init__(self,model,optimizer,scheduler,layers):self.model=model;self.optimizer=optimizer;self.scheduler=scheduler;self.layers=layers
 def update(self,x,y):
  assert self.model.training
  self.optimizer.zero_grad();z=self.model(x);loss=torch.nn.functional.cross_entropy(z,y);loss.backward();assert torch.isfinite(loss)
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters());self.optimizer.step()
  return loss,z.detach().clone()
 def train_epoch(self,batches,observer=None):
  losses=[]
  for batch,sample in enumerate(batches):
   last,z=self.update(sample[0],sample[1]);losses.append(float(last.detach()))
   if observer is not None:observer(batch,sample,losses[-1])
  assert losses
  # Source wrapper schedules on LAST batch loss, not epoch average.
  self.scheduler.step(last.detach())
  return dict(batch_losses=losses,last_batch_loss=losses[-1],learning_rate=self.optimizer.param_groups[0]['lr'])
 def evaluate(self,batches):
  # This is a state-mutating evaluation: every source forward zeroes mask holes.
  self.model.eval();pred=[];truth=[]
  with torch.no_grad():
   for sample in batches:
    pred.extend(self.model(sample[0]).argmax(1).cpu().tolist());truth.extend(sample[1].cpu().tolist())
  assert len(pred)==len(truth)>0
  return float(np.mean(np.asarray(pred)==np.asarray(truth)))
 def epoch(self,batches,validation,epoch,evaluation_period,checkpoint,observer=None):
  assert epoch>=1 and evaluation_period>=1
  result=self.train_epoch(batches,observer)
  if epoch%evaluation_period==0:
   result.update(training_accuracy=self.evaluate(batches),validation_accuracy=self.evaluate(validation));self.model.train()
   torch.save(self.model.state_dict(),checkpoint)
  return result
 def snapshot(self,identity,sampler):
  return copy.deepcopy(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),training=self.model.training,layers=self.layers,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),identity=identity,sampler=sampler))
 def restore(self,saved,expected_identity=None):
  if expected_identity is not None:equal(saved['identity'],expected_identity)
  assert self.layers==saved['layers'];self.model.load_state_dict(saved['model']);self.optimizer.load_state_dict(saved['optimizer']);self.scheduler.load_state_dict(saved['scheduler']);self.model.train(saved['training']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
