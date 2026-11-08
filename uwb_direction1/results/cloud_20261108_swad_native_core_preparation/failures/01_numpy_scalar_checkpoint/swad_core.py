"""Pinned published SWAD LossValley/default3/6/.3 CPU adapter and plain-state journal.

This uses the original author's LossValley and AveragedModel parameter arithmetic.
Only the hard-coded final .cuda() is redirected to CPU. Not a new algorithm and
not a reproduction of the author's pretrained/frozen-BN image benchmark.
"""
from pathlib import Path
import ast,collections,copy,contextlib,hashlib,importlib.util,io,json,math,types
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
receipt=json.loads((ROOT/'AUTHOR_SOURCE_RECEIPT.json').read_text())
assert receipt['commit']=='252190eb881169e736beccf51af5caaa5c5cf666'
for f in receipt['files']:
 p=ROOT/'author_reference'/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
spec=importlib.util.spec_from_file_location('pinned_swad_average_cpu',ROOT/'author_reference/domainbed/lib/swa_utils.py')
official_average=importlib.util.module_from_spec(spec);spec.loader.exec_module(official_average)
class CPUAverage(official_average.AveragedModel):
 def cuda(self,*args,**kwargs):
  assert not args and not kwargs,'CPU-only adapter; no arbitrary device override'
  return self
source=ast.parse((ROOT/'author_reference/domainbed/swad.py').read_text())
classes=[x for x in source.body if isinstance(x,ast.ClassDef) and x.name in ['SWADBase','LossValley']]
ns=dict(__name__='pinned_swad_lossvalley_cpu',copy=copy,deque=collections.deque,np=np,swa_utils=types.SimpleNamespace(AveragedModel=CPUAverage))
exec(compile(ast.Module(body=classes,type_ignores=[]),'pinned_SWAD_252190eb_LossValley','exec'),ns)
LossValley=ns['LossValley']
class _Logger:
 def __init__(self):self.messages=[]
 def error(self,s):self.messages.append(str(s))
def _pack(a):
 if a is None:return None
 return dict(model={k:v.detach().clone() for k,v in a.module.state_dict().items()},module_modes={k:m.training for k,m in a.module.named_modules()},n_averaged=a.n_averaged.detach().clone(),start_step=int(a.start_step),end_step=int(a.end_step),end_loss=float(a.end_loss) if hasattr(a,'end_loss') else None)
def _unpack(state,template):
 if state is None:return None
 a=CPUAverage(template);a.module.load_state_dict(state['model'],strict=True)
 assert state['n_averaged'].shape==() and state['n_averaged'].dtype==torch.long and state['n_averaged'].item()>=0
 a.n_averaged.copy_(state['n_averaged']);a.start_step=state['start_step'];a.end_step=state['end_step']
 modules=dict(a.module.named_modules());assert modules.keys()==state['module_modes'].keys()
 for name,m in modules.items():m.training=state['module_modes'][name]
 if state['end_loss'] is not None:a.end_loss=state['end_loss']
 return a
class NativeSWADState:
 """No optimizer/model mutation: collect every post-native-update parameter state.

One segment per full epoch, independent held-user validation BCE for LossValley.
Continue the fixed64 native trajectory if the valley dies; collect the fixed dense
last-half tail33..64 control on that same trajectory. BN calibration is intentionally
NOT hidden here and is not implemented in this core-only preparation package.
 """
 def __init__(self,model):
  self.segment=CPUAverage(model);self.tail=CPUAverage(model);self.logger=_Logger()
  self.valley=LossValley(types.SimpleNamespace(logger=self.logger),3,6,.3)
  self.updates=0;self.last_epoch=0;self.finalized=False;self.events=[]
 def post_step(self,model,completed_updates,epoch):
  assert not self.finalized and model.training and completed_updates==self.updates+1 and epoch==self.last_epoch+1
  self.segment.update_parameters(model,step=completed_updates)
  if epoch>=33:self.tail.update_parameters(model,step=completed_updates)
  self.updates=completed_updates
 def epoch_end(self,model,epoch,validation_macro,validation_BCE):
  assert not self.finalized and epoch==self.last_epoch+1 and self.segment.n_averaged.item()>0
  assert math.isfinite(validation_BCE) and validation_BCE>=0 and 0<=validation_macro<=1
  assert self.segment.end_step==self.updates
  out=io.StringIO()
  with contextlib.redirect_stdout(out):self.valley.update_and_evaluate(self.segment,validation_macro,validation_BCE,lambda *args:None)
  self.events.extend(out.getvalue().splitlines());self.segment=CPUAverage(model);self.last_epoch=epoch
 def finalize(self):
  assert not self.finalized and self.last_epoch>0
  out=io.StringIO()
  with contextlib.redirect_stdout(out):a=self.valley.get_final_model()
  self.events.extend(out.getvalue().splitlines());self.finalized=True
  return copy.deepcopy(a)
 def state_dict(self):
  v=self.valley
  return dict(version=1,defaults=(3,6,.3),updates=self.updates,last_epoch=self.last_epoch,finalized=self.finalized,segment=_pack(self.segment),tail=_pack(self.tail),converge_Q=[_pack(a) for a in v.converge_Q],smooth_Q=[_pack(a) for a in v.smooth_Q],final_model=_pack(v.final_model),converge_step=v.converge_step,dead_valley=v.dead_valley,threshold=v.threshold,events=list(self.events),logger_messages=list(self.logger.messages))
 def load_state_dict(self,s,template):
  assert s['version']==1 and tuple(s['defaults'])==(3,6,.3) and isinstance(s['updates'],int) and s['updates']>=0 and s['last_epoch']>=0
  assert len(s['converge_Q'])<=3 and len(s['smooth_Q'])<=6
  self.segment=_unpack(s['segment'],template);self.tail=_unpack(s['tail'],template)
  v=self.valley;v.converge_Q=collections.deque((_unpack(a,template) for a in s['converge_Q']),maxlen=3);v.smooth_Q=collections.deque((_unpack(a,template) for a in s['smooth_Q']),maxlen=6)
  v.final_model=_unpack(s['final_model'],template);v.converge_step=s['converge_step'];v.dead_valley=s['dead_valley'];v.threshold=s['threshold']
  self.updates=s['updates'];self.last_epoch=s['last_epoch'];self.finalized=s['finalized'];self.events=list(s['events']);self.logger.messages=list(s['logger_messages'])
