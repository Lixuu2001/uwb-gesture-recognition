"""Atomic evidence writes and label-blind native file observation isolation."""
from pathlib import Path
import json
def write(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def save(p,obj,torch):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)
def isolate_native(m,action):
 from engine import capture_model,restore_model,equal
 before=capture_model(m)
 try:
  result=action();after=capture_model(m)
  for k in ['weights','optimizer','metrics']:equal(before['native'][k],after['native'][k])
  equal(before['layer_rng'],after['layer_rng']);equal(before['module_training_modes'],after['module_training_modes'])
 finally:restore_model(before,m)
 equal(before,capture_model(m));return result
