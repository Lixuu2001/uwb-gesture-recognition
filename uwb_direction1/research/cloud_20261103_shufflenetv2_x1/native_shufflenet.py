"""Full official ShuffleNetV2 x1.0 pretrained transfer baseline, not new prior art."""
from pathlib import Path
import sys,ast,json,hashlib
from typing import Any,Callable,Optional
sys.path.insert(0,'/workspace/uwb-resnet18-finetune-source-20261102')
from native_finetune import np,torch,nn,preprocess,augmented_pixels,optimizer,step,encoder_digest
from torch import Tensor
B=Path(__file__).resolve().parent
_log_api_usage_once=lambda *a,**k:None
_raw=(B/'local_author_sources/shufflenetv2.py').read_bytes()
assert hashlib.sha1(b'blob '+str(len(_raw)).encode()+b'\0'+_raw).hexdigest()=='96736f6a7ac289102ef7a57cb4cbb960c02c625e'
_names=['channel_shuffle','InvertedResidual','ShuffleNetV2'];_nodes=[n for n in ast.parse(_raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in _names];assert [n.name for n in _nodes]==_names
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(B/'local_author_sources/shufflenetv2.py'),'exec'),globals())
def make(keep_original_head=False):
 p=B/'weights/shufflenetv2_x1-5666bf0f80.pth';meta=json.loads((B/'WEIGHTS_SOURCE.json').read_bytes());assert p.stat().st_size==meta['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];torch.manual_seed(20261029)
 m=ShuffleNetV2([4,8,4],[24,116,232,464,1024]);assert sum(p.numel() for p in m.parameters())==2278604;m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
 if not keep_original_head:torch.manual_seed(20261029);m.fc=nn.Linear(1024,12)
 m.requires_grad_(True)
 if keep_original_head:m.eval()
 else:m.train();assert sum(p.numel() for p in m.parameters())==1265904
 assert all(p.requires_grad for p in m.parameters());return m
def independent_driver(m,x):
 x=m.maxpool(m.conv1(x))
 for stage in [m.stage2,m.stage3,m.stage4]:
  for block in stage:
   if block.stride==1:a,b=x.chunk(2,1);x=torch.cat([a,block.branch2(b)],1)
   else:x=torch.cat([block.branch1(x),block.branch2(x)],1)
   channels=x.shape[1];indices=torch.arange(channels).reshape(2,channels//2).t().reshape(-1);x=x.index_select(1,indices)
 x=m.conv5(x);return m.fc(x.mean((2,3)))
