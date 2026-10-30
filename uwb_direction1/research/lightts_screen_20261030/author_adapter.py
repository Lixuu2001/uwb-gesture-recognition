"""Full library-maintainer UWaveGestureLibrary LightTS preset; no real-data launcher."""
import ast,random,shlex,sys,time
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
import torch.nn as nn
B=Path(__file__).resolve().parent;SOURCE=B/'vendor/tslib';sys.path.insert(0,str(SOURCE))
def source_check():
 for row in json.loads((B/'SOURCE_INDEX.json').read_text())['files']:
  data=(SOURCE/row['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
def configuration():
 script=(SOURCE/'scripts/classification/LightTS.sh').read_text();block=[v for v in script.split('\n\n') if '--model_id UWaveGestureLibrary' in v];assert len(block)==1
 words=shlex.split(block[0].replace('\\\n',' '));options={words[i][2:]:words[i+1] for i in range(len(words)-1) if words[i].startswith('--')}
 fixed={k:int(options[k]) for k in ['e_layers','batch_size','d_model','d_ff','top_k','train_epochs','patience']}
 assert fixed==dict(e_layers=3,batch_size=16,d_model=128,d_ff=256,top_k=3,train_epochs=100,patience=10)
 return SimpleNamespace(**fixed,task_name='classification',seq_len=90,label_len=48,pred_len=0,enc_in=567,num_class=12,num_kernels=6,dropout=.1,embed='timeF',freq='h',learning_rate=float(options['learning_rate']))
def make(seed=20261029):
 source_check();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 from models.LightTS import Model
 cfg=configuration();model=Model(cfg).float();opt=torch.optim.RAdam(model.parameters(),lr=cfg.learning_rate)
 return cfg,model,opt
def original_inner_loop():
 tree=ast.parse((SOURCE/'exp/exp_classification.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Exp_Classification');train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 return next(n for n in ast.walk(train) if isinstance(n,ast.For) and ast.unparse(n.target)=='(i, (batch_x, label, padding_mask))')
def step_function():
 loop=original_inner_loop();body=[]
 for n in loop.body:
  if isinstance(n,(ast.AugAssign,ast.If)):continue
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='train_loss.append':continue
  body.append(n)
 proto=ast.parse('def step(self,batch_x,label,padding_mask,model_optim,criterion):\n return loss').body[0];proto.body=body+proto.body
 ns=dict(torch=torch,nn=nn);exec(compile(ast.fix_missing_locations(ast.Module(body=[proto],type_ignores=[])),'exact_author_training_inner_AST','exec'),ns);return ns['step']
_STEP=step_function()
def step(model,opt,x,y):
 model.train();return _STEP(SimpleNamespace(model=model,device=torch.device('cpu')),x,y.reshape(-1,1),torch.ones(x.shape[:2]),opt,nn.CrossEntropyLoss())
def early_stopper(patience=10):
 tree=ast.parse((SOURCE/'utils/tools.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EarlyStopping');ns=dict(np=np,torch=torch);exec(compile(ast.Module(body=[node],type_ignores=[]),'exact_author_EarlyStopping_AST','exec'),ns);return ns['EarlyStopping'](patience=patience)
def snapshot(model,opt,stopper,epoch,step_count,sampler=None):
 return dict(model=model.state_dict(),optimizer=opt.state_dict(),early_stopper=dict(stopper.__dict__),epoch=epoch,total_steps=step_count,sampler=sampler,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),model_training=model.training)
def restore(state,model,opt,stopper):
 model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);stopper.__dict__.update(state['early_stopper']);model.train(state['model_training']);torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
def equal(a,b):
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a)==type(b) and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
