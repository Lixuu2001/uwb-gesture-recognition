"""Exact author TS2Vec minibatch AST with complete resumable state helpers.

No data loader, real inputs or fitting launcher. Model, crop, masks, loss,
AdamW and initial-inclusive average operations remain the pinned author AST.
Only loss return and caller-owned sampler/checkpoint bookkeeping are added.
"""
import ast,copy,random,sys
from pathlib import Path
import numpy as np
import torch
SCREEN=Path('/workspace/uwb-ts2vec-screen-20261030')
sys.path.insert(0,str(SCREEN))
from author_adapter import make_model,source_check
import ts2vec as original
def training_step():
 tree=ast.parse((SCREEN/'vendor/ts2vec/ts2vec.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TS2Vec');fit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
 loop=next(n for n in ast.walk(fit) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='batch')
 begin=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='x' for t in n.targets) and isinstance(n.value,ast.Subscript) and isinstance(n.value.value,ast.Name) and n.value.value.id=='batch')+1
 end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.net.update_parameters')+1
 prototype=ast.parse('def step(self,x,optimizer):\n return loss\n').body[0];prototype.body=copy.deepcopy(loop.body[begin:end])+prototype.body
 module=ast.fix_missing_locations(ast.Module(body=[prototype],type_ignores=[]));namespace=dict(vars(original));exec(compile(module,'exact_author_minibatch_AST','exec'),namespace)
 return namespace['step']
STEP=training_step()
def optimizer(model):return torch.optim.AdamW(model._net.parameters(),lr=model.lr)
def checkpoint(model,opt,step,sampler,identity):
 return dict(raw_model=model._net.state_dict(),averaged_model=model.net.state_dict(),optimizer=opt.state_dict(),step=step,n_iters=model.n_iters,n_epochs=model.n_epochs,sampler=sampler,identity=identity,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
def restore(model,opt,saved):
 model._net.load_state_dict(saved['raw_model']);model.net.load_state_dict(saved['averaged_model']);opt.load_state_dict(saved['optimizer']);model.n_iters=saved['n_iters'];model.n_epochs=saved['n_epochs']
 assert model.n_iters==saved['step'] and int(model.net.n_averaged)==saved['step']+1
 torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
 return saved['step'],saved['sampler']
def batch_indices(step,seed=20261029,n=4700,batch=16):
 assert 1<=step<=600 and n==4700 and batch==16
 per_epoch=n//batch;epoch=(step-1)//per_epoch+1;offset=(step-1)%per_epoch*batch
 order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).numpy()
 return order[offset:offset+batch],dict(epoch=epoch,offset=offset,next_offset=offset+batch,order_seed=seed+epoch)
