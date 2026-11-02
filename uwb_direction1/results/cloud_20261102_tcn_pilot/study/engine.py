"""Full author TCN, native source-order DataLoader and released LR schedule."""
import sys,copy
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-tcn-source-probe-20261102')
from native_tcn import np,torch,F,make,capture,restore as restore_model,equal
from torch.utils.data import TensorDataset,DataLoader
def values(x,n):
 if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.shape!=(n,90,567) or n<1 or not np.isfinite(x).all():raise ValueError('Finitefloat32[N,90,567] required')
def labels(y,n):
 if not isinstance(y,np.ndarray) or y.dtype.kind not in 'iu' or y.shape!=(n,) or n<1 or np.any((y<0)|(y>=12)):raise ValueError('Fixed12 integer labels required')
def setup(seed,n,epochs,batch):
 if any(type(v) is not int for v in [seed,n,epochs,batch]) or n<1 or epochs<1 or batch<1:raise ValueError('Explicit settings')
 m,opt=make(seed);s=dict(seed=seed,train_count=n,epochs=epochs,batch_size=batch,source_blob='b377e20b58e804443c2a34faf34cf2e5d27a0815',input_shape=[567,90],shuffle=False,drop_last=False,num_workers=0,selection='final',loss='NLLLoss')
 return m,opt,s
def optimizer_steps(opt):
 steps=[int(x['step']) for x in opt.state.values()]
 if not steps:return 0
 assert len(set(steps))==1;return steps[0]
def snapshot(m,opt,settings,identity,epoch,total_steps,batch=-1):
 if type(epoch) is not int or not 0<=epoch<=settings['epochs'] or total_steps!=optimizer_steps(opt):raise ValueError('Bad cursor')
 return dict(model_state=capture(m,opt),settings=copy.deepcopy(settings),identity=copy.deepcopy(identity),epoch=epoch,total_steps=total_steps,batch=batch)
def restore(saved,m,opt,settings,identity):
 steps=[int(x['step']) for x in saved['model_state']['optimizer']['state'].values()]
 if saved['settings']!=settings or saved['identity']!=identity or saved['total_steps']!=(steps[0] if steps else 0) or (steps and len(set(steps))!=1):raise ValueError('Wrong checkpoint identity/settings/cursor')
 restore_model(saved['model_state'],m,opt)
def epoch(m,opt,s,x,y,vx,vy,index,after_batch=None):
 labels(y,len(x));labels(vy,len(vx))
 if len(x)!=s['train_count'] or type(index) is not int or not 0<=index<s['epochs']:raise ValueError('Budget/shape mismatch')
 train=DataLoader(TensorDataset(torch.from_numpy(x).transpose(1,2),torch.from_numpy(y.astype('int64',copy=False)),torch.arange(len(x))),batch_size=s['batch_size'],shuffle=False,drop_last=False,num_workers=0)
 val=DataLoader(TensorDataset(torch.from_numpy(vx).transpose(1,2),torch.from_numpy(vy.astype('int64',copy=False))),batch_size=s['batch_size'],shuffle=False,drop_last=False,num_workers=0)
 lr=float(opt.param_groups[0]['lr']);m.train();total_loss=0.;correct=0;records=[]
 for j,(xb,yb,rows) in enumerate(train):
  if not torch.isfinite(xb).all():raise FloatingPointError('Nonfinite train values')
  opt.zero_grad();out=m(xb);loss=F.nll_loss(out,yb)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite NLL')
  loss.backward()
  if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('Nonfinite gradients')
  opt.step()
  if not all(torch.isfinite(p).all() for p in m.parameters()):raise FloatingPointError('Nonfinite parameters')
  total_loss+=float(loss.detach())*len(rows);correct+=int((out.argmax(1)==yb).sum());r=dict(batch=j,indices=rows.numpy().copy(),batch_size=len(rows),total_steps=optimizer_steps(opt),batch_loss=float(loss.detach()));records.append(r)
  if after_batch is not None:after_batch(r)
 m.eval();vl=0.;vc=0
 with torch.no_grad():
  for xb,yb in val:
   if not torch.isfinite(xb).all():raise FloatingPointError('Nonfinite validation')
   out=m(xb);loss=F.nll_loss(out,yb,reduction='sum');vl+=float(loss);vc+=int((out.argmax(1)==yb).sum())
 if not np.isfinite(vl):raise FloatingPointError('Nonfinite val NLL')
 if (index+1)%10==0:
  for g in opt.param_groups:g['lr']/=10
 return dict(logs=dict(loss=total_loss/len(x),accuracy=correct/len(x),val_loss=vl/len(vx),val_accuracy=vc/len(vx)),batch_records=records,total_steps=optimizer_steps(opt),learning_rate_before=lr,learning_rate_after=float(opt.param_groups[0]['lr']))
def predict(m,x,batch):
 values(x,len(x));loader=DataLoader(TensorDataset(torch.from_numpy(x).transpose(1,2)),batch_size=batch,shuffle=False,drop_last=False,num_workers=0);m.eval();out=[]
 with torch.no_grad():
  for (xb,) in loader:out.append(m(xb).numpy())
 a=np.concatenate(out)
 if a.shape!=(len(x),12) or not np.isfinite(a).all() or np.any(a>1e-6) or not np.allclose(np.exp(a).sum(1),1,rtol=1e-5,atol=1e-6):raise FloatingPointError('Invalid log probabilities')
 return a
def isolate(m,opt,action):
 before=capture(m,opt)
 try:
  result=action();after=capture(m,opt)
  for k in ['model','optimizer','modes']:equal(before[k],after[k])
 finally:restore_model(before,m,opt)
 equal(before,capture(m,opt));return result
