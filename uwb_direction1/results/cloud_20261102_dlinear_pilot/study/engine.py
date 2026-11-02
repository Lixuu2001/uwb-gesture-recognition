"""Complete source classification epoch semantics, validation-only early stopping."""
import sys,copy
sys.path.insert(0,'/workspace/uwb-dlinear-source-probe-20261102')
from native_dlinear import np,torch,F,make,forward,step,capture,restore as restore_model,equal
from torch.utils.data import TensorDataset,DataLoader
def values(x,n):
 if not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.shape!=(n,90,567) or n<1 or not np.isfinite(x).all():raise ValueError('Finite float32[N,90,567] required')
def labels(y,n):
 if not isinstance(y,np.ndarray) or y.dtype.kind not in 'iu' or y.shape!=(n,) or n<1 or np.any((y<0)|(y>=12)):raise ValueError('Fixed12 integer labels required')
def setup(seed,n,epochs,batch):
 if any(type(v) is not int for v in [seed,n,epochs,batch]) or n<1 or epochs<1 or batch<1:raise ValueError('Explicit settings required')
 m,opt,cls=make(seed);h=cls(patience=3,verbose=False);s=dict(seed=seed,train_count=n,epochs=epochs,batch_size=batch,input_shape=[90,567],source_blob='3f4d666a9ffe7fb6f58627ba43f1a9d3d9804d78',shuffle=True,drop_last=False,num_workers=0,selection='validation_micro_accuracy_latest_tie_patience3',loss='CrossEntropy',optimizer='RAdam.0001',gradient_clip=4.)
 return m,opt,h,s
def optimizer_steps(opt):
 steps=[int(x['step']) for x in opt.state.values()]
 if not steps:return 0
 assert len(set(steps))==1;return steps[0]
def snapshot(m,opt,h,s,identity,epoch,total_steps,batch=-1):
 if type(epoch) is not int or not 0<=epoch<=s['epochs'] or total_steps!=optimizer_steps(opt):raise ValueError('Wrong cursor')
 return dict(model_state=capture(m,opt),early=copy.deepcopy(h.__dict__),settings=copy.deepcopy(s),identity=copy.deepcopy(identity),epoch=epoch,total_steps=total_steps,batch=batch)
def restore(saved,m,opt,h,s,identity):
 if saved['settings']!=s or saved['identity']!=identity:raise ValueError('Wrong checkpoint identity/settings')
 steps=[int(v['step']) for v in saved['model_state']['optimizer']['state'].values()]
 if saved['total_steps']!=(steps[0] if steps else 0) or (steps and len(set(steps))!=1):raise ValueError('Wrong checkpoint optimizer cursor')
 restore_model(saved['model_state'],m,opt);h.__dict__.clear();h.__dict__.update(copy.deepcopy(saved['early']));assert optimizer_steps(opt)==saved['total_steps']
def loader(x,y,batch,shuffle):
 return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y.astype('int64',copy=False)),torch.arange(len(x))),batch_size=batch,shuffle=shuffle,drop_last=False,num_workers=0)
def validate(m,data):
 m.eval();losses=[];correct=0;count=0
 with torch.no_grad():
  for xb,yb,_ in data:
   out=forward(m,xb);losses.append(float(F.cross_entropy(out,yb)));correct+=int((out.argmax(1)==yb).sum());count+=len(yb)
 m.train();return float(np.mean(losses)),correct/count
def epoch(m,opt,h,s,x,y,vx,vy,index,out,after_batch=None):
 labels(y,len(x));labels(vy,len(vx))
 if len(x)!=s['train_count'] or type(index) is not int or not 0<=index<s['epochs'] or h.early_stop:raise ValueError('Wrong budget/shape or terminated early stopping')
 train=loader(x,y,s['batch_size'],True);val=loader(vx,vy,s['batch_size'],False);m.train();losses=[];records=[]
 for j,(xb,yb,rows) in enumerate(train):
  if not torch.isfinite(xb).all():raise FloatingPointError('Nonfinite training values')
  _,loss,norm=step(m,opt,xb,yb)
  if not all(torch.isfinite(p).all() for p in m.parameters()):raise FloatingPointError('Nonfinite parameters')
  losses.append(loss);r=dict(batch=j,indices=rows.numpy().copy(),batch_size=len(rows),total_steps=optimizer_steps(opt),batch_loss=loss,preclip_gradient_norm=norm);records.append(r)
  if after_batch is not None:after_batch(r)
 vl,va=validate(m,val);tl,ta=validate(m,val)
 if not np.isfinite([*losses,vl,va,tl,ta]).all():raise FloatingPointError('Nonfinite epoch metrics')
 assert vl==tl and va==ta;np.testing.assert_array_equal(np.sort(np.concatenate([r['indices'] for r in records])),np.arange(len(x)));previous=h.best_score;improved=previous is None or va>=previous
 h(-va,m,str(out));assert h.best_score is not None
 return dict(logs=dict(loss=float(np.mean(losses)),val_loss=vl,val_accuracy=va,second_validation_loss=tl,second_validation_accuracy=ta),batch_records=records,total_steps=optimizer_steps(opt),improved=improved,early_stop=h.early_stop,early_counter=h.counter)
def predict(m,x,batch):
 values(x,len(x));data=loader(x,np.zeros(len(x),dtype='int64'),batch,False);m.eval();out=[]
 with torch.no_grad():
  for xb,_,_ in data:out.append(forward(m,xb).numpy())
 a=np.concatenate(out)
 if a.shape!=(len(x),12) or not np.isfinite(a).all():raise FloatingPointError('Invalid logits')
 return a
