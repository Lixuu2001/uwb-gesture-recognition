"""Frozen source-held model shift sensitivity; never scores class labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audit_restored import digest,dump,quality_mask
from circular_signal import SignalNet,get_batch
from audit_circular_signal import independent_forward
P=Path(__file__).resolve().parents[1]/'research/SHIFT_DIAGNOSTIC_PROTOCOL.json'

def shift(x,v,off):
 z=x.clone();z[:,v]=0
 if off==1:z[:,v,...,1:]=x[:,v,...,:-1]
 elif off==-1:z[:,v,...,:-1]=x[:,v,...,1:]
 else:raise ValueError(off)
 return z

def js(a,b):
 p=torch.softmax(a.double(),-1);q=torch.softmax(b.double(),-1);m=(p+q)/2
 return (.5*((p*(p.clamp_min(1e-300).log()-m.clamp_min(1e-300).log())).sum(-1)+(q*(q.clamp_min(1e-300).log()-m.clamp_min(1e-300).log())).sum(-1))).numpy()

def load_model(runs,h):
 d=runs/f'held{h}';r=json.loads((d/'result.json').read_text())
 assert digest(d/'selected.pt')==r['files']['selected.pt']
 ck=torch.load(d/'selected.pt',weights_only=False,map_location='cpu');m=SignalNet().eval();m.load_state_dict(ck['states']['erm'])
 return m,ck['states']['erm'],r['files']['selected.pt']

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cost','run'])
 for n in ['data','prepared','runs','out']:ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--cost',type=Path);a=ap.parse_args();c=json.loads(P.read_text())
 if a.out.exists():raise ValueError('Use a new empty destination')
 torch.set_num_threads(c['threads']);torch.use_deterministic_algorithms(True)
 assert digest(a.prepared/'signals.npy')=='1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f'
 y=np.load(a.data/'cache/y_clutterremoved.npy');u=np.load(a.data/'cache/subj_clutterremoved.npy');good=quality_mask(y,u)&np.isin(u,c['source_users'])
 idsall=np.flatnonzero(good);assert len(idsall)==c['records']
 cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
 if a.mode=='cost':
  m,_,_=load_model(a.runs,3);x=get_batch(cache,idsall[u[idsall]==3][:64],'legacy');m(x[:2]);tick=time.perf_counter();m(x)
  for v in range(3):
   for off in c['offsets']:m(shift(x,v,off))
  secs=time.perf_counter()-tick;estimate=secs/len(x)*len(idsall)
  dump(a.out,{'script_sha256':digest(__file__),'protocol_sha256':digest(P),'probe_records':len(x),'seven_forward_seconds':secs,'estimated_seconds':estimate,'budget_seconds':c['cost_budget_seconds'],'passed':estimate<=c['cost_budget_seconds'],'labels_scored':False});print(a.out.read_text());return
 cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['script_sha256']==digest(__file__) and cost['protocol_sha256']==digest(P)
 a.out.mkdir();dump(a.out/'protocol.json',c);dump(a.out/'COST.json',cost);dump(a.out/'SOURCE_IDS.json',idsall.tolist())
 started=time.perf_counter();rows=[];audit=[];hashes={}
 for h in c['source_users']:
  ids=np.flatnonzero(good&(u==h));sp=json.loads((a.runs/f'held{h}'/'split.json').read_text());assert sp['diagnostic']==ids.tolist()
  assert not set(ids).intersection(sp['train']) and not set(ids).intersection(sp['val'])
  m,state,cksha=load_model(a.runs,h);bases=[];shifts=[];losses=[];jsvals=[];errs=[]
  for st in range(0,len(ids),c['batch']):
   x=get_batch(cache,ids[st:st+c['batch']],'legacy');base=m(x);batchlogs=[];batchloss=[];batchjs=[]
   for v in range(3):
    vlog=[];vloss=[];vjs=[]
    for off in c['offsets']:
     z=shift(x,v,off);out=m(z);vlog.append(out.numpy());vjs.append(js(base,out))
     den=x[:,v].double().square().sum((1,2,3));edge=x[:,v,...,0 if off==-1 else -1].double().square().sum((1,2));vloss.append(torch.where(den>0,edge/den,torch.ones_like(den)).numpy())
     if st==0:
      # Independent NumPy pad/slice shift and independent functional network.
      orig=x.numpy();ref=orig.copy();vv=orig[:,v]
      ref[:,v]=np.pad(vv,[(0,0),(0,0),(0,0),(1,1)])[...,0:63] if off==1 else np.pad(vv,[(0,0),(0,0),(0,0),(1,1)])[...,2:65]
      np.testing.assert_array_equal(ref,z.numpy());other=independent_forward(torch.from_numpy(ref),state).numpy();np.testing.assert_allclose(other,out.numpy(),rtol=1e-5,atol=1e-5);errs.append(float(np.max(np.abs(other-out.numpy()))))
      lost=(x[:,v].double().square().sum((1,2,3))-z[:,v].double().square().sum((1,2,3))).numpy();np.testing.assert_allclose(lost,edge.numpy(),atol=1e-9,rtol=1e-9)
    batchlogs.append(np.stack(vlog,1));batchloss.append(np.stack(vloss,1));batchjs.append(np.stack(vjs,1))
   bases.append(base.numpy());shifts.append(np.stack(batchlogs,1));losses.append(np.stack(batchloss,1));jsvals.append(np.stack(batchjs,1))
  base=np.concatenate(bases);z=np.concatenate(shifts);loss=np.concatenate(losses);j=np.concatenate(jsvals);flip=z.argmax(-1)!=base.argmax(-1)[:,None,None]
  saved=np.load(a.runs/f'held{h}'/'predictions.npz');np.testing.assert_array_equal(base,saved['erm']);np.testing.assert_array_equal(ids,saved['ids'])
  path=a.out/f'held{h}.npz';np.savez_compressed(path,ids=ids,base_logits=base,shift_logits=z,discarded_energy=loss,js_nats=j)
  hashes[path.name]=digest(path)
  for v,name in enumerate(c['views']):
   keep=loss[:,v]<=c['boundary_subset']['max_discarded_fraction'];n=int(keep.sum())
   rows.append({'held':h,'view':name,'pairs':int(keep.size),'flip_rate':float(flip[:,v].mean()),'js_mean':float(j[:,v].mean()),'discarded_energy_median':float(np.median(loss[:,v])),'low_boundary_pairs':n,'low_boundary_flip_rate':float(flip[:,v][keep].mean()) if n else None,'low_boundary_js_mean':float(j[:,v][keep].mean()) if n else None})
  audit.append({'held':h,'checkpoint_sha256':cksha,'unshifted_logits_equal_cached':True,'first64_six_perturbations_independent_forward_max_error':max(errs),'split_disjoint':True})
  print(json.dumps({'held':h,'state':'complete','labels_scored':False}),flush=True)
 view_summary={}
 for name in c['views']:
  rr=[r for r in rows if r['view']==name];coverage=all(r['low_boundary_pairs']>=c['boundary_subset']['min_pairs_each_user'] for r in rr)
  high=sum(r['low_boundary_flip_rate'] is not None and r['low_boundary_flip_rate']>=c['screen']['min_filtered_flip_rate'] for r in rr)
  view_summary[name]={'all_pairs_equal_user_flip_mean':float(np.mean([r['flip_rate'] for r in rr])),'low_boundary_equal_user_flip_mean':float(np.mean([r['low_boundary_flip_rate'] for r in rr])) if all(r['low_boundary_pairs'] for r in rr) else None,'low_boundary_pairs_by_user':[r['low_boundary_pairs'] for r in rr],'users_above_flip_threshold':high,'coverage_passed':coverage,'screen_passed':coverage and high>=c['screen']['min_users']}
 dump(a.out/'AUDIT.json',{'state':'passed','models':audit,'scope':'Independent shifted forward for first64 of each held user, six perturbations; complete unshifted replay; not all shifted logits independently replayed'})
 dump(a.out/'SUMMARY.json',{'state':'complete_frozen_source_sensitivity','records':len(idsall),'views':view_summary,'rows':rows,'prediction_files_sha256':hashes,'seconds':time.perf_counter()-started,'script_sha256':digest(__file__),'protocol_sha256':digest(P),'labels_scored':False,'accuracy_or_aliasing_causality_proven':False,'trained_models':0})
 print(json.dumps(view_summary),flush=True)
if __name__=='__main__':main()
