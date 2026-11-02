import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from huber_readout import fit,weighted_fit
from second_order_core import weights,macro
from radar_exchange import decision


def main(a):
 c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',c);identity=dict(protocol_sha256=digest(a.protocol),code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['run_huber_readout.py','huber_readout.py','second_order_core.py','radar_exchange.py','audit_restored.py']});dump(a.out/'IDENTITY.json',identity)
 if a.cost_only:
  rng=np.random.default_rng(17);x=np.c_[rng.normal(size=(4800,128)),np.ones(4800)];y=np.tile(np.arange(12).repeat(100),4);u=np.repeat(np.arange(3,7),1200);w=weights(y,u);target=np.eye(12)[y];initial=np.stack([weighted_fit(x,target[:,k],w,.01) for k in range(12)],1);t=time.perf_counter();fit(x,y,u,w,initial,.01,8,c);elapsed=time.perf_counter()-t;r=dict(synthetic_seconds=elapsed,estimate_seconds=elapsed*6*3,passed=elapsed*6*3<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],real_data_read=False);dump(a.out/'COST.json',r);print(json.dumps(r));return
 cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256'];old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];assert digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text());raw=Path('/workspace/uwb-restored-20260930/cache');yp=raw/'y_clutterremoved.npy';up=raw/'subj_clutterremoved.npy';assert digest(yp)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(up)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480';y=np.load(yp);u=np.load(up);scores={k:[] for k in c['arms']};rows=[];t=time.perf_counter()
 for h in c['users']:
  d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
  for n in ['models.npz','predictions.npz','selection.json','split.json']:assert digest(od/n)==rm[f'held{h}/{n}']
  assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];f=np.load(fd/'features.npz');m=np.load(od/'models.npz');sel=json.loads((od/'selection.json').read_text())['onehot'];split=json.loads((od/'split.json').read_text());tr=np.array(split['train']);va=np.array(split['val']);te=np.array(split['diagnostic']);assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te));lookup={int(v):i for i,v in enumerate(f['ids'])}
  def hidden(ids):
   x=f['hidden'][[lookup[int(v)] for v in ids]];return np.c_[(x-m['mean'])/m['scale'],np.ones(len(ids))]
  lam=sel['candidates'][sel['selected']]['lam'];initial=np.vstack([m['coefs'][sel['selected']],m['bias']]);models,extra,logs=fit(hidden(tr),y[tr],u[tr],weights(y[tr],u[tr]),initial,lam,h,c);models['ridge']=initial;pred={}
  for role,ids in [('val',va),('diagnostic',te)]:
   pred[role+'_ids']=ids
   for k,model in models.items():pred[role+'_'+k]=hidden(ids)@model
  np.savez_compressed(d/'models.npz',**models,mean=m['mean'],scale=m['scale']);np.savez_compressed(d/'training.npz',ids=tr,**extra);np.savez_compressed(d/'predictions.npz',**pred);dump(d/'split.json',split);dump(d/'selection.json',dict(new_selection=False,inherited_onehot=sel,lambda_used=lam));dump(d/'OPTIMIZER.json',logs)
  with np.load(od/'predictions.npz') as truth:
   np.testing.assert_array_equal(te,truth['ids']);np.testing.assert_array_equal(va,truth['validation_ids']);np.testing.assert_allclose(pred['diagnostic_ridge'],truth['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(pred['diagnostic_ridge'].argmax(1),truth['onehot'].argmax(1));ss={k:macro(truth['y'],pred['diagnostic_'+k].argmax(1)) for k in c['arms']};np.savez_compressed(d/'evaluation.npz',diagnostic_y=truth['y'],val_y=truth['validation_y'])
  for k,v in ss.items():scores[k].append(v)
  row=dict(held=h,scores=ss,clipped_fraction=float((extra['influence']<1).mean()),max_gradient=max(v['gradient_inf'] for v in logs));rows.append(row);dump(d/'result.json',row);print(json.dumps(row),flush=True)
 comp=decision(scores,c);summary=dict(state='complete',means={k:float(np.mean(v)) for k,v in scores.items()},scores=scores,comparisons=comp,gate_passed=all(v['passed'] for v in comp.values()),rows=rows,seconds=time.perf_counter()-t,nonlinear_output_fits=72,control_linear_output_fits=144,encoder_training=False);dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
 with threadpool_limits(limits=1):main(a)
