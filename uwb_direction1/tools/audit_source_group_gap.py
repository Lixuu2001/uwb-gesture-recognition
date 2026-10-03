"""Independent per-group scoring and role reconstruction from saved logits."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_restored import digest,dump

def main(root):
 s=json.loads((root/'SUMMARY.json').read_text());c=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'GROUPS.json').read_text());models=json.loads((root/'MODELS.json').read_text())
 for n,h in s['files'].items():assert digest(root/n)==h
 for m in models:assert digest(m['path'])==m['sha256']
 y=np.load(Path(c['data'])/'cache/y_clutterremoved.npy');u=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy');count=0;recomputed=[];total=0;maxerr=0
 for row in rows:
  h=row['held'];v=h+1 if h<8 else 3;z=np.load(root/f'held{h}'/f"epoch{row['epoch']:02d}.npz");good=(u>=3)&(u<=8)&~((u==6)&(y==3));new=dict(held=h,epoch=row['epoch'],roles={})
  for role,mask in [('train',good&(u!=h)&(u!=v)),('val',good&(u==v))]:
   ids=np.flatnonzero(mask);np.testing.assert_array_equal(ids,z[role+'_ids']);logit=z[role+'_logits'].astype('float64');assert np.isfinite(logit).all();total+=len(ids)
   shift=logit-logit.max(1,keepdims=True);loss=np.log(np.exp(shift).sum(1))-shift[np.arange(len(ids)),y[ids]];pred=logit.argmax(1);groups={}
   for user in np.unique(u[ids]):
    cc={}
    for label in np.unique(y[ids][u[ids]==user]):
     m=(u[ids]==user)&(y[ids]==label);n=int(m.sum());err=float(np.count_nonzero(pred[m]!=label)/n);ce=float(sum(loss[m])/n);cc[str(label)]=dict(n=n,ce=ce,error=err)
     ref=row['roles'][role][str(user)]['classes'][str(label)];assert ref['n']==n and abs(ref['error']-err)<1e-14;maxerr=max(maxerr,abs(ref['ce']-ce));assert abs(ref['ce']-ce)<1e-12;count+=1
    group=dict(macro_ce=sum(x['ce'] for x in cc.values())/len(cc),macro_error=sum(x['error'] for x in cc.values())/len(cc),classes=cc);groups[str(user)]=group
    ref=row['roles'][role][str(user)];assert abs(group['macro_error']-ref['macro_error'])<1e-14 and abs(group['macro_ce']-ref['macro_ce'])<1e-12
   new['roles'][role]=groups
  recomputed.append(new)
 qualifying=[]
 for h in c['source_users']:
  passes=[]
  for epoch in c['gate']['epochs']:
   row=next(x for x in recomputed if x['held']==h and x['epoch']==epoch);e=[x['macro_error'] for x in row['roles']['train'].values()];passes.append(max(e)>=.05 and max(e)-min(e)>=.03)
  if all(passes):qualifying.append(h)
 for stage in s['stages']:
  rr=[r for r in recomputed if r['epoch']==stage['epoch']]
  for field,role,metric,agg in [('train_worst_error','train','macro_error',max),('val_error','val','macro_error',max),('train_worst_ce','train','macro_ce',max),('val_ce','val','macro_ce',max)]:
   ref=[agg(v[metric] for v in r['roles'][role].values()) for r in rr];np.testing.assert_allclose(stage[field],ref,rtol=0,atol=1e-12)
 assert qualifying==s['qualifying_trajectories'];assert (len(qualifying)>=4)==s['persistent_training_group_gate'];assert total==s['source_role_rows'];raw=json.loads((root/'FORWARD_AUDIT.json').read_text());assert len(raw)==72 and all(x['max_error']<2e-5 for x in raw)
 dump(root/'AUDIT.json',dict(passed=True,artifact_sha256=len(s['files']),checkpoint_sha256=len(models),group_cells=count,rows=total,max_independent_CE_error=maxerr,independent_forward_batches=len(raw),max_forward_error=max(x['max_error'] for x in raw),held_signal_excluded=True,gate_recomputed=True))
 print(json.dumps(dict(passed=True,group_cells=count,rows=total,qualifying=qualifying)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
