"""Independently audit saved source pairing IDs, flip scores, gates and hashes.
Raw inference was separately checked inside the diagnostic on fixed64/condition.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args();src=a.runs
if (src/'POST_AUDIT.json').exists():raise ValueError('Audit already exists; preserve it')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
s=json.loads((src/'SUMMARY.json').read_text());c=json.loads((src/'protocol.json').read_text());assert all(sha(src/n)==v for n,v in s['files_sha256'].items())
rows=json.loads((src/'READINGS.json').read_text());checks=[];passed=[]
for v in range(3):
 wins=[]
 for h in range(3,9):
  d=np.load(src/f'held{h}.npz');ids=d['ids'];y=d['source_y'];u=d['source_users'];base=d['original_logits'];assert len(set(ids))==len(ids) and set(u)==set(range(3,9))-{h,h+1 if h<8 else 3}
  loc={int(n):i for i,n in enumerate(ids)};np.testing.assert_array_equal(d['identity_logits'],base);np.testing.assert_allclose(d['joint_logits'],base[[loc[int(n)] for n in d['joint_ids']]],atol=2e-5,rtol=2e-5)
  ok=[]
  for seed in c['permutation_seeds']:
   flips={}
   for kind in ['same','cross']:
    t=d[f'{kind}_{seed}_target_ids'];ix=np.array([loc[int(n)] for n in t]);assert set(t)==set(ids) and np.all(t!=ids);assert np.all(y[ix]==y);assert np.all((u[ix]==u) if kind=='same' else (u[ix]!=u))
    pred=d[f'{kind}_{seed}_v{v}_logits'].argmax(1);vals=[]
    for user in sorted(set(u)):
     vals.append(np.mean([np.mean(pred[(u==user)&(y==cls)]!=base.argmax(1)[(u==user)&(y==cls)]) for cls in sorted(set(y[u==user]))]))
    flips[kind]=float(np.mean(vals));r=next(x for x in rows if (x['held'],x['kind'],x['seed'],x['view'])==(h,kind,seed,v));assert abs(r['flip']-flips[kind])<1e-12
   ok.append(flips['cross']>=.05 and flips['cross']-flips['same']>=.02)
  if all(ok):wins.append(h)
 assert wins==s['screens'][v]['passing_trajectories'];passed.append(len(wins)>=4)
assert any(passed)==s['screen_passed'];audit={'state':'passed','all72_flip_readings_independently_recomputed':True,'all_source_id_maps_and_joint_identity_controls_verified':True,'all_declared_artifact_sha256_verified':True,'gate_independently_recomputed':True,'all_raw_inference_repeated':False,'raw_inference_scope':'production diagnostic independently recomputed64 random samples x14 conditions x6 trajectories; this post audit checks saved arrays/maps/metrics'}
(src/'POST_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
