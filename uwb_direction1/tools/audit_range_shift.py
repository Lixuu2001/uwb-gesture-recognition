"""Independently recompute saved sensitivity statistics and full boundary ratios."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import softmax,xlogy

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);a=p.parse_args()
 s=json.loads((a.out/'SUMMARY.json').read_text());cache=np.load(a.prepared/'signals.npy',mmap_mode='r');err=0.;seen=[]
 for h in range(3,9):
  path=a.out/f'held{h}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==s['prediction_files_sha256'][path.name]
  q=np.load(path);ids=q['ids'];seen.extend(ids.tolist());pp=softmax(q['base_logits'].astype(np.float64),axis=-1)[:,None,None,:];qq=softmax(q['shift_logits'].astype(np.float64),axis=-1);mm=(pp+qq)/2
  ref=.5*(np.sum(xlogy(pp,pp)-xlogy(pp,mm),axis=-1)+np.sum(xlogy(qq,qq)-xlogy(qq,mm),axis=-1))
  np.testing.assert_allclose(ref,q['js_nats'],atol=1e-12,rtol=1e-9);err=max(err,float(np.max(np.abs(ref-q['js_nats']))))
  loss=[]
  for start in range(0,len(ids),32):
   x=np.asarray(cache[ids[start:start+32]][:,:,[0,2]],dtype=np.float64)
   den=np.square(x).sum((2,3,4));edge=np.square(x[...,[0,-1]]).sum((2,3))
   loss.append(np.divide(edge,den[...,None],out=np.ones_like(edge),where=den[...,None]>0))
  np.testing.assert_allclose(np.concatenate(loss),q['discarded_energy'],atol=1e-12,rtol=1e-12)
  flip=q['shift_logits'].argmax(-1)!=q['base_logits'].argmax(-1)[:,None,None]
  for v,name in enumerate(['Left','Top','Right']):
   row=next(r for r in s['rows'] if r['held']==h and r['view']==name);keep=q['discarded_energy'][:,v]<=.01
   assert row['low_boundary_pairs']==int(keep.sum());assert abs(row['flip_rate']-float(flip[:,v].mean()))<1e-15
   if keep.any():assert abs(row['low_boundary_flip_rate']-float(flip[:,v][keep].mean()))<1e-15
 assert len(seen)==len(set(seen))==7100 and sorted(seen)==json.loads((a.out/'SOURCE_IDS.json').read_text())
 report={'state':'passed','saved_prediction_hashes':6,'source_ids_unique':7100,'boundary_energy_ratios_independently_recomputed':42600,'js_ratios_independently_recomputed':42600,'max_js_error':err,'flip_statistics_recomputed':True,'scope':'All boundary/JS/flip statistics; model logits independently replayed only as specified in AUDIT.json'}
 (a.out/'POST_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
