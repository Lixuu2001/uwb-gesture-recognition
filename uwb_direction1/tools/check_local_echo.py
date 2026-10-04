"""Analytic shift/sign and independent scalar-correlation numerical checks."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import pearsonr
from local_echo_correspondence import scores,match,envelope,select_ids


def main():
 c=json.loads((Path(__file__).resolve().parent.parent/'research/LOCAL_ECHO_PROTOCOL.json').read_text());rng=np.random.default_rng(19)
 a=rng.normal(size=(4,189));centers=np.array(c['centers']);count=0
 for d in [-5,-2,0,3,5]:
  b=np.roll(a,d,axis=-1)*2.5+7;r=match(a,b,centers,c)
  np.testing.assert_array_equal(r['shift'],d);np.testing.assert_array_equal(r['back'],-d);assert r['reliable'].all();count+=1
  v,_,_=scores(a,b,np.broadcast_to(centers,(len(a),len(centers))),8,6)
  for center in centers[::3]:
   for shift in [-6,-2,0,4,6]:
    scalar=pearsonr(a[0,center-8:center+9],b[0,center+shift-8:center+shift+9]).statistic
    np.testing.assert_allclose(v[0,list(centers).index(center),shift+6],scalar,atol=2e-15,rtol=0)
 zero=match(np.zeros_like(a),np.zeros_like(a),centers,c);assert not zero['eligible'].any() and not zero['reliable'].any()
 periodic=np.tile(np.array([1.,-1.]),95)[:189][None];assert not match(periodic,periodic,centers,c)['reliable'].any()
 # A Hilbert envelope of a known integer-period cosine has known constant amplitude.
 x=np.cos(2*np.pi*np.arange(189)*9/189)[None];np.testing.assert_allclose(envelope(x),1,atol=2e-14)
 y=np.tile(np.arange(12),8*10);u=np.repeat(np.arange(1,9),120);ids=select_ids(y,u,c);yy=y.copy();yy[u<3]=999
 np.testing.assert_array_equal(ids,select_ids(yy,u,c));assert (u[ids]>=3).all() and not ((u[ids]==6)&(y[ids]==3)).any()
 print(json.dumps({'state':'passed','known_shift_cases':count,'scalar_pearson_checks':125,'zero_texture_rejected':True,'ambiguous_periodic_rejected':True,'hilbert_known_envelope':True,'excluded_labels_isolated':True}))


if __name__=='__main__':main()
