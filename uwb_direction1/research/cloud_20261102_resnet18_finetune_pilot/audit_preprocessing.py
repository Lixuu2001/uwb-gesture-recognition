"""Full raw-RMS/base-pixel audit; inherits explicitly disclosed analytic bound."""
from pathlib import Path
import ast
from engine import np,preprocess,raw
_old=Path('/workspace/uwb-resnet18-imagenet-pilot-20261102/engine.py');_node=next(n for n in ast.parse(_old.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pixels_reference');exec(compile(ast.Module(body=[_node],type_ignores=[]),str(_old),'exec'),globals())
def pixel_bound():
 u=2**-24;g=lambda n:n*u/(1-n*u);return (g(5)+g(12)+g(3)*(90+.5)+g(3)*(189+.5)+4*u+3*g(6))/.224
def verify(c,maps,ids):
 maximum=0.;count=0
 for offset in range(0,6900,8):
  a=raw(maps,np.arange(offset,min(offset+8,6900)));native=preprocess(a).numpy();ideal=pixels_reference(a);error=np.abs(native.astype(np.float64)-ideal);assert np.isfinite(ideal).all() and float(error.max())<=pixel_bound();maximum=max(maximum,float(error.max()));count+=error.size
 original=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
 for offset in range(0,6900,64):
  a=np.array(original[ids[offset:offset+64]],copy=True);r=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32));ref=(a/np.maximum(r,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,maps[offset:offset+64],rtol=2e-5,atol=2e-6)
  d=a.astype(np.float64);d=(d/np.maximum(np.sqrt(np.mean(d*d,axis=(2,3),keepdims=True)),1e-8)).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,d,rtol=2e-5,atol=2e-6)
 return dict(all6900_raw_RMS_cache_FP64_verified=True,pixel_count=int(count),pixel_maximum_error=maximum,pixel_bound=pixel_bound())
