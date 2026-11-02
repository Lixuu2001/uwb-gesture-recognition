"""IEEE-754 forward bound for the unchanged float32 resize,not a fit parameter.

u=2^-24,gamma_n=n*u/(1-n*u). Minmax values lie in[0,1]. The half-pixel
coordinate uses division,multiplication,subtraction;absolute coordinate error
is bounded by gamma3*(input_extent+.5),plus two u for weight storage. Bilinear
interpolation is 1-Lipschitz per coordinate for values in[0,1]. gamma5 covers
input minmax/floor, gamma12 interpolation arithmetic, gamma6*3 ImageNet
constant/subtraction/division effects. Divide by the smallest ImageNet std.
This conservative bound is derived from dtype/dimensions,never measured labels.
"""
import sys
sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-pilot-20261102')
from engine import np,pixels_reference,preprocess
def pixel_error_bound():
 u=2.0**-24;g=lambda n:n*u/(1-n*u)
 coordinate=g(3)*(90+.5)+g(3)*(189+.5)+4*u
 bound=(g(5)+g(12)+coordinate+3*g(6))/.224
 assert 0<bound<1e-3
 return bound
def verify_pixels(x):
 actual=preprocess(x).numpy();ideal=pixels_reference(x);error=np.abs(actual.astype(np.float64)-ideal);assert np.isfinite(actual).all() and np.isfinite(ideal).all()
 assert float(error.max())<=pixel_error_bound()
 return float(error.max()),int(error.size)
