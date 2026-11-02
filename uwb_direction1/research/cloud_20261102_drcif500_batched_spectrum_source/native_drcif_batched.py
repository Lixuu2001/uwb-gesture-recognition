"""Execution adapter: batch the unchanged native periodogram,keep all outputs.

All500 CIT20/intervals/seed/float64/views unchanged. Original aeon and prior
failed cost artifacts remain untouched. No novelty or Java parity claim.
"""
import sys
sys.path.insert(0,'/workspace/uwb-drcif500-source-probe-20261102')
from native_drcif import np,fit,predict,digest,learned_state
from native_drcif import make as original_make,assert_native as original_assert
from aeon.transformations.collection import PeriodogramTransformer
from aeon.classification.interval_based import DrCIFClassifier
class BatchedNativePeriodogram(PeriodogramTransformer):
 def _transform(self,X,y=None):
  assert isinstance(X,np.ndarray) and X.ndim==3 and X.dtype==np.float64 and self.pad_series and self.pad_with=='constant' and self.constant_value==0
  width=2**int(np.ceil(np.log2(X.shape[2])))//2
  output=np.empty((len(X),X.shape[1],width),dtype=np.float64)
  for start in range(0,len(X),64):output[start:start+64]=super()._transform(X[start:start+64],None)
  return output
class BatchedSpectrumDrCIF(DrCIFClassifier):
 def _fit(self,X,y):
  # Publicfit resets constructorstate first. Install here after that reset and
  # before invoking the unchanged native forest implementation.
  self.series_transformers[2]=BatchedNativePeriodogram()
  return super()._fit(X,y)
def make(jobs=4):
 return BatchedSpectrumDrCIF(**original_make(jobs).get_params(deep=False))
def assert_native(m,fitted=False,jobs=4):
 original_assert(m,fitted,jobs);assert isinstance(m,BatchedSpectrumDrCIF)
 if fitted:assert isinstance(m.series_transformers[2],BatchedNativePeriodogram) and isinstance(m._series_transformers[2],BatchedNativePeriodogram)
