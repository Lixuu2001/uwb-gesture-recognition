"""Synthetic semantic parity for the SVM job-count adapter and compatibility."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import json,time
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from author_adapter import BASE,source_check
from readout_adapter import definition,fit_svm
def main():
 assert not (BASE/'READOUT_SYNTHETIC_CHECK.json').exists();started=time.monotonic();source_check()
 x=np.random.default_rng(809).normal(size=(60,8));y=np.arange(60)%3;query=np.random.default_rng(810).normal(size=(12,8));calls=[]
 def reference_grid(*args,**kwargs):
  assert kwargs['n_jobs']==5;calls.append(dict(n_jobs=5,cv=kwargs['cv'],C_values=[str(v) for v in args[1]['C']]))
  kwargs['n_jobs']=1;return GridSearchCV(*args,**kwargs)
 expected=definition(False,reference_grid)(x,y);actual=fit_svm(x,y)
 assert len(calls)==1 and len(calls[0]['C_values'])==10 and calls[0]['cv']==5
 assert expected.get_params()==actual.get_params();np.testing.assert_array_equal(expected.support_,actual.support_);np.testing.assert_array_equal(expected.dual_coef_,actual.dual_coef_);np.testing.assert_array_equal(expected.decision_function(query),actual.decision_function(query));np.testing.assert_array_equal(expected.predict(query),actual.predict(query))
 try: LogisticRegression(multi_class='ovr');legacy_lr='accepted'
 except TypeError as e: legacy_lr=repr(e)
 result=dict(state='pass',synthetic_only=True,real_UWB_values_or_labels_read=False,real_training_started=False,sklearn_version=sklearn.__version__,original_SVM_grid_with_inf_supported=True,original_C_grid_values=calls[0]['C_values'],original_training_CV_folds=5,parallel_only_change='n_jobs5to1',original_source_parity_predictions_dual_support_exact=True,source_optional_linear_helper_compatibility=legacy_lr,selected_real_protocol=False,wall_seconds=time.monotonic()-started,limitation='Synthetic SVM compatibility/parallelism check, no real accuracy result. Original CLI explicitly uses SVM; optional old linear helper is not reproduced under sklearn1.8.')
 (BASE/'READOUT_SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
