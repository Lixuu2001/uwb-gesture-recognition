import json,math
from pathlib import Path
from native_drcif import np,B
from aeon.classification.sklearn._continuous_interval_tree import _TreeNode
from aeon.transformations.collection import PeriodogramTransformer
from aeon.utils.numba.general import first_order_differences_3d
from aeon.utils.numba.stats import row_mean,row_std,row_slope,row_median,row_iqr,row_numba_min,row_numba_max
rng=np.random.default_rng(20261029);x=rng.normal(size=(24,3,90)).astype(np.float32);x[0,0]=1
p=PeriodogramTransformer().fit_transform(x)
ref=np.abs(np.fft.fft(np.pad(x.astype(np.float64),((0,0),(0,0),(0,38))),axis=2)[:,:,:64]);np.testing.assert_allclose(p,ref,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(first_order_differences_3d(x),np.diff(x.astype(np.float64),axis=2),rtol=1e-6,atol=1e-6)
z=x[:,0,7:37].astype(np.float64);tt=np.arange(z.shape[1]);slope=((z-z.mean(1,keepdims=True))*(tt-tt.mean())).sum(1)/((tt-tt.mean())**2).sum()
for fun,reference in [(row_mean,z.mean(1)),(row_std,z.std(1)),(row_slope,slope),(row_median,np.median(z,axis=1)),(row_iqr,np.quantile(z,.75,axis=1)-np.quantile(z,.25,axis=1)),(row_numba_min,z.min(1)),(row_numba_max,z.max(1))]:np.testing.assert_allclose(fun(z),reference,rtol=1e-12,atol=1e-12)
xx=np.array([[0.],[1.],[np.nan]]);yy=np.array([0,1,2]);expected=[np.array([1.,0,0]),np.array([0.,1,0]),np.array([0.,0,1])]
gain,dist,ent=_TreeNode.information_gain(xx,yy,0,.5,math.log2(3),3)
for got,want in zip(dist,expected):np.testing.assert_array_equal(got,want)
np.testing.assert_allclose(gain,math.log2(3),rtol=1e-12,atol=1e-12)
got=_TreeNode.split_data(xx,0,.5)
for a,b in zip(got,[[0],[1],[2]]):np.testing.assert_array_equal(a,b)
result=dict(state='passed_independent_native_DrCIF_representations_seven_stats_CIT_missing_split_entropy',periodogram_shape=list(p.shape),FP64_diff_FFT_and_seven_statistics=True,missing_values_distinct_three_way_split_correct=True,actual_UWB_values_labels_fit=False,full500_fit_not_started=True)
(B/'BOUNDARY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
