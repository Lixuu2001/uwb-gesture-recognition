from pathlib import Path
import json,pickle
from native_drcif_batched import np,make,original_make,fit,predict,assert_native,digest,BatchedNativePeriodogram
from aeon.transformations.collection import PeriodogramTransformer
B=Path(__file__).resolve().parent;rng=np.random.default_rng(20261029)
for shape in [(131,3,90),(67,567,90)]:
 x=rng.normal(size=shape);x[0,0]=1
 expected=PeriodogramTransformer().fit_transform(x);actual=BatchedNativePeriodogram().fit_transform(x);np.testing.assert_array_equal(actual,expected)
x=rng.normal(0,.15,(24,3,90)).astype(np.float32);y=np.arange(24)%12
for i in range(24):x[i]+=np.sin(np.arange(90)[None,:]*(y[i]+1)/11)+y[i]/4
models=[]
for factory in [original_make,make]:
 m=factory(4);fit(m,x,y);models.append(m)
a,b=models;assert_native(b,True);assert digest(a)==digest(b);pa=predict(a,x,24);pb=predict(b,x,7);np.testing.assert_array_equal(pa['probabilities'],pb['probabilities']);np.testing.assert_array_equal(pa['predictions'],pb['predictions'])
q=pickle.loads(pickle.dumps(b,protocol=5));assert_native(q,True);assert digest(q)==digest(b);pq=predict(q,x,5);np.testing.assert_array_equal(pq['probabilities'],pb['probabilities']);np.testing.assert_array_equal(pq['predictions'],pb['predictions'])
r=dict(state='passed_full500_CIT20_batched_native_periodogram_execution_equivalence',native_spectrum_full_vs64batch_exact_shapes=[[131,3,90],[67,567,90]],constant_channel_tested=True,all500_trees_intervals_feature_subsets_node_states_leaf_probabilities_exact=True,all_native_and_batched_probabilities_predictions_exact=True,pickle_all_scientific_state_probabilities_predictions_exact=True,seed=20261029,trees=500,CIT_thresholds=20,native_jobs=4,parallel_backend='threading',spectrum_batch_size=64,original_native_spectrum_source_unmodified=True,model_precision='float64_both',input_views_channels_unchanged=True,full4700_classifier_not_fitted=True,actual_UWB_values_labels_fit=False,novelty_claim=False)
(B/'EQUIVALENCE_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
