"""Full500-tree integration on a small synthetic multivariate fixture only."""
from pathlib import Path
import pickle,json,copy
from native_drcif import np,B,make,assert_native,digest,predict
rng=np.random.default_rng(20261029);x=rng.normal(0,.15,(24,3,90)).astype(np.float32);y=np.arange(24)%12
for i in range(24):x[i]+=np.sin(np.arange(90)[None,:]*(y[i]+1)/11)+y[i]/4
models=[]
for jobs in [1,4]:
 m=make(jobs);m.fit(x,y);assert_native(m,True,jobs);models.append(m)
a,b=models;da=digest(a);db=digest(b);assert da==db
pa=predict(a,x,24);pb=predict(b,x,7);np.testing.assert_array_equal(pa['probabilities'],pb['probabilities']);np.testing.assert_array_equal(pa['predictions'],pb['predictions']);np.testing.assert_array_equal(a.predict(x),pa['predictions'])
q=pickle.loads(pickle.dumps(b,protocol=5));assert digest(q)==digest(b);pq=predict(q,x,5);np.testing.assert_array_equal(pq['probabilities'],pb['probabilities']);np.testing.assert_array_equal(pq['predictions'],pb['predictions'])
# Independent application of representation/interval/CIT probability composition.
# Includes every selected feature (native efficient-prediction skip masks disabled).
q=copy.deepcopy(b);reps=q._predict_setup(x);member=[];selected_features=0;raw_nonfinite=0
for tree,group in zip(q.estimators_,q.intervals_):
 pieces=[]
 for r,s in enumerate(group):
  s._transform_features=None
  for a0,b0,c0,f0,d0 in s.intervals_:
   if hasattr(f0,'_transform_features'):f0._transform_features=None
   if hasattr(f0,'transform_features_'):f0.transform_features_=None
  part=s.transform(reps[r]);raw_nonfinite+=int((~np.isfinite(part)).sum());pieces.append(part);selected_features+=part.shape[1]
 features=np.hstack(pieces);features=np.nan_to_num(features,False,np.nan,np.nan,np.nan);member.append(tree.predict_proba(features))
manual=np.sum(member,axis=0)/500;np.testing.assert_array_equal(manual,pb['probabilities'])
result=dict(state='passed_full500_CIT_native_DrCIF_integration_on_synthetic_fixture',fixture_shape=list(x.shape),classes=12,full_trees=500,CIT_thresholds=20,att_subsample_size=10,intervals_per_representation=b._n_intervals,serial_jobs1_parallel_threading4_all_intervals_feature_subsets_nodes_leaf_probabilities_exact=True,native_vs_batched_probabilities_predictions_exact=True,pickle_all_learned_state_and_predictions_exact=True,all_selected_features_unskipped_probability_composition_exact=True,selected_feature_columns_over500trees=selected_features,raw_nonfinite_values_native_NaN_policy_retained=raw_nonfinite,full567_classifier_not_fitted=True,formal4700_cost_not_started=True,actual_UWB_values_labels_fit=False,novelty_claim=False)
(B/'INTEGRATION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
