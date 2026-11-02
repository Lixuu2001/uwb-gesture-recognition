"""Synthetic integration parity;not resource measurement or UWB selection."""
from pathlib import Path
import json,hashlib
from native_quant import make,features,fit,predict,scientific_arrays,np,torch,author,cloudpickle,ExtraTreesClassifier,B
def main():
    rng=np.random.default_rng(20261029);x=rng.standard_normal((24,3,90),dtype=np.float32);x[:,1]=0;y=np.arange(24)%12
    m=make();f,_=features(m,x,y,True,7)
    original=author.Quant();native=original.fit_transform(torch.from_numpy(x),y).numpy()
    np.testing.assert_array_equal(f,native);assert f.flags.f_contiguous and native.flags.c_contiguous
    m['classifier'].fit(f,y)
    c=ExtraTreesClassifier(n_estimators=200,max_features=.1,criterion='entropy',n_jobs=4,random_state=20261029);c.fit(np.ascontiguousarray(f),y)
    ref=dict(transform=original,classifier=c)
    for a,b in zip(scientific_arrays(m),scientific_arrays(ref)):np.testing.assert_array_equal(a,b)
    v=rng.standard_normal((5,3,90),dtype=np.float32);p=predict(m,v,2)
    q=c.predict_proba(original.transform(torch.from_numpy(v)).numpy());np.testing.assert_array_equal(p['probabilities'],q);np.testing.assert_array_equal(p['predictions'],c.classes_[q.argmax(1)])
    clone=cloudpickle.loads(cloudpickle.dumps(m,protocol=5));r=predict(clone,v,2)
    for key in ['probabilities','predictions']:np.testing.assert_array_equal(r[key],p[key])
    for a,b in zip(scientific_arrays(m),scientific_arrays(clone)):np.testing.assert_array_equal(a,b)
    report=dict(state='passed_complete_author_QUANT200_tree_integration_on_synthetic_fixture',author_commit='f3907f7e531b6a304e3c54310869a0023a9e0fd6',depth=6,div=4,trees=200,max_features=.1,criterion='entropy',seed=20261029,native_tree_jobs=4,Torch_threads=1,training_fixture_shape=list(x.shape),training_fixture_features=f.shape[1],full567_feature_transform_width_separately_verified=539217,native_full_vs_batched_features_exact=True,Fortran_vs_C_all200_tree_arrays_exact=True,native_predictions_probabilities_exact=True,cloudpickle_alltransform_tree_arrays_and_predictions_exact=True,original_author_code_unmodified=True,full4700_classifier_not_fitted=True,formal_cost_not_started=True,actual_UWB_values_labels_fit=False,method_not_selected=True,accuracy_or_novelty_claim=False)
    (B/'INTEGRATION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
