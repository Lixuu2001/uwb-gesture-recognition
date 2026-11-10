"""Freeze complete full8 predictions/IDs/private proof; never open gesture truth."""
from common import *
def main():
 all_private();resource_check();cfg=read(ROOT/'PROTOCOL.json');raw=Path(cfg['training_data_root'])/'cache'
 subjects_path=raw/'subj_clutterremoved.npy';ids_path=Path(cfg['source_ids'])
 assert sha(subjects_path)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
 assert sha(ids_path)=='d932f8910326ebf9c114c36e90026e5bf1cf8fdc71233816c9b389dabe3bdc50'
 subjects=np.load(subjects_path,mmap_mode='r',allow_pickle=False)
 with np.load(ids_path,allow_pickle=False) as z:dev=z['ids'].copy();users=z['users'].copy()
 np.testing.assert_array_equal(subjects[dev],users)
 ids=np.sort(np.concatenate([dev,np.flatnonzero(np.isin(subjects,[7,8]))])).astype(np.int64);assert len(ids)==len(set(ids.tolist()))==9300
 from outer_subject_roles import plan_roles
 files={};folds=[]
 for h in range(1,9):
  f=ROOT/f'folds/held{h}';frozen=read(f/'PREDICTIONS_FROZEN.json');files.update(frozen['files'])
  for path in [f/'PREDICTIONS_FROZEN.json',f/'PRIVATE_FOLD_RELEASE.json',f/'private_backup/MANIFEST.json',f/'private_backup/READBACK_VERIFIED.json',f/'private_backup/READBACK_INPUT.json']:files[str(path)]=sha(path)
  role=plan_roles(subjects[ids].tolist(),ids.tolist(),h,8);test=np.asarray(role['sample_ids']['test'],dtype=np.int64)
  arms={}
  for arm in ARMS:
   pred=f/'train'/f'{arm}_test_unscored.npz';model=f/'train'/f'selected_{arm}.pt'
   with np.load(pred,allow_pickle=False) as z:
    np.testing.assert_array_equal(z['ids'],test);assert z['scores'].shape==(len(test),12) and np.isfinite(z['scores']).all()
   arms[arm]=dict(prediction=str(pred),model=str(model),prediction_sha256=sha(pred),model_sha256=sha(model))
  folds.append(dict(held=h,validation_user=h%8+1,train_users=role['users']['train'],test_count=len(test),test_IDs_sha256=hashlib.sha256(test.tobytes()).hexdigest(),train_IDs_sha256=hashlib.sha256(np.asarray(role['sample_ids']['train'],dtype=np.int64).tobytes()).hexdigest(),validation_IDs_sha256=hashlib.sha256(np.asarray(role['sample_ids']['validation'],dtype=np.int64).tobytes()).hexdigest(),arms=arms))
 private_ids=HERE/'COHORT_IDS.npz';assert not private_ids.exists();np.savez_compressed(private_ids,ids=ids,users=subjects[ids]);files[str(private_ids)]=sha(private_ids)
 truth=raw/'y_clutterremoved.npy';assert sha(truth)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
 write(HERE/'ALL_EIGHT_FROZEN.json',dict(held_users=list(range(1,9)),primary='tail',baseline='native',controls=['native_sourceBN'],files=files,folds=folds,cohort_ids=str(private_ids),truth_path=str(truth),truth_sha256=sha(truth),gesture_truth_values_read=False,held_scores_computed=0,gate=cfg['gate'],historical_all8_exploratory=True,method_selected_before_new7_8_method_choice=True))
 print('All8 freeze/role/private/model/prediction manifest written without gesture truth values')
if __name__=='__main__':main()
