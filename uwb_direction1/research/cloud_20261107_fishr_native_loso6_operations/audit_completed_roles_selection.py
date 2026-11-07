"""Independent read-only role/ID and logged validation-selection intent audit; no held truth."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
P=Path('/workspace/uwb-fishr-native-loso6-20261107')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(held):
 assert held and len(set(held))==len(held) and all(h in range(1,7) for h in held)
 tag='_'.join(map(str,held));out=P/f'COMPLETED_FOLDS_{tag}_ROLE_SELECTION_AUDIT.json';assert not out.exists()
 rows=[]
 for h in held:
  F=P/f'folds/held{h}';c=read(F/'PROTOCOL.json');assert read(F/'PHASE_TERMINATION.json')['exit_code']==0
  assert c['held']==h and c['validation_user']==h%6+1 and c['train_users']==[u for u in range(1,7) if u not in [h,h%6+1]]
  assert sha(c['ids'])==c['ids_sha256']
  with np.load(c['ids']) as z:ids=z['ids'];users=z['users']
  assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900 and set(np.unique(users))==set(range(1,7))
  loc=dict(train=np.flatnonzero(np.isin(users,c['train_users'])),validation=np.flatnonzero(users==c['validation_user']),diagnostic=np.flatnonzero(users==h));sets=[set(ids[ix].tolist()) for ix in loc.values()]
  assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)) and len(set.union(*sets))==6900
  assert [len(v) for v in loc.values()]==c['counts']
  for file in [F/'roles.npz',F/'run/train/split.npz',F/'run/replay/split.npz']:
   with np.load(file) as z:
    for role,ix in loc.items():np.testing.assert_array_equal(z[role],ids[ix])
  records=[read(F/f'run/train/epoch{epoch:03}.json') for epoch in range(1,65)];expected=max(records,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))['epoch']
  producer=read(F/'run/train/COMPLETE.json');replay=read(F/'run/replay/COMPLETE.json');assert producer['selected_epoch']==replay['selected_epoch']==expected
  rows.append(dict(held=h,validation_user=c['validation_user'],train_users=c['train_users'],roles={role:dict(n=len(ix),users=np.unique(users[ix]).tolist(),sample_ID_sha256=hashlib.sha256(ids[ix].tobytes()).hexdigest(),dtype=str(ids.dtype)) for role,ix in loc.items()},source_ids_sha256=c['ids_sha256'],private_roles_and_both_splits_exact=True,all_roles_subject_and_ID_disjoint=True,selected_epoch=expected,selection='max logged validation present-macro, then lower logged native global BCE, then earlier epoch',logged_rule_independently_recomputed=True,initial_asset_sha256=sha(F/'initial.pt'),all64_epochs_retained=True,test_labels_or_predictions_opened=False))
 result=dict(state='completed_fold_subject_ID_firewall_and_validation_selection_intent_audited_without_fitting_or_test_pairing',folds=rows,scope=held,scientific_source_sha256=sha(P/'FROZEN_HASHES.json'),source_script_sha256=sha(__file__),optimizer_updates=0,test_labels_or_predictions_opened=False,selection_rule_audit_does_not_independently_recompute_logged_validation_metrics=True)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(out),folds=held,all_roles_disjoint=True,selection_rule_validated=True,new_fits=0)))
if __name__=='__main__':main([int(v) for v in sys.argv[1:]])
