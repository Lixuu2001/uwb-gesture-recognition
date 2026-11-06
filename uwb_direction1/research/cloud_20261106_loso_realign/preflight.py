"""Read-only role/reference/reuse inventory. Never score MMD outputs or start training."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from loso_roles_metrics import roles,synthetic_checks
B=Path(__file__).resolve().parent
G=Path('/workspace/uwb-cloud-current/uwb_direction1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    p=read(B/'PROTOCOL.json');ids_path=Path(p['source_ids']);assert sha(ids_path)==p['source_ids_sha256']
    with np.load(ids_path) as z:ids=z['ids'];users=z['users']
    assert len(ids)==len(np.unique(ids))==6900
    private={};folds=[]
    for h in range(1,7):
        v,q=roles(users,h);counts={k:len(x) for k,x in q.items()};assert list(counts.values())==p['counts'][str(h)]
        for role,ix in q.items():private[f'held{h}_{role}']=ids[ix]
        folds.append(dict(held=h,validation_user=v,training_users=sorted(set(users[q['train']].tolist())),counts=counts,role_ID_hashes={k:hashlib.sha256(ids[ix].tobytes()).hexdigest() for k,ix in q.items()},epochs=64,optimizer_updates=64*math.ceil(counts['train']/64)))
    private_path=B/'roles_private.npz'
    if private_path.exists():
        with np.load(private_path) as z:
            assert set(z.files)==set(private)
            for k,v in private.items():np.testing.assert_array_equal(z[k],v)
    else:np.savez_compressed(private_path,**private)
    source=Path('/workspace/uwb-mmd-gap64-20261106')
    assert not(source/'SUMMARY.json').exists() and not(source/'FINALIZE_ONCE.lock').exists()
    assert read(source/'STATE.json')['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate'
    with np.load(source/'run/train/split.npz') as z:
        for k in ['train','validation','diagnostic']:np.testing.assert_array_equal(z[k],private['held1_'+k])
    for path,digest in p['frozen_reference_files'].items():assert sha(path)==digest,path
    assert sha(source/'FROZEN_HASHES.json')==p['MMD_pilot_scientific_manifest_sha256']
    for n,h in read(source/'FROZEN_HASHES.json').items():assert sha(source/n)==h,n
    baseline=G/'results/cloud_20261030_gap_source6/COMPLETE_AUDIT.json';old=read(baseline)
    for x,y in zip(folds,old['folds']):assert x['held']==y['held'] and x['optimizer_updates']==y['optimizer_steps']
    baseline_dir=Path('/workspace/uwb-gap-sixfold-20261030/run01')
    inventory={str(h):dict(selected_model_present=(baseline_dir/f'held{h}/best.pt').exists(),prediction_present=(baseline_dir/f'held{h}/predictions_unscored.npz').exists(),persistent_receipt=str(G/f'results/cloud_20261030_gap_source6/HELD{h}_DRIVE_BACKUP_RECEIPT.json') if h<6 else str(G/'results/cloud_20261030_gap_source6/FINAL_DRIVE_BACKUP_RECEIPT.json')) for h in range(2,7)}
    q=dict(state='roles_metrics_and_frozen_reference_preflight_pass',observed_unix=time.time(),synthetic_tests=synthetic_checks(),folds=folds,private_roles_sha256=sha(private_path),actual_sample_ID_values_in_public_report=False,MMD_fold1_IDs_exact=True,MMD_fold1_unscored=True,MMD_old_single_fold_gate_not_invoked=True,baseline_fold2_6_inventory=inventory,baseline_recovery_required=not all(x['selected_model_present'] and x['prediction_present'] for x in inventory.values()),no_baseline_refit_needed_unless_recovery_or_native_compatibility_audit_fails=True,new_training_started=False,users7_8_labels_used=False)
    (B/'PREFLIGHT_RESULT.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps({k:v for k,v in q.items() if k!='folds'}))
if __name__=='__main__':main()
