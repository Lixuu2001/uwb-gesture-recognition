"""Audit-pending private insurance for an actually completed Adam100 producer."""
from pathlib import Path
import argparse,json,sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    b=Path(parser.parse_args().root).resolve();sys.path.insert(0,str(b))
    from trial_common import ready,read,write,sha,term_ok
    from backup import scientific_files
    from stream_archive import build
    c=ready();term=read(b/'supervisor01/TERMINATION.json');assert term_ok(term)
    out=b/'run01';assert read(out/'STATE.json')['state']=='producer_complete_diagnostic_unscored'
    endpoint=read(out/'TRAIN_ENDPOINT.json')
    assert endpoint['epochs']==100 and endpoint['updates']==23500
    assert endpoint['endpoint']=='fixed_epoch100' and endpoint['best_validation_selection_used'] is False
    assert endpoint['diagnostic_labels_read'] is False and endpoint['users7_8_labels_used'] is False
    assert not (out/'SUMMARY.json').exists(),'No rebuilding or rescoring completed diagnostic results'
    hashes=read(out/'FILES.json')
    required=['initial12_head.pt','final_model.pt','final_full.pt','split.npz',
        'steps.jsonl','TRAIN_ENDPOINT.json','validation_unscored.npz',
        'diagnostic_unscored.npz','IDENTITY.json','STATE.json']
    required += [f'epoch{i:03d}.json' for i in range(1,101)]
    required += [f'validation_epoch{i:03d}.npz' for i in range(1,101)]
    assert all(name in hashes for name in required)
    for name,digest in hashes.items():
        assert Path(name).name==name and sha(out/name)==digest,name
    target=b/'producer_complete_insurance';assert not target.exists(),'Existing upload IDs/receipts require inspection, no rebuild'
    files=scientific_files(c)
    for name in ['FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json',
                 'INITIAL_PRIVATE_INSURANCE.json','CHECKPOINT45_PRIVATE_INSURANCE.json',
                 'CONTROLLER_LAUNCH.json']:
        files['study/'+name]=b/name
    for name in [*hashes,'FILES.json']:files['run01/'+name]=out/name
    for path in sorted((b/'supervisor01').iterdir()):
        if path.is_file():files['supervisor01/'+path.name]=path
    files['operations/preserve_adam100_completed_producer.py']=Path(__file__).resolve()
    m=build(target,'adam100_recovery45_producer_complete_20261103',files,
        'producer_complete_ShuffleNetV2_x1_Adam100_checkpoint45_recovery_without_numeric_audit',
        chunk_size=c['streaming_chunk_bytes'],limit=c['archive_upper_bytes'])
    m.update(state='complete_Adam100_producer_unscored_audit_pending_insurance',
        complete_modelset=False,full_training_complete=True,
        numeric_audit_certified=False,diagnostic_labels_scored=False)
    write(target/'MANIFEST.json',m)
    for name,digest in hashes.items():assert sha(out/name)==digest,name
    print(json.dumps(dict(directory=str(target),parts=len(m['chunks']),
        archive_bytes=m['archive_bytes'],payload_members=m['tracked_payload_files'],
        producer_updates=23500,numeric_audit_certified=False,diagnostic_labels_scored=False)))

if __name__=='__main__':main()
