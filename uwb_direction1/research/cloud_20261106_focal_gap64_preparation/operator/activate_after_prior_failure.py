"""One conditional activation from the ORIGINAL prior consumed once gate; never fit or score."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from common import B,ready,read,sha,write
from controller import identity
from stream_archive import verify
def main():
    c=ready();assert not (B/'ACTIVATION.json').exists() and not (B/'run').exists()
    prior=Path(c['prior_SD_summary']).parent;q=read(prior/'SUMMARY.json');state=read(prior/'STATE.json')
    assert q['predeclared_gate_passed'] is False and q['route']==c['prior_SD_exact_failed_route'] and q['finalizer_invocations']==1
    assert q['users7_8_labels_used'] is False and (prior/'FINALIZE_ONCE.lock').exists()
    assert state['finalizer_consumed'] and state['summary_sha256']==sha(prior/'SUMMARY.json')
    assert sha(prior/'FROZEN_HASHES.json')==c['prior_SD_scientific_manifest_sha256']
    release=read(prior/'REAL_RELEASE.json');assert sha(prior/'operator/finalize_once.py')==release['once_scorer_sha256']=='9321449c2a3924b1059fc0c051352eb88d06497a3ee393a76076f75188b9f528'
    for f in [prior/'CONTROLLER_IDENTITY.json',prior/'run/execution/train/IDENTITY.json',prior/'run/execution/replay/IDENTITY.json',prior/'run/execution/output/IDENTITY.json']:
        old=read(f);now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['pid','startticks','boot_id'])
    for stage in ['train','replay','output']:
        term=read(prior/f'run/execution/{stage}/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
    assert read(prior/'run/PREDICTIONS_FROZEN.json')['full_training_replayed']
    d=prior/'full_backup';manifest=read(d/'MANIFEST.json');proof=read(d/'READBACK_VERIFIED.json');rows=read(d/'READBACK_INPUT.json')
    assert manifest['complete_modelset'] and proof['complete_modelset_verified'] and verify(manifest,rows)==proof
    for row in rows:
        md=row['metadata'];assert md['shared'] is False and md['permissions']==[dict(role='owner',type='user')] and md['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert row['raw_fetch'] and row['download_verified'] and row['include_base64'] is False
    assert read(B/'SYNTHETIC_RESULT.json')['state']=='pass' and read(B/'RESOURCE_RESULT.json')['gate_passed']
    result=dict(state='activate_only_preregistered_Focal_after_original_SD_exact_once_gate_failed_closed',prior_summary_sha256=sha(prior/'SUMMARY.json'),prior_original_once_scorer_sha256=release['once_scorer_sha256'],prior_full_private_archive_sha256=manifest['archive_sha256'],prior_full_private_modelset_raw_every_member_reverified=True,prior_all_model_processes_inactive=True,conditional_preregistration_commit=read(B/'REMOTE_PREREGISTRATION.json')['commit'],frozen_scientific_manifest_sha256=sha(B/'FROZEN_HASHES.json'),prior_SD_summary_read=True,prior_SD_raw_diagnostic_labels_read=False,users7_8_labels_used=False,source_configuration_or_seed_changed=False,new_UWB_fit_started=False)
    write(B/'ACTIVATION.json',result);print(json.dumps(result))
if __name__=='__main__':main()
