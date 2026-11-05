"""Complete producer plus coherent OWN fresh-replay checkpoint; no training/scoring."""
from pathlib import Path
import sys,json,time,shutil,hashlib
S=Path('/workspace/uwb-mixup-gap64-20261105');D=Path(__file__).resolve().parent
sys.path.insert(0,str(S));sys.path.insert(1,str(S/'operator'))
from common import ready,read,write,sha
from controller import identity
from verify_completion_metadata import verify as completion
def capture():
    c=ready(True);proof=completion('train');old=read(S/'CONTROLLER_IDENTITY.json');a=identity(old['pid'])
    assert a and a['state']!='Z' and a['startticks']==old['startticks'] and a['boot_id']==old['boot_id']
    wid=read(S/'run/execution/replay/IDENTITY.json');active=identity(wid['pid']);assert active and active['state']!='Z' and active['startticks']==wid['startticks']
    r=S/'run/replay';p=read(r/'PROGRESS.json');assert 0<p['epoch']<64 and not (r/'COMPLETE.json').exists()
    slot=(r/p['slot']).read_bytes();assert hashlib.sha256(slot).hexdigest()==p['checkpoint_sha256']
    log=(r/'steps.jsonl').read_bytes();prefix=log[:p['prefix_bytes']]
    assert log.endswith(b'\n') and hashlib.sha256(prefix).hexdigest()==p['prefix_sha256'] and len(prefix.splitlines())==p['updates']
    assert (S/'run/train/steps.jsonl').read_bytes().startswith(log)
    out=D/'capture01';out.mkdir(exist_ok=False);study=out/'study';own=study/'run/replay';own.mkdir(parents=True)
    (own/p['slot']).write_bytes(slot);(own/'steps.jsonl').write_bytes(log);write(own/'PROGRESS.json',p)
    for epoch in range(p['epoch']+1):
        for n in [f'epoch{epoch:03}.json',f'journal{epoch:03}.json']:shutil.copyfile(r/n,own/n)
        assert read(r/f'epoch{epoch:03}.json')==read(S/f'run/train/epoch{epoch:03}.json')
    for n in ['initial.pt','split.npz']:shutil.copyfile(r/n,own/n)
    for folder in ['run/train','run/execution','operator']:
        for src in (S/folder).rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and not src.name.endswith(('.lock','.tmp')):
                dst=study/folder/src.relative_to(S/folder);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for n in [*read(S/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','INITIAL_ASSET.json','initial.pt','roles.npz','SYNTHETIC_RESULT.json','RESOURCE_RESULT.json','CONTROLLER_IDENTITY.json','LAUNCH.json','REAL_PREFLIGHT.json','MATCHED_BASELINE_IDENTITY.json']:
        dst=study/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(S/n,dst)
    ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['stage']=='replay' and ledger['wall']<43200 and ledger['cpu']<43200;write(study/'RESOURCE_LEDGER.json',ledger)
    for n,h in read(S/'DEPENDENCIES.json')['files'].items():
        assert sha(n)==h;dst=out/'native_source'/Path(n).relative_to(c['native_source']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dst)
    for label,path in [('private_IDs/source_ids.npz',c['ids']),('lineage/noMixup_reference_SUMMARY.json',c['source1_reference_summary']),('lineage/INITIAL_ARCHIVE_INDEX.json',S/'initial_backup/CLOUD_DOWNLOAD_INDEX.json')]:
        dst=out/label;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    table={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in out.rglob('*') if f.is_file()};write(out/'PAYLOAD_HASHES.json',table)
    obs=dict(state='complete64_producer_plus_own_fresh_replay_checkpoint_captured_unscored',observed_unix=time.time(),controller=identity(old['pid']),own_replay_identity=identity(wid['pid']),own_progress=p,own_log_rows=len(log.splitlines()),own_uncommitted_log_rows=len(log.splitlines())-p['updates'],own_log_exact_prefix_of_original=True,producer_canonical_completion=proof,resource_ledger=ledger,diagnostic_labels_read=False,users7_8_labels_used=False,new_fit_started_by_preservation=False,scientific_code_models_or_original_completion_modified=False,budget_reset=False)
    write(D/'OBSERVATION.json',obs);print(json.dumps(obs))
def native():
    out=D/'capture01';study=out/'study'
    for n,q in read(out/'PAYLOAD_HASHES.json').items():assert sha(out/n)==q['sha256'] and (out/n).stat().st_size==q['bytes']
    import engine as e
    from journal import load
    result={}
    for stage in ['train','replay']:
        state,p=load(study/'run'/stage);m,o,g=e.make();e.restore(state,m,o,g);rec=e.capture(m,o,g,state['epoch'],state['updates'],state['best_key'],state['best_model'],state['best_epoch'])
        assert e.semantic(rec)==e.semantic(state)
        result[stage]=dict(epoch=state['epoch'],updates=state['updates'],state_sha256=e.semantic(state),checkpoint_sha256=p['checkpoint_sha256'],selected_epoch=state['best_epoch'],native_model_optimizer_all_RNG_restore_recapture_exact=True)
    train=study/'run/train';c=read(train/'COMPLETE.json');q=read(train/'COMPLETION_STATE_METADATA_CORRECTION.json')
    assert q['actual_final_native_state_sha256']==result['train']['state_sha256'] and q['original_completion_record_sha256']==sha(train/'COMPLETE.json')
    assert sha(train/'final_full.pt')==result['train']['checkpoint_sha256'] and c['selected_sha256']==sha(train/'selected.pt')
    for role in ['validation','diagnostic']:assert c[role+'_outputs_sha256']==sha(train/(role+'_unscored.npz'))
    own=study/'run/replay';p=read(own/'PROGRESS.json');raw=(own/'steps.jsonl').read_bytes();assert (train/'steps.jsonl').read_bytes().startswith(raw)
    assert hashlib.sha256(raw[:p['prefix_bytes']]).hexdigest()==p['prefix_sha256'] and len(raw[:p['prefix_bytes']].splitlines())==p['updates']
    for epoch in range(p['epoch']+1):assert read(own/f'epoch{epoch:03}.json')==read(train/f'epoch{epoch:03}.json')
    proof=dict(state='pass_complete_producer_and_own_fresh_replay_native_checkpoint',stages=result,producer_full64_and_actual_termination_preserved=True,original_completion_metadata_alias_and_canonical_correction_preserved=True,own_records_exact=p['epoch']+1,own_full_log_prefix_exact=True,payload_files_verified=len(read(out/'PAYLOAD_HASHES.json')),new_forward_or_optimizer_update=False,diagnostic_labels_read=False,users7_8_labels_used=False,full64_own_replay_complete=False)
    write(D/'NATIVE_SNAPSHOT_VERIFIED.json',proof);print(json.dumps(proof))
def archive():
    assert read(D/'NATIVE_SNAPSHOT_VERIFIED.json')['state'].startswith('pass_')
    from stream_archive import build
    files={'capture/'+str(f.relative_to(D/'capture01')):f for f in (D/'capture01').rglob('*') if f.is_file()}
    for n in ['preserve.py','RESTORE_GUIDE.md','OBSERVATION.json','NATIVE_SNAPSHOT_VERIFIED.json']:files['insurance/'+n]=D/n
    m=build(D/'private_backup','mixup_gap64_producer_complete_replay_insurance01_20261105',files,'complete_Mixup_GAP64_producer_plus_partial_own_replay_checkpoint_not_final_pilot_or_scoring_authority',chunk_size=7340032,limit=268435456)
    m.update(state='complete_producer_and_partial_own_replay_checkpoint_streamed',complete_modelset=False,complete_producer=True,full64_own_replay_complete=False,final_scoring_authority=False);m.pop('shared_initial_encoder_included_once');write(D/'private_backup/MANIFEST.json',m)
    return m
def verify_private():
    from stream_archive import verify
    d=D/'private_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        z=r['metadata'];assert not z['shared'] and z['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and z['permissions']==[dict(role='owner',type='user')]
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='complete_producer_and_partial_own_replay_owner_only_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],complete_producer=True,full64_own_replay_complete=False,complete_pilot_modelset=False,final_scoring_authority=False,diagnostic_labels_read=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'DRIVE_BACKUP_RECEIPT.json',receipt);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt));return v
if __name__=='__main__':
    if sys.argv[1]=='capture':capture()
    elif sys.argv[1]=='native':native()
    else:print(json.dumps(archive() if sys.argv[1]=='archive' else verify_private()))
