"""Preserve actual independent-auditor progress; no fitting, inference or labels."""
from pathlib import Path
import sys,json,time,shutil,subprocess,re,hashlib
from delta import read,write,sha,create,apply
B=Path(__file__).resolve().parent
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
BASE=Path('/workspace/uwb-source6-held3-audit-delta04-20261104')
G=Path('/workspace/uwb-cloud-current')
sys.path.insert(0,str(ROOT))
from common import privacy,alive,process_identity,config
from stream_archive import verify,build
PARENT='04932f425a1ca97ef9f1190389369dc3ba7505b8'
BASE_SHA='ded25c2be3c18e3f49af9e558392408f7a979f241d1811cb22e75f67e8b6da91'
ANCESTOR_SHA='c77fe20741e4e2cdde385c163a30ae44708379f8bc4bd0dd80422c495f6c3906'
RP='uwb_direction1/results/cloud_20261104_held3_audit_delta05'
CP='uwb_direction1/research/cloud_20261104_held3_audit_delta05'
CODE=['snapshot.py','augment_capture.py','verify_native.py','delta.py','pipeline.py','README.md']
def proof(root,arm):return read(root/f'{arm}_SNAPSHOT_VERIFIED.json')
def prepare():
    for arm in ['timemil','shuffle']:
        q=proof(B/'capture01',arm);assert q['native_instantiation_restore_recapture_exact'] and q['complete_member_manifest_verified'] and q['selected_model_matches_native_committed_state']
    q=proof(B/'capture01','shuffle');assert q['own_replay_epoch']>=74 and q['own_checkpoint_native_restore_recapture_exact'] and q['own_replay_prefix_of_producer_steps_verified']
    archive=BASE/'delta04_backup';m=read(archive/'MANIFEST.json');v=read(archive/'READBACK_VERIFIED.json');rows=read(archive/'READBACK_INPUT.json');privacy(rows)
    assert m['archive_sha256']==v['archive_sha256']==BASE_SHA and m['chain_depth']==1
    assert verify(m,rows)==v and sha(archive/'MEMBERS.json')==m['members_manifest_sha256']
    table={x['path']:x for x in read(archive/'MEMBERS.json')['members']}['delta_payload/CURRENT_PAYLOAD_HASHES.json']
    assert sha(BASE/'reconstructed01/PAYLOAD_HASHES.json')==table['sha256']
    p=create(BASE/'reconstructed01',B/'capture01',B/'delta_payload',BASE_SHA)
    p.update(chain_depth=2,ancestor_archive_sha256=ANCESTOR_SHA,base_reconstruction_required=True,base_index_repository_path='uwb_direction1/results/cloud_20261104_held3_audit_delta04/delta04_backup/CLOUD_DOWNLOAD_INDEX.json')
    write(B/'delta_payload/DELTA.json',p)
    shutil.copyfile(archive/'CLOUD_DOWNLOAD_INDEX.json',B/'delta_payload/BASE_DELTA04_INDEX.json')
    shutil.copyfile(BASE/'delta_payload/BASE_ARCHIVE_INDEX.json',B/'delta_payload/ANCESTOR_PRODUCER_INSURANCE03_INDEX.json')
    r=apply(BASE/'reconstructed01',B/'delta_payload',B/'reconstructed01');write(B/'LOCAL_RECONSTRUCTION.json',r)
    assert sha(B/'capture01/PAYLOAD_HASHES.json')==sha(B/'reconstructed01/PAYLOAD_HASHES.json')
    for n in ['snapshot.py','augment_capture.py','verify_native.py','delta.py']:assert sha(B/n)==sha(BASE/n),n
    write(B/'BASE_VERIFIED.json',dict(state='prior_raw_delta_and_reconstructed_payload_table_verified',archive_sha256=BASE_SHA,ancestor_archive_sha256=ANCESTOR_SHA,chain_depth=2,base_payload_table_sha256=p['base_payload_table_sha256'],current_payload_table_sha256=p['current_payload_table_sha256'],all_four_helpers_unchanged=True))
    return dict(state=r['state'],current_files=r['files'],changed_files=r['changed_files'],own_replay_epoch=q['own_replay_epoch'])
def observe():
    from checks import verify_member,verify_audits
    from controller import valid_fold,budget_reason
    ident=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(ident) and ident['pid']==573 and ident['startticks']==34322 and ident['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
    audit=process_identity(1754);assert audit and audit['startticks']==801143 and audit['boot_id']==ident['boot_id'] and audit['state']!='Z'
    assert read(ROOT/'STATE.json')==dict(held=3,state='auditing_members_unscored')
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
    assert valid_fold(2);verify_audits(2)
    for arm in ['timemil','shuffle']:
        verify_member(3,arm)
        t=read(ROOT/'attempts/attempt002/held3'/f'{arm}_train_TERM.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
    cert=read(ROOT/'folds/held3/timemil/inference_audit/CERTIFICATE.json');assert cert['state']=='pass' and not cert['full_training_replayed']
    t=read(ROOT/'folds/held3/timemil/inference_audit/TERMINATION.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
    for n,h in cert['artifact_hashes'].items():assert sha(n)==h
    ledger=read(ROOT/'RESOURCE_LEDGER.json');_,p=config();assert ledger['folds']['2']==dict(wall=22834.174004506,cpu=84851.68278)
    assert budget_reason(ledger['folds'],3,p,ledger['last_observed_combined_RSS_KiB']) is None
    captured=read(B/'capture01/folds/held3/shuffle/replay/PROGRESS.json');current=read(ROOT/'folds/held3/shuffle/replay/PROGRESS.json')
    assert 74<=captured['epoch']<=current['epoch']<100
    o=dict(state='same_registered_held3_independent_replay_advanced_no_scientific_change',observed_unix=time.time(),controller=process_identity(573),auditor=process_identity(1754),captured_own_replay=captured,current_own_replay=current,resource_ledger=ledger,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),complete_producer_epochs={'timemil':64,'shuffle':100},TimeMIL_output_audit_passed=True,TimeMIL_full64_training_replayed=False,full100_shuffle_replay_complete=False,held2_not_retrained=True,budget_reset=False,scientific_configuration_changed=False,new_training_process_started=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,original_contribution_validated=False,data_redownloaded=False)
    write(B/'LIVE_OBSERVATION.json',o);return o
def archive():
    for arm in ['timemil','shuffle']:
        a=proof(B/'capture01',arm);r=proof(B/'reconstructed01',arm)
        for k in ['state_sha256','checkpoint_sha256','committed_epoch','committed_updates','own_replay_epoch','native_instantiation_restore_recapture_exact','retained_output_audit_verified','own_checkpoint_native_restore_recapture_exact','own_replay_prefix_of_producer_steps_verified','own_epoch_records_exact']:assert a[k]==r[k],(arm,k)
    o=observe();q=proof(B/'reconstructed01','shuffle');assert q['own_checkpoint_native_restore_recapture_exact'] and q['own_replay_prefix_of_producer_steps_verified']
    names=CODE+['CAPTURE.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','RECONSTRUCTED_TIMEMIL.log','RECONSTRUCTED_SHUFFLE.log','LOCAL_RECONSTRUCTION.json','BASE_VERIFIED.json','LIVE_OBSERVATION.json']
    write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
    files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
    for d,prefix in [(B/'delta_payload','delta_payload'),(B/'capture01','capture_proofs'),(B/'reconstructed01','reconstructed_proofs')]:
        for f in d.rglob('*'):
            if f.is_file() and (prefix=='delta_payload' or f.name.endswith('_SNAPSHOT_VERIFIED.json')):files[prefix+'/'+str(f.relative_to(d))]=f
    m=build(B/'delta05_backup',f'uwb_held3_own{q["own_replay_epoch"]}_delta05_20261104',files,'held3_advanced_own_replay_checkpoint_delta_depth2_not_full_audit_or_scoring_authority',chunk_size=7340032,limit=268435456)
    m.update(state='held3_advanced_own_audit_checkpoint_delta_streamed',complete_modelset=False,delta_dependency=True,chain_depth=2,base_archive_sha256=BASE_SHA,ancestor_archive_sha256=ANCESTOR_SHA,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False)
    write(B/'delta05_backup/MANIFEST.json',m);return {k:v for k,v in m.items() if k!='chunks'}|dict(parts=len(m['chunks']))
def verify_backup():
    b=B/'delta05_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows)
    v=verify(m,rows);write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
    r=dict(state='held3_advanced_own_checkpoint_delta_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=True,chain_depth=2,base_archive_sha256=BASE_SHA,ancestor_archive_sha256=ANCESTOR_SHA,base_index_repository_path=read(B/'delta_payload/DELTA.json')['base_index_repository_path'],initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False,diagnostic_labels_read=False,new_accuracy_result=False,full_training_audit_complete=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
    write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));return v
def publication():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==PARENT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
    o=read(B/'LIVE_OBSERVATION.json');assert alive(o['controller'])
    m=read(B/'delta05_backup/MANIFEST.json');v=read(B/'delta05_backup/READBACK_VERIFIED.json');assert m['archive_sha256']==v['archive_sha256'] and m['chain_depth']==2
    d=read(B/'delta_payload/DELTA.json');q=proof(B/'reconstructed01','shuffle');r=G/RP;c=G/CP;r.mkdir(parents=True);c.mkdir(parents=True)
    for n in ['LIVE_OBSERVATION.json','LOCAL_RECONSTRUCTION.json','BASE_VERIFIED.json','SOURCE_HASHES.json']:shutil.copyfile(B/n,r/n)
    for role in ['capture01','reconstructed01']:
        for arm in ['timemil','shuffle']:shutil.copyfile(B/role/f'{arm}_SNAPSHOT_VERIFIED.json',r/f'{role}_{arm}_SNAPSHOT_VERIFIED.json')
    for n in ['DELTA.json','CURRENT_PAYLOAD_HASHES.json','BASE_DELTA04_INDEX.json','ANCESTOR_PRODUCER_INSURANCE03_INDEX.json']:
        f=r/'delta_payload'/n;f.parent.mkdir(exist_ok=True);shutil.copyfile(B/'delta_payload'/n,f)
    for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
        f=r/'delta05_backup'/n;f.parent.mkdir(exist_ok=True);shutil.copyfile(B/'delta05_backup'/n,f)
    for n in CODE:shutil.copyfile(B/n,c/n)
    write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in CODE})
    lr=o['resource_ledger']['folds']['3'];epoch=q['own_replay_epoch'];arc=m['archive_sha256'];count=len(read(B/'delta_payload/CURRENT_PAYLOAD_HASHES.json'))
    note=f'''## 2026-11-04 held3 same independent audit continues; own{epoch} state privately reconstructed

Same original bootad409b48/controller573/startticks34322 and independent auditor1754/startticks801143 remain active, with no restart, new fit, scientific/config/seed/gate change, diagnostic score or cap reset. Both producers and TimeMIL output audit remain complete; full100Shuffle replay is still unfinished (live epoch{o['current_own_replay']['epoch']} at observation). New intermediate delta05 preserves auditor OWN{epoch}/{225*epoch}updates. All{count}current payloads reconstructed in a NEW directory; both native states/selection/TimeMIL certificate and own-audit native restore/RNG/epoch-record/original-log-prefix checks pass. The four recovery helpers are unchanged, not new scientific methods or repeated model fitting.

Owner-only raw delta05: {len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads, SHA{arc}; whole archive/all members verified. Dependency depth2: producer insurance03(c77fe207...) -> delta04(ded25c2b...) -> delta05; BOTH indexes/guide included at `{RP.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. Never restore over newer/live state; use own auditor state and preserve cumulative caps/charges. Held3 cost{lr['wall']:.3f}wall/{lr['cpu']:.3f}CPU seconds within original43200/172800. After full audits/fusion, preserve a complete-fold archive to end the dependency chain. All5newfolds/full final private recovery still required before once aggregate; source1 reused/users7–8 excluded/exploratory repeated-user caveat unchanged. No new accuracy or originality claim.

'''
    report=f'''# Same registered audit, advanced own checkpoint insurance

{note}
The existing independent auditor continues the original full100Shuffle replay, with no producer restart or method/selection change. Both held3 producers (TimeMIL64/4544updates selected50; Shuffle100/22500updates fixed100) and TimeMIL selected-state/unscored-output audit were already complete and are reused. TimeMIL does NOT receive a full64 trajectory-replay claim. No diagnostic labels or new accuracy score are read. This continuation preserves substantial real audit progress because prior cloud boot losses occurred; it neither repeats training as a new experiment nor opens a rescue grid.

Captured own checkpoint epoch{epoch}/{225*epoch}updates has its full native model/optimizer/sampler/Torch/Python/NumPy RNG state, journal chain and exact original producer step-log prefix checked. Its{q['own_epoch_records_exact']}epoch records, including epoch0, match the original producer records; the initial state matches the registered seed. All{count}current payloads ({len(d['changed'])}changed) were reconstructed from the verified previous overlay into a NEW directory and independently passed both native model/optimizer/RNG/selection verifiers and retained TimeMIL certificate checks. Substantive hashes match the source capture; metrics from separate verification processes are not claimed byte-identical. The snapshot/augmentation/native/delta helpers are byte-identical to delta04; actual new data-state verification was performed, no fitting or inference by the preservation code.

Private delta archive SHA `{arc}`: {len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads plus member manifest. Metadata establishes owner-only/nonshared private parent, MIME/size before download_raw_file=true/include_base64=false fetch; immediate materialization/SHA/size and whole archive/every member pass, at most3complete concurrent pipelines. Actual IDs, tensors, optimizer/RNG state, unscored binary predictions and complete logs remain private. Git contains helpers, hashes, observations, proofs and indexes; original789source snapshot and registered60scientific files are unchanged.

This delta is NOT standalone. Required chain depth2: raw-verify producer insurance03 archive `{ANCESTOR_SHA}` and identify its overlay/; raw-verify delta04 archive `{BASE_SHA}` and apply it into a NEW directory; raw-verify delta05 and apply it over that exact reconstructed delta04 overlay into another NEW directory. BASE_DELTA04_INDEX.json and ANCESTOR_PRODUCER_INSURANCE03_INDEX.json are included. Original initial/source/seed/ID/data/maps/runtime and preregistration dependencies remain required. Never modify the base or overwrite live/newer state. Only confirmed loss and all old identities inactive permit original controller --resume from verified state, using the auditor's OWN checkpoint and retaining every cumulative cap/previous conservative interruption charge. Do not refit completed models or seed an auditor from producer-final state.

This is intermediate audit insurance, not full100audit completion, a completed three-arm fold or final scoring authority. After full replay/output audits/fusion, preserve a non-delta complete-fold archive rather than extending the chain further. Actual cumulative held3 wall{lr['wall']:.6f}/CPU{lr['cpu']:.6f}seconds is within unchanged43200wall/172800CPU limits; all5newfold216000wall/864000CPU caps remain. The same controller must still finish held3 and held4–6, all3arms/audits/frozen outputs and FULL final private recoverability before once-only source-six diagnostic aggregation. Existing repeated source1/eight-user development remains exploratory, source1 is reused without rescoring and users7–8 labels remain excluded from new-method selection. Accuracy improvement and original-contribution evidence remain separate and unresolved.
'''
    (r/'RUN_REPORT.md').write_text(report)
    updates=dict(state='source6_held3_same_independent_audit_running_advanced_own_checkpoint_insured',latest_report=RP.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=573,background_training_observed_unix=o['observed_unix'],background_training_status_note=f'Same controller573/bootad409b48 and auditor1754, no restart or scientific change. Own{epoch}checkpoint reconstructed/native-verified and privately raw-backed by delta05 over insurance03->delta04, depth2. Full100audit unfinished, no new accuracy/originality result; cumulative caps retained.')
    key='source6_held3_audit_delta05_20261104';insert=dict(report=updates['latest_report'],observation=RP.removeprefix('uwb_direction1/')+'/LIVE_OBSERVATION.json',private_download_index=RP.removeprefix('uwb_direction1/')+'/delta05_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,base_archive_sha256=BASE_SHA,ancestor_archive_sha256=ANCESTOR_SHA,delta_chain_depth=2,captured_own_replay_epoch=epoch,current_payloads=count,changed_payloads=len(d['changed']),full100_shuffle_replay_complete=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
    paths=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
    for path in paths:
        f=G/path;body=f.read_text()
        if path.endswith('STATUS.json'):
            assert '"'+key+'"' not in body
            for k,val in updates.items():
                rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
                body,n=re.subn(rx,lambda m:m[1]+json.dumps(val,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
            body=body.rstrip();assert body.endswith('}');body=body[:-1]+',\n  "'+key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n';json.loads(body)
        else:body=note+body
        f.write_text(body)
    assert not subprocess.check_output(['git','diff','--name-only',PARENT,'--','uwb_direction1/original'],cwd=G,text=True).strip()
    files=[]
    for path in paths+[str(f.relative_to(G)) for f in sorted(r.rglob('*')) if f.is_file()]+[str(f.relative_to(G)) for f in sorted(c.rglob('*')) if f.is_file()]:
        raw=(G/path).read_bytes();assert not path.endswith(('.pt','.pth','.npy','.npz','.bin','.jsonl'))
        row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        if path in paths:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
        files.append(row)
    write(B/'PUBLICATION_MANIFEST.json',dict(root=str(G),parent=PARENT,files=files,note=note,status_updates=updates,status_insert_key=key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2)))
    return dict(state='advanced_own_audit_checkpoint_publication_prepared',files=len(files),archive_sha256=arc,own_replay_epoch=epoch,delta_dependency_depth=2,new_accuracy_result=False)
if __name__=='__main__':
    assert len(sys.argv)==2
    commands={'prepare':prepare,'archive':archive,'verify':verify_backup,'publication':publication}
    print(json.dumps(commands[sys.argv[1]]()))
