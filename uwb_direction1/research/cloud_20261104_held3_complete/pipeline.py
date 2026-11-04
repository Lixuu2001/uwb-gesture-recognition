"""Complete unscored held3 archive; preserve original training and sealed truth."""
from pathlib import Path
import sys,json,time,hashlib,shutil,subprocess,re
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');G=Path('/workspace/uwb-cloud-current');sys.path.insert(0,str(ROOT))
from common import read,write,sha,privacy,alive,process_identity,require_initial,config
from checks import verify_audits
from controller import valid_fold,budget_reason
from stream_archive import build,verify
PARENT='ccd438ffe43b7d4af9840896fc58b4f9593c86eb'
RP='uwb_direction1/results/cloud_20261104_held3_complete'
CP='uwb_direction1/research/cloud_20261104_held3_complete'
def observe():
    identity=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(identity) and identity['pid']==573 and identity['startticks']==34322 and identity['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
    for h in [2,3]:assert valid_fold(h);verify_audits(h)
    state=read(ROOT/'STATE.json');assert state==dict(held=4,state='training_fixed_members_unscored',users7_8_labels_used=False)
    c,p=config();ledger=read(ROOT/'RESOURCE_LEDGER.json');assert budget_reason(ledger['folds'],4,p,ledger['last_observed_combined_RSS_KiB']) is None
    assert ledger['folds']['3']['wall']<=p['per_fold_wall_seconds'] and ledger['folds']['3']['cpu']<=p['per_fold_CPU_seconds']
    assert ledger['folds']['2']==dict(wall=22834.174004506,cpu=84851.68278)
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
    identities=read(ROOT/'attempts/attempt002/held4/PROCESSES.json')['identities'];active=[process_identity(x['pid']) for x in identities if alive(x)];assert len(active)==2
    fold=ROOT/'folds/held3';full=read(fold/'shuffle/replay/COMPLETE_AUDIT.json');t=read(fold/'shuffle/replay/TERMINATION.json')
    assert full['state']=='pass' and full['all100epochs_all_updates_replayed_exactly'] and full['initial_audit_started_from_fresh_seed'] and full['updates']==22500 and t['exit_code']==0
    o=dict(state='held3_protocol_complete_audited_three_arms_frozen_unscored_original_controller_training_held4',observed_unix=time.time(),controller=process_identity(573),current_state=state,active_children=active,held4_progress={a:read(ROOT/'folds/held4'/a/'PROGRESS.json') for a in ['timemil','shuffle']},held3_member_progress={a:read(fold/a/'PROGRESS.json') for a in ['timemil','shuffle']},held3_full_shuffle_audit=full,held3_shuffle_replay_termination=t,held3_TimeMIL_output_certificate={k:v for k,v in read(fold/'timemil/inference_audit/CERTIFICATE.json').items() if k!='artifact_hashes'},held3_Shuffle_output_certificate={k:v for k,v in read(fold/'shuffle/inference_audit/CERTIFICATE.json').items() if k!='artifact_hashes'},held3_frozen={k:v for k,v in read(fold/'fusion/FOLD_FROZEN.json').items() if k!='files'},held3_fusion_termination=read(fold/'FUSION_TERMINATION.json'),resource_ledger=ledger,budget_reset=False,scientific_configuration_changed=False,new_process_started_by_preservation=False,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),new_accuracy_result=False,diagnostic_labels_read=False,users7_8_labels_used=False,original_contribution_validated=False,source6_all5newfolds_complete=False)
    write(B/'LIVE_OBSERVATION.json',o);write(B/'COMPLETED_FOLD_RESOURCE.json',dict(held=3,terminal_fold_cost=ledger['folds']['3'],all_prior_interruption_charges_preserved=True,budget_reset=False));return o
def build_backup():
    require_initial();o=observe();fold=ROOT/'folds/held3';files={}
    for f in fold.rglob('*'):
        if f.is_file() and not f.name.endswith(('.lock','.tmp')) and '__pycache__' not in f.parts:files['folds/held3/'+str(f.relative_to(fold))]=f
    for directory in (ROOT/'attempts').glob('*/held3'):
        for f in directory.rglob('*'):
            if f.is_file() and not f.name.endswith(('.lock','.tmp')):files['attempts/'+str(f.relative_to(ROOT/'attempts'))]=f
    for n in ['CONFIG.json','METHOD_LOCK.json','FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json','SEED_FILES.json','ASSET_FILES.json','DEPENDENCIES.json']:files['source_lineage/'+n]=ROOT/n
    for n in ['CLOUD_DOWNLOAD_INDEX.json','DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json']:files['initial_archive_lineage/'+n]=ROOT/'initial_backup'/n
    for n in read(ROOT/'FROZEN_HASHES.json'):
        if n.endswith(('.py','.md','.json','.log')):files['registered_source/'+n]=ROOT/n
    for n in ['pipeline.py','README.md','LIVE_OBSERVATION.json','COMPLETED_FOLD_RESOURCE.json']:files['preservation/'+n]=B/n
    write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in ['pipeline.py','README.md','LIVE_OBSERVATION.json','COMPLETED_FOLD_RESOURCE.json']});files['preservation/SOURCE_HASHES.json']=B/'SOURCE_HASHES.json'
    m=build(B/'held3_complete_backup','softvote_source6_held3_complete_20261104',files,'complete_held3_protocol_audits_three_arms_unscored_no_delta_dependency_not_final_source6_scoring_authority',chunk_size=7340032,limit=536870912)
    m.update(state='complete_unscored_held3_all_protocol_audits_and_outputs_streamed',held=3,delta_dependency=False,initial_data_runtime_source_dependencies_required=True,source6_all5newfolds_complete=False,final_scoring_authority=False)
    write(B/'held3_complete_backup/MANIFEST.json',m);return {k:v for k,v in m.items() if k!='chunks'}|dict(parts=len(m['chunks']))
def verify_backup():
    assert valid_fold(3);verify_audits(3)
    b=B/'held3_complete_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows)
    v=verify(m,rows);assert v['complete_modelset_verified'] and not m['delta_dependency'] and not m['final_scoring_authority']
    entries={x['path']:x for x in read(b/'MEMBERS.json')['members']}
    for arm in ['timemil','shuffle']:
        for n in ['selected.pt','final_full.pt','ARM_COMPLETE.json','inference_audit/CERTIFICATE.json','inference_audit/TERMINATION.json']:
            k=f'folds/held3/{arm}/{n}';assert entries[k]['sha256']==sha(ROOT/k)
    for n in ['shuffle/replay/COMPLETE_AUDIT.json','shuffle/replay/TERMINATION.json','fusion/FOLD_FROZEN.json','fusion/diagnostic_unscored.npz','fusion/validation_unscored.npz','FUSION_TERMINATION.json']:
        k='folds/held3/'+n;assert entries[k]['sha256']==sha(ROOT/k)
    assert sha(b/'MEMBERS.json')==m['members_manifest_sha256']
    write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
    r=dict(state='complete_held3_models_all_protocol_audits_and_three_arms_raw_privately_verified',held=3,scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=False,initial_data_runtime_source_dependencies_required=True,source6_all5newfolds_complete=False,final_scoring_authority=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
    write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));return v
def publication():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==PARENT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
    original=read(B/'LIVE_OBSERVATION.json');o=observe()
    assert o['resource_ledger']['folds']['3']==original['resource_ledger']['folds']['3']
    # Preserve the exact archived observation instead of changing its source hash.
    shutil.copyfile(B/'LIVE_OBSERVATION.json',B/'POST_BACKUP_LIVE_OBSERVATION.json');write(B/'LIVE_OBSERVATION.json',original)
    m=read(B/'held3_complete_backup/MANIFEST.json');v=read(B/'held3_complete_backup/READBACK_VERIFIED.json');assert v['archive_sha256']==m['archive_sha256'] and v['complete_modelset_verified']
    r=G/RP;c=G/CP;r.mkdir(parents=True);c.mkdir(parents=True)
    for n in ['LIVE_OBSERVATION.json','POST_BACKUP_LIVE_OBSERVATION.json','COMPLETED_FOLD_RESOURCE.json','SOURCE_HASHES.json']:shutil.copyfile(B/n,r/n)
    for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
        p=r/'held3_complete_backup'/n;p.parent.mkdir(exist_ok=True);shutil.copyfile(B/'held3_complete_backup'/n,p)
    for n in ['pipeline.py','README.md']:shutil.copyfile(B/n,c/n)
    write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in ['pipeline.py','README.md']})
    cost=o['resource_ledger']['folds']['3'];p4=o['held4_progress'];arc=m['archive_sha256']
    note=f'''## 2026-11-04 held3 complete: full100Shuffle replay, both output audits and three-arm freeze passed

Same original bootad409b48/controller573/startticks34322/attempt002 continues; held3 full100Shuffle independent replay matched ALL22500updates/epoch states/order/loss/validation outputs and final native state, actual exit0/no stop/no OOM. Both selected-output audits and fixed three-arm vector/scalar/ID fusion passed with actual successful terminations; held3 is frozen UNScored. TimeMIL64/4544updates selected50 has output/native-state audit, NOT full64 training replay. No diagnostic accuracy/originality claim or scientific/config/seed/gate change. Completed held3 cumulative cost{cost['wall']:.6f}wall/{cost['cpu']:.6f}CPU seconds includes both interruption charges, within original caps and unchanged after entering held4.

New complete-fold private archive: {len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads, SHA{arc}; all raw parts/whole archive/every member verified, owner-only folder unchanged. Complete models/all required audits/actual terminations/three-arm unscored outputs/IDs/logs/registered source and initial lineage are included. NO earlier checkpoint-delta dependency; initial data/runtime/source still required. Details/index: `{RP.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. This ends held3's insurance03->delta04->delta05 dependency for completed-fold recovery; prior forensic archives remain retained. It is not all5fold final scoring authority.

Original controller already trains held4 (TimeMIL epoch{p4['timemil']['epoch']}, Shuffle epoch{p4['shuffle']['epoch']} at current observation); no duplicate, refit or restart by preservation. Newfolds2/3 complete,4in progress,5/6pending. Inspect current identities/checkpoints before actions. Continue all5newfolds/3arms/audits/frozen outputs and FULL final private recovery before once source-six aggregate. Source1 reused without rescoring, users7–8 excluded, repeated source/eight-user development remains exploratory; accuracy and originality evidence remain unresolved.

'''
    report=f'''# Held3 protocol complete and independently replayed; original experiment continues on held4

{note}
## Actual completion evidence and limits

Parent `{PARENT}` preserved own75 while the audit was running. The SAME original controller completed the independent Shuffle full100 replay, both selected-output audits and fixed three-arm fusion, then advanced to held4 under its already registered sequence. No process was restarted or fitting duplicated by this continuation. `POST_BACKUP_LIVE_OBSERVATION.json` contains current actual controller/child identities, held4 progress, complete audit/termination evidence and cumulative ledgers. Completed held2 remains unchanged and is not uploaded again.

The Shuffle auditor started from the registered fresh seed and used only its own checkpoint for recovery. It matched all22500native updates, sample-order/training-ID hashes, losses,100epoch states/optimizer/sampler/RNG records, validation logits and final producer native state; full audit termination is actual exit0/no timeout/no resource stop/no cgroup OOM. Shuffle selected-model validation and unscored diagnostic logits also reproduced exactly in a separate successful output audit. TimeMIL selectedepoch50 from its fixed64epoch run passed complete native/journal/optimizer/RNG/selection and all validation/unscored-diagnostic output checks with actual successful termination; this is explicitly NOT a full64 training-trajectory replay claim.

The fixed equal-weight probability mixture passed the independent vector/scalar comparison (absolute tolerance1e-14), sample-ID/ordering and prediction checks before three-arm outputs were frozen. `FOLD_FROZEN.json` and its actual fusion termination establish immutable unscored TimeMIL/Shuffle/mixture outputs. No held diagnostic label or source-six diagnostic accuracy is read. All60 frozen scientific files remain exact; original789source snapshot is untouched. Both prior real boot-loss interruption charges remain in the completed held3 ledger, with NO cap reset. Held3 terminal cost is{cost['wall']:.6f}wall/{cost['cpu']:.6f}CPU seconds, within43200/172800 per-fold and original five-fold216000/864000 limits.

## Complete-fold private recoverability

Archive SHA256 `{arc}` contains{m['tracked_payload_files']}payloads plus member manifest, {len(m['chunks'])}parts and{m['archive_bytes']}bytes. It contains the whole immutable held3 directory (selected/final/initial/optimizer/RNG models, all original logs/IDs/epoch journals/validation arrays, complete replay state/audit proofs, both output certificates and actual terminations, frozen three-arm unscored predictions), all actual held3 attempt histories (including interrupted attempts without invented successful terminations), frozen registered source/configuration and initial-archive lineage. Actual sample IDs, tensor states, binary prediction arrays and full training logs remain private. Raw radar arrays and runtime binaries are not duplicated or put in Git.

Every part's metadata establishes owner-only/nonshared original folder, correct parent/MIME/size before raw fetch with download_raw_file=true/include_base64=false; immediate materialization/size/SHA verification and whole archive/every member readback pass, at most3 complete concurrent pipelines. Critical selected/final models, audit certificates/terminations and frozen outputs were independently matched against raw archive member digests. No authenticated URL is output or committed. The complete-modelset verification is scoped to held3, not all five new folds.

The completed held3 archive has NO dependency on the prior insurance03/delta04/delta05 checkpoint chain. Keep those archives as forensic history, but completed-fold recovery uses this new complete archive plus original initial data/runtime/source dependencies. Restore only into a NEW inspection directory after confirmed loss, verify metadata/raw archive/members/native/audits/IDs/source hashes, preserve existing/newer/live files and cumulative ledgers, and never rerun completed model fitting or use its held scores to select a method. This archive does not substitute for eventual FULL all5fold private recovery or permit early diagnostic aggregation.

## Remaining registered experiment

Original controller573/current boot is still active with both held4 trainers (TimeMIL{p4['timemil']['epoch']}, Shuffle{p4['shuffle']['epoch']}epochs at post-backup observation). The original sequence handles first-update audits, producer fitting, full Shuffle replay/output audits and three-arm freeze for held4–6. Only all five new folds/three arms/audits/frozen outputs AND full final private recoverability permit the once-only source-six aggregate and unchanged bounded accuracy gates. Source1 is reused without rescoring; users7–8 labels remain excluded from new-method selection. Prior repeated source1/eight-user development is exploratory. No new accuracy estimate, eight-fold improvement, independent-validation claim or original research contribution follows from this completion milestone.
'''
    (r/'RUN_REPORT.md').write_text(report)
    updates=dict(state='source6_held3_protocol_complete_privately_verified_original_held4_training',latest_report=RP.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=573,background_training_observed_unix=o['observed_unix'],background_training_status_note='Held3 full100Shuffle replay/all22500updates, both output audits and three-arm freeze passed with actual exit0; complete unscored fold privately raw/allmember verified with no checkpoint-delta dependency. Same controller573/bootad409b48 now trains held4. No restart/scientific/gate/seed/budget change, diagnostic accuracy or originality claim.',compute='2026-11-04 bootad409b48-89ce-4704-beee-3658103d5856; CPU quota4/RAM16GiB; Python3.12.14 NumPy2.3.5 Torch2.15.0+cpu; actual held4 native trainers1+4threads; no GPU')
    key='source6_held3_complete_20261104';insert=dict(report=updates['latest_report'],observation=RP.removeprefix('uwb_direction1/')+'/POST_BACKUP_LIVE_OBSERVATION.json',private_download_index=RP.removeprefix('uwb_direction1/')+'/held3_complete_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,checkpoint_delta_dependency=False,held3_protocol_complete_unscored=True,Shuffle_full100_training_replayed=True,Shuffle_exact_updates=22500,TimeMIL_full64_training_replayed=False,completed_newfolds=[2,3],active_held=4,all5newfolds_complete=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
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
        raw=(G/path).read_bytes();assert not path.endswith(('.pt','.pth','.npz','.npy','.bin','.jsonl'))
        row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        if path in paths:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
        files.append(row)
    write(B/'PUBLICATION_MANIFEST.json',dict(root=str(G),parent=PARENT,files=files,note=note,status_updates=updates,status_insert_key=key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2)))
    return dict(state='complete_held3_unscored_evidence_prepared',files=len(files),archive_sha256=arc,checkpoint_delta_dependency=False,held4_current_epochs={a:p4[a]['epoch'] for a in p4},new_accuracy_result=False)
if __name__=='__main__':
    assert len(sys.argv)==2;commands={'build':build_backup,'verify':verify_backup,'publication':publication};print(json.dumps(commands[sys.argv[1]]()))
