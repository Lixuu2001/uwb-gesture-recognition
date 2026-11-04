from pathlib import Path
import sys, json, time, hashlib, re, shutil, subprocess
B=Path(__file__).resolve().parent
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
G=Path('/workspace/uwb-cloud-current')
sys.path.insert(0,str(ROOT))
from common import read, write, sha, process_identity, alive, config
from controller import valid_fold, budget_reason
from checks import verify_audits

parent='d01f8141e13ce278151f922236528c0a8cfa930f'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==parent
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
c,p=config()
for n,h in read(ROOT/'FROZEN_HASHES.json').items():
    assert sha(ROOT/n)==h,n
identity=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(identity)
assert identity['pid']==573 and identity['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
assert valid_fold(2);verify_audits(2)
ledger=read(ROOT/'RESOURCE_LEDGER.json')
assert ledger['folds']['2']==read(B/'PRE_RESOURCE_LEDGER.json')['folds']['2']
state=read(ROOT/'STATE.json');assert state['held']==3 and state['state']=='training_fixed_members_unscored'
assert budget_reason(ledger['folds'],3,p,ledger['last_observed_combined_RSS_KiB']) is None
suffix={}
for arm,count in [('timemil',71),('shuffle',22)]:
    old=read(B/'PRE_RESUME_LOGS.json')[arm]
    receipt=read(ROOT/'folds/held3'/arm/'SUFFIX_REPLAY_VERIFIED.json')
    assert receipt['original_committed_updates']==old['progress']['updates']
    assert receipt['uncommitted_updates_reproduced_exactly']==count
    before=(B/(arm+'_PRE_RESUME_STEPS.jsonl')).read_bytes()
    with (ROOT/'folds/held3'/arm/'steps.jsonl').open('rb') as f:
        assert f.read(len(before))==before
    suffix[arm]=dict(receipt=receipt,current_progress=read(ROOT/'folds/held3'/arm/'PROGRESS.json'),complete_old_raw_byte_prefix_verified=True)
backup=read(B/'held3_checkpoint_backup/DRIVE_BACKUP_RECEIPT.json')
assert backup['state']=='held3_native_checkpoint27_57_raw_allmember_verified'
assert backup['archive_sha256']=='bd9f35c5a15d3ddd12cecab355c94e7458574c9bf5a08c987b8f6431aa96628c'
attempt=ROOT/'attempts/attempt002/held3'
if not attempt.exists():
    matches=list((ROOT/'attempts').glob('*002/held3'));assert len(matches)==1;attempt=matches[0]
procs=read(attempt/'PROCESSES.json')['identities']
current=[process_identity(x['pid']) for x in procs if alive(x)]
assert len(current)==2 and all(x['boot_id']==identity['boot_id'] for x in current)
obs=dict(state='second_boot_recovery_verified_original_experiment_continues_unscored',observed_unix=time.time(),controller=process_identity(identity['pid']),active_children=current,attempt=str(attempt),controller_state=state,resource_ledger=ledger,resource_stop_reason=None,perfold_wall_limit=p['per_fold_wall_seconds'],perfold_CPU_limit=p['per_fold_CPU_seconds'],budget_reset=False,held2_completed_unscored_not_retrained=True,held3_members=suffix,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),scientific_configuration_changed=False,seed_changed=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,original_contribution_validated=False,data_redownloaded=False,all_five_new_folds_complete=False,private_checkpoint_archive_sha256=backup['archive_sha256'])
write(B/'LIVE_OBSERVATION.json',obs)

rp='uwb_direction1/results/cloud_20261104_second_boot_recovery_held3'
cp='uwb_direction1/research/cloud_20261104_second_boot_recovery_held3'
result=G/rp;code=G/cp;result.mkdir(parents=True);code.mkdir(parents=True)
evidence=['PRE_RECOVERY_OBSERVATION.json','PRE_RESUME_LOGS.json','PRE_CONTROLLER_IDENTITY.json','PRE_RESOURCE_LEDGER.json','PRE_STATE.json','RESUME_LAUNCH.json','POST_RECOVERY_VERIFIED.json','LIVE_OBSERVATION.json','timemil_PRIOR_SUFFIX_PROOF.json','shuffle_PRIOR_SUFFIX_PROOF.json','timemil_NATIVE_RESTORE_VERIFIED.json','shuffle_NATIVE_RESTORE_VERIFIED.json']
for n in evidence:shutil.copyfile(B/n,result/n)
for arm in ['timemil','shuffle']:
    shutil.copyfile(B/'capture01'/(arm+'_SNAPSHOT_VERIFIED.json'),result/(arm+'_SNAPSHOT_VERIFIED.json'))
for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
    dest=result/'held3_checkpoint_backup'/n;dest.parent.mkdir(exist_ok=True);shutil.copyfile(B/'held3_checkpoint_backup'/n,dest)
for n in ['verify_state.py','snapshot.py','verify_native.py','build_backup.py','verify_backup.py','prepare_publication.py','README.md']:
    shutil.copyfile(B/n,code/n)
write(code/'SOURCE_HASHES.json',{n:sha(code/n) for n in ['verify_state.py','snapshot.py','verify_native.py','build_backup.py','verify_backup.py','prepare_publication.py','README.md']})

tm=suffix['timemil']['current_progress']['epoch'];sh=suffix['shuffle']['current_progress']['epoch'];lr=ledger['folds']['3']
note=f'''## 2026-11-04 second boot interruption repaired; same held3 experiment resumed

The old boot529611fa/controller610 was absent; workspace/data and native states remained. Exact provider restart cause is unknown. Original controller --resume is actually running as PID573/startticks34322 on bootad409b48, with two verified live producers (held3 TimeMIL epoch{tm}, Shuffle epoch{sh} at observation). Native model/optimizer/sampler/Torch/Python/NumPy RNG states restored exactly. This interruption's71TimeMIL/22Shuffle uncommitted steps reproduced exactly and the ENTIRE old raw log byte prefixes match; first recovery37/11 receipts were retained separately. All prior charges plus this interruption's conservative wall+5CPU/wallsecond charge are carried forward; no budget/config/seed/gate change. Held3 cumulative wall{lr['wall']:.3f}/CPU{lr['cpu']:.3f} seconds at observation; original limits43200/172800. Held2 remains completed/audited/frozen UNScored and was not retrained.

New private intermediate held3 checkpoint27/57 backup:10parts/68880133bytes/328payloads, archive SHA bd9f35c5a15d3ddd12cecab355c94e7458574c9bf5a08c987b8f6431aa96628c; all remote raw parts and archive members verified, owner-only folder unchanged. No checkpoint-delta dependency, but initial source/data/maps/runtime remain required. This is not a completed-modelset/final scoring authority. Detailed evidence and persistent index: `{rp.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. No data redownload, new accuracy, users7–8 selection, or originality claim.

Next inspect PID573/current boot/attempt002 and latest checkpoints before any action; never duplicate or overwrite newer state. Continue held3–6 under the original registered all-five-fold/three-arm audit/private-backup/once-aggregate protocol. Reused source1 and repeated eight-user development remain exploratory; accuracy and originality evidence remain unresolved.

'''
report=f'''# Second boot interruption: repaired under the original registered experiment

{note}
## Failure and exact recovery evidence

The previous remote parent is `{parent}`. The old controller610/startticks311286 belongs to boot529611fa-f196-4301-8416-61ee6e8f2bc0 and was inactive on the current bootad409b48-89ce-4704-beee-3658103d5856. No surviving recorded producer or controller was found before the explicit resume. Provider restart trigger is not established; no invented OOM or successful termination is assigned to the interrupted attempt. Source/data remained accessible, so no Drive data restoration or Windows transfer was performed.

Before resume, TimeMIL committed checkpoint27/1917updates and Shuffle57/12825updates were instantiated through the original family implementations. Full native semantic model/optimizer/sampler/Torch/Python/NumPy RNG equality passed. Entire old raw logs and the prior first-recovery suffix receipts were copied before any new process began. The original native StepLog reproduced the current interruption's71/22 suffix updates exactly, preserving the complete original raw byte prefixes. `POST_RECOVERY_VERIFIED.json` checks both original committed update counts and actual suffix counts, preventing acceptance of the older37/11 receipts. This audit makes no full-training-replay claim for TimeMIL.

Actual `controller.py --resume` was launched at Unix1793769769.2376. `RESUME_LAUNCH.json` and `LIVE_OBSERVATION.json` contain PID573/startticks34322, current boot, actual two producer identities, progress and cumulative ledger. It skipped valid completed held2 only after its existing audit/fusion checks; no held2 refit or diagnostic score was computed. All60 frozen scientific files matched their registered SHA256 hashes. No experimental source, hyperparameter, seed, selection rule, mixture weight or accuracy gate changed.

## Cumulative resource accounting

The previous held3 ledger had9493.98105336113wall/41178.57776090066CPU seconds, already including the FIRST recovery charge. The read-only precheck observed an additional704.3473715782166seconds since the last old ledger and projected3521.7368578910828CPU seconds at the original conservative5CPU/wallsecond rule. The actual original controller charged the full slightly later launch interval; the precheck is not substituted for its cumulative ledger. At the latest saved observation, held3 is{lr['wall']:.6f}wall/{lr['cpu']:.6f}CPU seconds. Perfold43200wall/172800CPU and five-newfold216000wall/864000CPU limits remain unchanged; neither time nor CPU is reset. Held2's22834.174004506wall/84851.68278CPU seconds are unchanged.

## Private checkpoint insurance and restoration limits

The new archive contains328payloads (plus its member manifest),10parts and68880133bytes; SHA256 `bd9f35c5a15d3ddd12cecab355c94e7458574c9bf5a08c987b8f6431aa96628c`. Every part's metadata/privacy/parent/MIME/length was checked before raw fetch with download_raw_file=true/include_base64=false, immediately materialized and size/SHA verified, with at most3 complete concurrent pipelines. Whole archive and every member passed independent raw readback verification. No authentication URL is persisted or published. Folder12AGMwT99yXCTYBQocQaX901Br-IelmOf remains shared=false/one owner permission.

This snapshot intentionally preserves committed TimeMIL27/Shuffle57, not the later live epochs. TimeMIL snapshot has1988complete logged updates (71 beyond its1917commit); Shuffle snapshot has13044complete logged updates (219 beyond its12825commit) because capture continued after exact22-step resume. The219 capture suffix is distinct from the22 pre-resume forensic suffix. Native snapshot verification covers coherent committed model/optimizer/RNG/journal/log-prefix state and selection storage aliases; it performs no new fitting or held-label inference. Actual IDs, checkpoints, binary states and complete training logs stay in the private archive, not Git. The previous23part held2 complete-fold archive remains the completed held2 authority; it is not duplicated.

This is an intermediate recovery archive, not a completed model set or final scoring authority. It has no checkpoint-delta dependency; the original initial registration/source/data/maps/runtime dependencies are still needed. Restore into a NEW inspection directory only after confirmed loss, all old identities inactive, no newer/live state, and full metadata/raw/archive/member/native/journal/suffix verification; retain original ledgers and logs. Do not overwrite live or newer checkpoints or reset cumulative caps.

## Scientific status and next action

The fixed literature-backed uniform TimeMIL+Shuffle probability mixture remains the SAME registered source-six transfer experiment. Source1 is reused without rescoring; users7–8 labels remain excluded from selection. Held2 is completed and unscored; held3 is in progress, held4–6 still pending. Only all five new folds with all three arms, audits, frozen outputs and a full final private recoverable archive permit the once-only source-six diagnostic aggregate under the unchanged bounded gate. Existing repeated eight-user and source1 development is exploratory. No new accuracy estimate, eight-fold LTR improvement, independent validation or original research contribution is established by this operational repair.

Inspect the current boot/PID/startticks and newest progress/ledger first at the next continuation. Preserve any failure and repair within the original native method. The requested research goal is unresolved; the actual verified controller continues the authorized experiment after this turn.
'''
(result/'RUN_REPORT.md').write_text(report)
updates=dict(state='source6_second_boot_recovery_verified_held3_native_resume',latest_report=rp.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=identity['pid'],background_training_observed_unix=obs['observed_unix'],background_training_status_note='Second boot change recovered by original --resume; native states plus71/22suffix steps and entire old raw prefixes exact. PID573/current boot and cumulative ledger verified; held2 not retrained. Private intermediate held3 archive raw/allmember verified. No new diagnostic accuracy or originality claim.',compute='2026-11-04 bootad409b48-89ce-4704-beee-3658103d5856; CPU quota4/RAM16GiB; Python3.12.14 NumPy2.3.5 Torch2.15.0+cpu; actual native member processes1+4threads; no GPU')
insert_key='source6_second_boot_recovery_20261104'
insert=dict(report=updates['latest_report'],observation=rp.removeprefix('uwb_direction1/')+'/LIVE_OBSERVATION.json',private_download_index=rp.removeprefix('uwb_direction1/')+'/held3_checkpoint_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=backup['archive_sha256'],native_suffix_updates_exact={'timemil':71,'shuffle':22},whole_old_raw_prefixes_identical=True,budget_reset=False,held2_not_retrained=True,new_accuracy_result=False,original_contribution_validated=False)
authorities=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
for path in authorities:
    file=G/path;prior=file.read_text()
    if path.endswith('STATUS.json'):
        assert '"'+insert_key+'"' not in prior
        body=prior
        for k,v in updates.items():
            rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
            body,n=re.subn(rx,lambda m:m[1]+json.dumps(v,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
        body=body.rstrip();assert body.endswith('}')
        body=body[:-1]+',\n  "'+insert_key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n'
        json.loads(body)
    else:body=note+prior
    file.write_text(body)
files=[]
for path in authorities+[str(f.relative_to(G)) for f in sorted(result.rglob('*')) if f.is_file()]+[str(f.relative_to(G)) for f in sorted(code.rglob('*')) if f.is_file()]:
    raw=(G/path).read_bytes();assert not path.endswith(('.pt','.npy','.npz','.pth','.jsonl'))
    row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    if path in authorities:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
    files.append(row)
assert not subprocess.check_output(['git','diff','--name-only',parent,'--','uwb_direction1/original'],cwd=G,text=True).strip()
write(B/'PUBLICATION_MANIFEST.json',dict(root=str(G),parent=parent,files=files,note=note,status_updates=updates,status_insert_key=insert_key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2)))
print(json.dumps(dict(state='new_recovery_evidence_prepared',files=len(files),controller=obs['controller'],active_children=current,held3_epochs=dict(timemil=tm,shuffle=sh),held3_cumulative=lr,source_snapshot_unchanged=True,private_archive_bytes=backup['archive_bytes'])))
