from pathlib import Path
import sys,json,re,shutil,subprocess,hashlib
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');G=Path('/workspace/uwb-cloud-current')
sys.path.insert(0,str(ROOT));from common import read,write,sha,alive
parent='2dd67579958d1a4982c29b0bf27fa288c0499f7d'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==parent
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
observation=read(B/'LIVE_OBSERVATION.json');assert alive(observation['controller'])
assert observation['state']=='same_registered_held3_shuffle_producer_complete_timemil_still_running_unscored'
receipt=read(B/'held3_producer_backup/DRIVE_BACKUP_RECEIPT.json');manifest=read(B/'held3_producer_backup/MANIFEST.json')
assert receipt['state']=='held3_complete_shuffle100_and_running_timemil_raw_allmember_verified'
assert receipt['archive_sha256']==manifest['archive_sha256'] and not receipt['final_scoring_authority']
tm=read(B/'capture01/timemil_SNAPSHOT_VERIFIED.json');sh=read(B/'capture01/shuffle_SNAPSHOT_VERIFIED.json')
assert sh['committed_epoch']==100 and sh['committed_updates']==22500 and sh['complete_member_manifest_verified'] and sh['selected_model_matches_native_committed_state']
for v in [tm,sh]:assert v['native_instantiation_restore_recapture_exact'] and not v['full_training_replayed']
rp='uwb_direction1/results/cloud_20261104_held3_shuffle100_insurance03'
cp='uwb_direction1/research/cloud_20261104_held3_shuffle100_insurance03'
r=G/rp;c=G/cp;r.mkdir(parents=True);c.mkdir(parents=True)
for n in ['LIVE_OBSERVATION.json','CAPTURE_TRIGGER.json','SHUFFLE_ARM_COMPLETE.json','SOURCE_HASHES.json']:
    shutil.copyfile(B/n,r/n)
for arm in ['timemil','shuffle']:shutil.copyfile(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json',r/f'{arm}_SNAPSHOT_VERIFIED.json')
for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
    d=r/'held3_producer_backup'/n;d.parent.mkdir(exist_ok=True);shutil.copyfile(B/'held3_producer_backup'/n,d)
code_names=['snapshot.py','verify_native.py','capture_when_complete.py','observe.py','build_backup.py','verify_backup.py','prepare_publication.py','README.md']
for n in code_names:shutil.copyfile(B/n,c/n)
write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in code_names})
lr=observation['resource_ledger']['folds']['3'];parts=len(manifest['chunks']);arc=manifest['archive_sha256']
note=f'''## 2026-11-04 held3 ShuffleNet100 producer complete; original TimeMIL training continues

Same bootad409b48/controller573/startticks34322/attempt002, with no restart, duplicate fit, scientific-source/config/seed/gate change, diagnostic scoring or budget reset. Held3 ShuffleNet completed100epochs/22500updates and fixed final100 selection; actual producer exit0/no resource stop/no OOM. TimeMIL still trains (captured epoch{tm['committed_epoch']}/{tm['committed_updates']}updates). Both captured native families were instantiated/restored/recaptured exactly, including optimizer/sampler/all RNG state; complete Shuffle payload hashes and selected model match its native committed state. This is NOT the independent full100 training replay or selected-output audit, which remain required after both producers finish. Held2 remains complete/audited/frozen UNScored without retraining.

Private intermediate recovery archive: {parts}parts/{manifest['archive_bytes']}bytes/{manifest['tracked_payload_files']}payloads, SHA{arc}; all owner-only raw parts, whole archive and every member verified. No checkpoint-delta dependency, initial source/data/maps/runtime still required. Persistent index/details: `{rp.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. Held3 cumulative{lr['wall']:.3f}wall/{lr['cpu']:.3f}CPU seconds includes both prior interruption charges, within unchanged43200/172800caps. No new accuracy or original-contribution result; source1/eight-user development remains exploratory and users7–8 labels stay excluded.

Next inspect actual current process/boot/checkpoints first; preserve newer/live state and continue the original all-five-fold/three-arm audit/private-backup/once-aggregate protocol. Never rerun completed held2 or Shuffle producer merely to recreate progress.

'''
report=f'''# Held3 ShuffleNet producer completed and privately insured

{note}
## Verified execution milestone

Parent `{parent}` recorded held3 checkpoint27/57 recovery. The original controller and boot are still alive. The existing held3 ShuffleNet producer completed100epochs/22500updates (4500train samples, batch20), fixed final100 selection, preserved final native state and validation/diagnostic UNScored predictions. Actual attempt002 producer termination is exit0, no timeout, no resource stop and no cgroup OOM. `SHUFFLE_ARM_COMPLETE.json` lists payload hashes; `LIVE_OBSERVATION.json` records actual process identities, completion/termination evidence, current progress and cumulative resource accounting. The helper waited only for the existing producer and did not start any training or alter its protocol. Held2 remains unchanged and unscored.

## Checkpoint and artifact verification

The new snapshot coherently captures TimeMIL epoch{tm['committed_epoch']}/{tm['committed_updates']}updates with{tm['captured_complete_step_rows']}complete log rows and{tm['uncommitted_suffix_rows']}uncommitted suffix rows, and the full immutable ShuffleNet100/22500update producer. Each native family runs in a separate read-only verification process. The original model/optimizer is instantiated, checkpoint state restored (including sampler/Torch/Python/NumPy RNG), then recaptured with exact native semantic equality. Existing journal chains, slot SHA, log prefix, first-update certificates and TimeMIL selection/storage aliases pass. The completed Shuffle payload table is fully SHA checked and its fixed selected model matches the native committed100state exactly. No forward pass, optimizer update, new model fitting or diagnostic-label access is performed by these recovery checks.

The original production engines and all60 registered scientific files are unchanged. Only recovery capture/verification helpers were added or strengthened; the restore/recapture check and completed selected-model consistency checks operate on REAL captured states. This is not a substitute for the separately registered full100 Shuffle independent training replay or selected-output audits. At observation these audits had not started for held3 because TimeMIL was still training. No final-modelset or full-training-replay success is claimed.

## Private recovery scope

Archive scope `{manifest['scope']}` contains{manifest['tracked_payload_files']}payloads (plus its member manifest),{parts}parts and{manifest['archive_bytes']}bytes, SHA256 `{arc}`. Every part passed owner-only/nonshared metadata, parent/MIME/length checks before raw fetch with download_raw_file=true/include_base64=false, immediate materialization/length/SHA verification and at most3 complete concurrent pipelines. Whole archive and every member passed raw readback verification. Actual sample IDs, checkpoints, native optimizer/RNG tensors, unscored binary predictions and complete training logs are retained privately, not uploaded to Git; raw radar arrays/runtime binaries are excluded. Folder visibility remains unchanged.

The archive is intermediate insurance. It contains a complete Shuffle producer and one coherent running TimeMIL checkpoint, not a completed three-arm fold or final scoring authority. It has no checkpoint-delta dependency; original initial source/data/maps/runtime and registration dependencies remain required. Restore only into a NEW inspection directory after confirmed loss, old process identities inactive, no newer/live state, metadata/raw/archive/member/native verification and remaining cumulative-budget checks. Preserve old/full raw logs, original suffixes and all previous interruption charges; never reset caps or overwrite newer/live checkpoints. The completed held2 archive and previous recovery evidence remain valid and are not duplicated.

## Scientific limits and next steps

Observed held3 wall{lr['wall']:.6f}/CPU{lr['cpu']:.6f}seconds includes both previous conservative interruption charges. Perfold43200wall/172800CPU and five-fold216000wall/864000CPU caps remain frozen. One actual TimeMIL worker continues, with the same controller. After both producers finish, the controller runs the original independent Shuffle replay, selected-output audits and fixed three-arm fusion before moving to held4–6. All five new folds, frozen outputs and a FULL final privately recoverable archive are still prerequisites for once-only source-six diagnostic aggregation. Source1 is reused without rescoring; users7–8 labels remain excluded from selection. Repeated source1/eight-user development remains exploratory. This milestone establishes completed producer execution and reproducible recovery, no new accuracy estimate, eight-fold improvement, independent validation or original research contribution.
'''
(r/'RUN_REPORT.md').write_text(report)
updates=dict(state='source6_held3_shuffle100_producer_complete_timemil_running_unscored',latest_report=rp.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=observation['controller']['pid'],background_training_observed_unix=observation['observed_unix'],background_training_status_note='Same original PID573/bootad409b48. Held3 Shuffle100/22500updates fixed final100 complete with actual exit0; TimeMIL continues. Native restore/recapture, complete Shuffle selected-state hashes and private raw/allmember checkpoint archive verified. Full independent replay and output audits remain pending; no diagnostic accuracy or originality claim.')
key='source6_held3_shuffle100_producer_insurance03_20261104'
insert=dict(report=updates['latest_report'],observation=rp.removeprefix('uwb_direction1/')+'/LIVE_OBSERVATION.json',private_download_index=rp.removeprefix('uwb_direction1/')+'/held3_producer_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,captured_epochs={'timemil':tm['committed_epoch'],'shuffle':100},shuffle_updates=22500,shuffle_selected_epoch=100,native_restore_recapture_exact=True,full_independent_training_replay_complete=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
authorities=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
for path in authorities:
    f=G/path;prior=f.read_text()
    if path.endswith('STATUS.json'):
        assert '"'+key+'"' not in prior
        body=prior
        for k,v in updates.items():
            rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
            body,n=re.subn(rx,lambda m:m[1]+json.dumps(v,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
        body=body.rstrip();assert body.endswith('}')
        body=body[:-1]+',\n  "'+key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n';json.loads(body)
    else:body=note+prior
    f.write_text(body)
assert not subprocess.check_output(['git','diff','--name-only',parent,'--','uwb_direction1/original'],cwd=G,text=True).strip()
files=[]
for path in authorities+[str(x.relative_to(G)) for x in sorted(r.rglob('*')) if x.is_file()]+[str(x.relative_to(G)) for x in sorted(c.rglob('*')) if x.is_file()]:
    raw=(G/path).read_bytes();assert not path.endswith(('.pt','.pth','.npz','.npy','.bin','.jsonl'))
    row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    if path in authorities:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
    files.append(row)
write(B/'PUBLICATION_MANIFEST.json',dict(root=str(G),parent=parent,files=files,note=note,status_updates=updates,status_insert_key=key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2)))
print(json.dumps(dict(state='held3_shuffle100_insurance_evidence_prepared',files=len(files),archive_sha256=arc,captured_epochs=insert['captured_epochs'],controller_alive=True,new_accuracy_result=False,source_snapshot_unchanged=True)))
