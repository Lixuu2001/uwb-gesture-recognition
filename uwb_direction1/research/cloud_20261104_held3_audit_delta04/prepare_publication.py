from pathlib import Path
import sys,json,re,hashlib,shutil,subprocess
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');G=Path('/workspace/uwb-cloud-current');sys.path.insert(0,str(ROOT))
from common import read,write,sha,alive
parent='0fbca232c7e3fd154a23bde210c1ad35024f9949'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==parent
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
obs=read(B/'LIVE_OBSERVATION.json');assert alive(obs['controller'])
m=read(B/'delta04_backup/MANIFEST.json');receipt=read(B/'delta04_backup/DRIVE_BACKUP_RECEIPT.json')
assert receipt['state']=='held3_complete_producers_own_replay_checkpoint_delta_raw_allmember_verified' and receipt['archive_sha256']==m['archive_sha256']
assert m['delta_dependency'] and m['chain_depth']==1 and not m['complete_modelset']
tm=read(B/'reconstructed01/timemil_SNAPSHOT_VERIFIED.json');sh=read(B/'reconstructed01/shuffle_SNAPSHOT_VERIFIED.json')
assert tm['retained_output_audit_verified'] and sh['own_checkpoint_native_restore_recapture_exact'] and sh['own_replay_prefix_of_producer_steps_verified']
assert sh['own_replay_epoch']==7 and sh['own_epoch_records_exact']==8
rp='uwb_direction1/results/cloud_20261104_held3_audit_delta04';cp='uwb_direction1/research/cloud_20261104_held3_audit_delta04'
r=G/rp;c=G/cp;r.mkdir(parents=True);c.mkdir(parents=True)
for n in ['LIVE_OBSERVATION.json','LOCAL_RECONSTRUCTION.json','BASE_VERIFIED.json','SOURCE_HASHES.json']:
    shutil.copyfile(B/n,r/n)
for role in ['capture01','reconstructed01']:
    for arm in ['timemil','shuffle']:shutil.copyfile(B/role/f'{arm}_SNAPSHOT_VERIFIED.json',r/f'{role}_{arm}_SNAPSHOT_VERIFIED.json')
for n in ['DELTA.json','CURRENT_PAYLOAD_HASHES.json','BASE_ARCHIVE_INDEX.json']:
    d=r/'delta_payload'/n;d.parent.mkdir(exist_ok=True);shutil.copyfile(B/'delta_payload'/n,d)
for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
    d=r/'delta04_backup'/n;d.parent.mkdir(exist_ok=True);shutil.copyfile(B/'delta04_backup'/n,d)
names=['snapshot.py','augment_capture.py','verify_native.py','delta.py','build_delta.py','build_backup.py','verify_backup.py','observe.py','prepare_publication.py','README.md']
for n in names:shutil.copyfile(B/n,c/n)
write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in names})
current=obs['current_own_replay']['epoch'];lr=obs['resource_ledger']['folds']['3'];arc=m['archive_sha256'];parts=len(m['chunks'])
note=f'''## 2026-11-04 held3 both producers complete; TimeMIL output audit passed; own Shuffle replay insured

Same bootad409b48/controller573/startticks34322/attempt002, no restart or scientific/config/seed/gate change. Held3 TimeMIL64/4544updates selectedepoch50 and Shuffle100/22500updates fixedfinal100 both completed with actual exit0/no resource stop/no OOM. TimeMIL selected-state and ALL validation/unscored-diagnostic logits audit passed with actual successful termination; NOT full64 training replay. Independent Shuffle full100 replay is actually running as PID1754 (epoch{current} at observation), not complete. Held2 remains complete/audited/frozen UNScored without refitting; no diagnostic score or new accuracy/originality claim.

New private delta04 preserves both complete producers, the TimeMIL certificate and auditor OWN7/1575updates. Real new-directory reconstruction of all581current payloads (86changed) passed both native model/optimizer/all RNG/selection checks, own replay state restore/recapture,8exact epoch records and original-step-log prefix. Owner-only raw delta: {parts}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads, SHA{arc}. Dependency depth1: producer insurance03 SHA c77fe20741e4e2cdde385c163a30ae44708379f8bc4bd0dd80422c495f6c3906 -> delta04; BASE_ARCHIVE_INDEX.json/guide included. This is intermediate audit insurance, not final scoring authority. Initial source/data/maps/runtime still required; never restore over newer/live state or reset cumulative caps.

Held3 cumulative{lr['wall']:.3f}wall/{lr['cpu']:.3f}CPU seconds retains both prior interruption charges, within original43200/172800caps. Details/index: `{rp.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. Next inspect actual current boot/PID1754/checkpoints first, continue full100 replay/output audits/3arm freeze; then preserve a complete-fold archive to end the delta dependency. All5newfolds/full final private recovery remain prerequisites for once aggregate. Source1reused/users7–8sealed/exploratory repeated-user caveat unchanged.

'''
report=f'''# Held3 completed producers, passed TimeMIL output audit, independent Shuffle replay in progress

{note}
## Execution and scientific scope

Remote parent `{parent}` had privately preserved TimeMIL48/Shuffle100. The same original controller continues on the same boot; no new fit or restart was launched by this work. Both held3 producers are now complete: TimeMIL64/4544logical updates, selectedepoch50 using inner validation, and Shuffle100/22500logical updates with fixed final100 selection. Both actual attempt002 terminations are exit0, no timeout, no resource stop and no cgroup OOM. These per-attempt lower-bound timings are not substituted for the cumulative whole-fold ledger, which retains all prior interruptions and exact registered accounting.

The original TimeMIL selected-model audit verified its complete native optimizer/RNG/journal/model-selection state and reproduced ALL validation and unscored diagnostic logits exactly, with actual exit0. Its certificate explicitly does NOT claim full64 training replay. The generic controller has entered the original independent Shuffle full100 replay from a fresh seed; actual PID1754/startticks801143 on bootad409b48 is active. It resumes only its own independent checkpoint if interrupted. At observation this audit was at epoch{current}, while the insured coherent own checkpoint is epoch7/1575updates. Full100 success, subsequent Shuffle selected-output audit and fixed three-arm fusion are still pending. No held diagnostic label or score was read by this continuation.

## Recovery implementation and real reconstruction checks

The coherent snapshot contains both complete immutable producer manifests, first-update proofs, selected/final native states, actual IDs, unscored predictions, complete original training logs and the auditor's own committed journal/checkpoint. The actual TimeMIL inference certificate and successful termination were added explicitly because they were created after its producer manifest. Every certificate artifact digest matches its captured payload. All60 registered scientific files remain SHA-identical; only recovery helpers were added or strengthened.

Both original native model/optimizer families were independently instantiated, restored and recaptured with exact semantic equality of model/optimizer/sampler/Torch/Python/NumPy RNG state. Both selected models match their native committed selection states; every producer payload and journal/log prefix passes. TimeMIL best-selection key/storage aliases pass. The own Shuffle auditor state separately passes original native restore/recapture; its8records from epoch0 through7 match producer records exactly, its initial record matches the registered seed state, and its1575-update log digest is the corresponding exact prefix of the immutable22500-row producer log. These checks perform no training updates or inference and are not reported as a new full100 replay.

The unchanged content-checked delta utility compared the new payload table against the previously raw-verified producer insurance03 base, copied86changed paths and reconstructed ALL581current payloads into a NEW directory without mutating the base. Both native verifiers then passed on that real reconstruction with the same substantive state/checkpoint hashes as the source capture. Current-table, base-table, base member-manifest and full raw archive identities were checked. Generic delta code is byte-identical to the earlier tested utility; no old corruption-test result is presented as a new experiment.

## Private archive, dependency and restoration

New delta archive: {parts}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads (plus member manifest), SHA256 `{arc}`. Every part passed owner-only/private parent/MIME/length metadata checks before download_raw_file=true/include_base64=false fetch, immediate materialization and length/SHA verification, at most3 complete concurrent pipelines. Whole archive and every member passed raw readback verification. No authenticated URL is output or committed. Actual IDs, tensor models/optimizer/RNG state, binary predictions and full logs stay private; Git contains code, manifests/digests, compact observations, verification receipts and persistent indexes.

This delta has dependency depth1. Restore producer insurance03 (archive SHA c77fe20741e4e2cdde385c163a30ae44708379f8bc4bd0dd80422c495f6c3906) into a separate inspection area, raw-verify it and identify its overlay/ subtree. Its persistent index is `uwb_direction1/results/cloud_20261104_held3_shuffle100_insurance03/held3_producer_backup/CLOUD_DOWNLOAD_INDEX.json`; the same BASE_ARCHIVE_INDEX.json is embedded here. Raw-verify delta04 and apply its delta_payload/ over the exact base overlay into another NEW output directory, then rerun native/own-audit/certificate checks. Retain the base and all prior forensic logs/omitted paths. Initial source/seed/ID/data/maps/runtime and original preregistration are still required.

Do not restore over live/newer state. Only confirmed loss and inactive old identities permit the original registered controller --resume from verified reconstructed state, with the auditor's OWN checkpoint, all cumulative caps and prior conservative interruption charges preserved. Do not refit completed held2 or either held3 producer. This intermediate archive is not a completed three-arm fold/model set or final scoring authority. Once the current full audit/output audit/fusion finish, preserve a complete-fold archive to end the delta dependency rather than extending it unnecessarily.

## Remaining bounded experiment

Held3 cumulative cost at observation is{lr['wall']:.6f}wall/{lr['cpu']:.6f}CPU seconds, within unchanged43200wall/172800CPU per-fold and216000wall/864000CPU five-newfold limits. All prior charges remain; no budget reset or rescue configuration is introduced. After held3 full audits/freeze, the same controller must continue held4–6. Only all five new folds, all three arms/audits/frozen outputs and full final private recoverability permit once-only source-six diagnostic aggregation under the already registered gates. Source1 is reused without rescoring; users7–8 labels remain excluded from new-method selection. Repeated source1/eight-user development is exploratory. Accuracy improvement, eight-fold generalization and original contribution remain unestablished by this operational/model-completion milestone.
'''
(r/'RUN_REPORT.md').write_text(report)
updates=dict(state='source6_held3_both_producers_complete_TimeMIL_output_audit_pass_Shuffle_replay_running',latest_report=rp.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=obs['controller']['pid'],background_training_observed_unix=obs['observed_unix'],background_training_status_note='Same controller573/bootad409b48. Both held3 producers complete with actual exit0; TimeMIL selected-output audit passed. Actual independent Shuffle100 replayPID1754 ongoing. Both complete models and own7 checkpoint reconstructed/native-verified through raw private delta04 over producer insurance03, dependency depth1. No new diagnostic accuracy/originality claim or budget/scientific change.',compute='2026-11-04 bootad409b48-89ce-4704-beee-3658103d5856; CPU quota4/RAM16GiB; Python3.12.14 NumPy2.3.5 Torch2.15.0+cpu; both held3 producers complete, actual Shuffle independent replay uses4threads; no GPU')
key='source6_held3_audit_delta04_20261104'
insert=dict(report=updates['latest_report'],observation=rp.removeprefix('uwb_direction1/')+'/LIVE_OBSERVATION.json',private_download_index=rp.removeprefix('uwb_direction1/')+'/delta04_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,base_archive_sha256=m['base_archive_sha256'],delta_chain_depth=1,current_payloads=581,changed_payloads=86,completed_producer_epochs={'timemil':64,'shuffle':100},captured_own_shuffle_replay_epoch=7,full100_shuffle_replay_complete=False,TimeMIL_output_audit_passed=True,TimeMIL_full64_training_replayed=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
authorities=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
for path in authorities:
    f=G/path;prior=f.read_text()
    if path.endswith('STATUS.json'):
        assert '"'+key+'"' not in prior;body=prior
        for k,v in updates.items():
            rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
            body,n=re.subn(rx,lambda m:m[1]+json.dumps(v,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
        body=body.rstrip();assert body.endswith('}')
        body=body[:-1]+',\n  "'+key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n';json.loads(body)
    else:body=note+prior
    f.write_text(body)
assert not subprocess.check_output(['git','diff','--name-only',parent,'--','uwb_direction1/original'],cwd=G,text=True).strip()
files=[]
for path in authorities+[str(f.relative_to(G)) for f in sorted(r.rglob('*')) if f.is_file()]+[str(f.relative_to(G)) for f in sorted(c.rglob('*')) if f.is_file()]:
    raw=(G/path).read_bytes();assert not path.endswith(('.pt','.pth','.npz','.npy','.bin','.jsonl'))
    row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    if path in authorities:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
    files.append(row)
write(B/'PUBLICATION_MANIFEST.json',dict(root=str(G),parent=parent,files=files,note=note,status_updates=updates,status_insert_key=key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2)))
print(json.dumps(dict(state='held3_complete_producers_and_own_replay_insurance_evidence_prepared',files=len(files),archive_sha256=arc,dependency_depth=1,controller_alive=True,new_accuracy_result=False,original_snapshot_unchanged=True)))
