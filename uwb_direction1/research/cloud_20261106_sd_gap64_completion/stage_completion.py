"""Stage actual SD completion/private proof or the original consumed once gate; never fit."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, shutil
B=Path('/workspace/uwb-sd-gap64-20261105');M=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from common import read,write,sha,ready
from controller import identity
from stream_archive import verify
SCORER='9321449c2a3924b1059fc0c051352eb88d06497a3ee393a76076f75188b9f528'
def main():
 mode,parent=sys.argv[1:];assert mode in ['full_archive','final'];ready(True)
 a=read(M/'PILOT_AUDIT.json');d=B/'full_backup';m=read(d/'MANIFEST.json');v=read(d/'READBACK_VERIFIED.json')
 assert v['complete_modelset_verified'] and verify(m,read(d/'READBACK_INPUT.json'))==v
 assert a['producer_epochs']==a['replay_epochs']==64 and a['epoch_records_exact']==65 and a['updates_per_trajectory']==4736
 assert sha(B/'operator/finalize_once.py')==a['original_once_scorer_sha256']==read(B/'REAL_RELEASE.json')['once_scorer_sha256']==SCORER
 assert read(M/'OBSERVER_TERMINATION.json')['exit_code']==read(M/'FULL_ARCHIVE_BUILD_TERMINATION.json')['exit_code']==0
 for n in ['CONTROLLER_IDENTITY.json','run/execution/train/IDENTITY.json','run/execution/replay/IDENTITY.json','run/execution/output/IDENTITY.json']:
  old=read(B/n);now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['pid','startticks','boot_id'])
 members=read(d/'MEMBERS.json')['members'];keys={r['path'] for r in members}
 assert any(p.startswith('initial_backup/') and '.part' in p for p in keys)
 inventory=dict(state='full_producer64_OWNfresh64_native_all_RNG_all4736steps_all65records_selected_outputs_verified',producer_epochs=64,replay_epochs=64,updates_each=4736,selected_epoch=a['selected_epoch'],canonical_native_terminal_sha256=a['native_final_sha256'],tracked_payloads=m['tracked_payload_files'],initial_native_state_copies=[r for r in members if r['path'].endswith('/initial.pt')],initial_raw_guard_chunks=[p for p in sorted(keys) if p.startswith('initial_backup/') and '.part' in p],full_private_allmember_verified=True,raw_radar_RMS_runtime_external=True)
 write(M/'MODELSET_INVENTORY.json',inventory)
 for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
  p=M/'full_backup'/n;p.parent.mkdir(exist_ok=True);shutil.copyfile(d/n,p)
 shutil.copyfile(B/'STATE.json',M/('POST_SCORE_STATE.json' if mode=='final' else 'PRE_SCORE_STATE.json'))
 (M/'FULL_RESTORE_GUIDE.md').write_text('''# Complete fixed DomainBed SD pilot recovery

FULL preserves the original producer64 and independently fresh OWN64 replay, all4736 steps and65 epoch records per trajectory, complete native model/optimizer/BN/Python-NumPy-Torch-order RNG, validation-selected model, frozen unscored outputs, initial raw guard chunks, native49, ten namespaced author references and license notices, source/operators, private IDs/roles and consumed prior EFDMix summary/state/lock. Raw radar/RMS/runtime external. Reuse latest original raw-data indexes without manual uploads.

Metadata owner-only first; raw(download_raw_file=true,include_base64=false), immediate size/SHA, whole archive and every member, max3 complete pipelines. Known IDs reused. Unknown ID failure requires successful exactfolder AND global exactname zero/no-morepage proofs plus durable marker before at most one explicit retry; any match/searchfailure/priorretry stops. No authenticated URL output.

FULL is PRE-SCORE. Preserve newer SUMMARY/postscore STATE and consumed FINALIZE_ONCE.lock before restoring; never rescore a consumed gate. Original immutable scorer SHA9321449c2a3924b1059fc0c051352eb88d06497a3ee393a76076f75188b9f528. Fixed SDsd_reg.1/noanneal/uncenteredmean_B12logitsquared/native12OvRBCE/full617612GAP/B64/64epochs/seed20261029/AdamW.001_decay.0001_Lookahead.5_k6/clip2. Fixed once source1 gate .9841666666666667+.005/tol1e-12; failure closes exact configuration without coefficient/anneal/decay/layer/seed/epoch rescue. Positive requires separately registered same-protocol source6 confirmation. Users7–8 labels forbidden. Repeated source/eight-user development exploratory; accuracy and originality evidence separate. No independent/original/legacy8fold improvement claim. Original789, repository visibility and Drive privacy preserved.
''')
 t=a['actual_terminations']['output']['cumulative']
 common=f"Fixed published DomainBed SD sd_reg.1/noanneal/uncenteredmean_B12logitsquared/nativehard12OvRBCE/full617612GAP/rawforwardB64/64epochs/seed20261029 completed ORIGINAL64 plus OWN independently fresh64, every4736step/65epoch record/native model-optimizer-allRNG-BN/validation-selected epoch{a['selected_epoch']} outputs exact. Actual train/replay/output exits0, canonical terminal{a['native_final_sha256']}, unchanged20science{a['scientific_manifest_sha256']}. Cumulative{t['wall']:.6f}wall/{t['cpu']:.6f}CPU seconds under43200wall/CPU/12GiB; no duplicate fitting/restart/budget reset. Published component transfer, not complete original image/DomainBed benchmark or gradient-starvation/originality proof.\n\nFULL owner-only private modelset {len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads SHA{m['archive_sha256']}, membersSHA{m['members_manifest_sha256']}. Metadata/raw(download_raw_file=true,include_base64=false)/immediate sizeSHA/whole/everymember verified, max3complete pipelines. Initial guard rawchunks/native49/ten namespaced referencepaths/license notices/privateIDs/roles/logs/operators and consumed priorEFDMix lineage included; raw radar/RMS/runtime external. Persistentindex results/cloud_20261106_sd_gap64_{mode}/full_backup/CLOUD_DOWNLOAD_INDEX.json. PRE-SCORE archive: preserve newer summary/STATE/consumedlock before recovery, NEVER rescore.\n\n"
 files=['PILOT_AUDIT.json','OBSERVER_TERMINATION.json','FULL_ARCHIVE_BUILD_TERMINATION.json','MODELSET_INVENTORY.json','FULL_RESTORE_GUIDE.md']+['full_backup/'+n for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']]
 ins=dict(producer_epochs=64,replay_epochs=64,updates_per_trajectory=4736,epoch_records_exact=65,selected_epoch=a['selected_epoch'],native_final_sha256=a['native_final_sha256'],actual_terminations=a['actual_terminations'],archive_parts=len(m['chunks']),archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],archive_payloads=m['tracked_payload_files'],complete_private_modelset_owner_only_raw_every_member_verified=True,original_once_scorer_sha256=SCORER,scientific20_changed=False,original789_changed=False)
 scalar=dict(last_completed_update_utc=datetime.now(timezone.utc).isoformat())
 if mode=='full_archive':
  assert not(B/'SUMMARY.json').exists() and not(B/'FINALIZE_ONCE.lock').exists();files.append('PRE_SCORE_STATE.json')
  action='Execute ORIGINAL immutable release-bound SD once scorer once after this exact FULL private proof publication. No active SD fitting. Fixed +.005 source1 gate; failure closes exact sdreg.1/noanneal/GAP64/B64/64/seed20261029 without rescue then continue next literature-backed method; positive separately register same-protocol source6 confirmation.'
  note='## 2026-11-06: SD full64/OWNfresh64 and FULL private modelset verified before ORIGINAL once gate\n\n'+common+action+'\n\nDiagnostic UNSCORED. Users7–8 labels forbidden; repeated development exploratory. Accuracy and originality separate/unresolved; goal not achieved, automatic continuation continues. Cloud data reused without Windows; original789/visibility/Driveprivacy unchanged.\n\n'
  state='SD_full64_OWNfresh64_FULL_private_modelset_verified_ready_for_original_once_gate';ins.update(diagnostic_labels_read=False,once_gate_consumed=False,new_accuracy_result=False)
 else:
  q=read(B/'SUMMARY.json');st=read(B/'STATE.json');assert q['finalizer_invocations']==1 and st['finalizer_consumed'] and (B/'FINALIZE_ONCE.lock').exists() and st['summary_sha256']==sha(B/'SUMMARY.json')
  assert read(M/'SCORING_TERMINATION.json')['exit_code']==0
  shutil.copyfile(B/'SUMMARY.json',M/'SUMMARY.json');write(M/'ONCE_GATE_CONSUMED.json',dict(summary_sha256=sha(B/'SUMMARY.json'),original_once_scorer_sha256=SCORER,finalizer_invocations=1,exclusive_consumed_lock_exists=True,preserve_before_pre_score_restore=True))
  files+=['SUMMARY.json','POST_SCORE_STATE.json','ONCE_GATE_CONSUMED.json','SCORING_TERMINATION.json']
  action='Separately register SAME fixed SD same-protocol source6 confirmation before further held-user outcomes.' if q['predeclared_gate_passed'] else 'Exact SD sdreg.1/noanneal/full617612GAP/B64/64epochs/seed20261029 closed without coefficient/anneal/decay/layer/epoch/seed rescue. Continue next separately fixed, literature-backed method; no diagnostic sweep.'
  note='## 2026-11-06: Original SD once diagnostic completed; exact registered route retained\n\n'+common+f"ORIGINAL release-bound scorer executed exactly once after FULLprivateproof, actualexit0. Exploratorysource1 macro{q['diagnostic_macro']*100:.9f}% vs matchednoMixupGAP{q['reference_noMixup_GAP64_macro']*100:.9f}%, delta{q['difference']*100:+.9f}percentagepoints. Fixed +.5point/tol1e-12 gate={q['predeclared_gate_passed']}, validationonlyselected epoch{q['selected_epoch']}. Route={q['route']}. {action}\n\nScorer consumed; preserve summary/postscoreSTATE/lock, NEVER rescore. Users7–8 labels excluded; repeated source/eight-user development exploratory, accuracy/originality evidence separate. No independent-validation/original-contribution/legacy8fold conclusion. Goal unresolved/automatic continuation continues; no active SD fitting. Original789/visibility/Driveprivacy unchanged, cloud-only.\n\n"
  state=q['route'];ins.update(diagnostic_labels_read=True,once_gate_consumed=True,new_accuracy_result=True,summary=q);scalar.update(latest_source_diagnostic='results/cloud_20261106_sd_gap64_final/SUMMARY.json')
 if (M/'OPERATIONAL_STAGING_REPAIR.json').exists(): files.append('OPERATIONAL_STAGING_REPAIR.json')
 scalar['next_candidate']=action
 write(M/(mode+'_SPEC.json'),dict(parent=parent,stage=mode,note=note,include_source=False,metadata_files=files,state=state,running=False,pid=None,status_note=action,scalar_updates=scalar,status_insert=ins));print(json.dumps(dict(stage=mode,metadata_files=len(files),parts=len(m['chunks']),archive_sha256=m['archive_sha256'])))
if __name__=='__main__':main()
