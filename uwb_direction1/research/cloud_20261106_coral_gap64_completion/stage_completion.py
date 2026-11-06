"""Stage actual audited CORAL FULL private modelset or original consumed once result."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,shutil
S=Path('/workspace/uwb-coral-gap64-20261106');M=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import ready,read,write,sha
from controller import identity
from stream_archive import verify
SCORER='085f666b3561488b281e172fde504a8a6b762fe46c2fcb81fd956f850e14d697'
CONFIG='CORALgamma1/lastfull128dim/pooledB64sourceusers3–6/pairmean+unbiasedcovariance_squaredmean/validgroups>=2/full617612GAP/64epochs/seed20261029/nativehard12OvRBCE/AdamW.001_decay.0001_Lookahead.5_k6/clip2'
def main():
 stage,parent=sys.argv[1:];assert stage in ['full_archive','final'];ready(True)
 a=read(M/'PILOT_AUDIT.json');d=S/'full_backup';m=read(d/'MANIFEST.json');v=read(d/'READBACK_VERIFIED.json')
 assert m['complete_modelset'] and v['complete_modelset_verified'] and verify(m,read(d/'READBACK_INPUT.json'))==v
 assert a['producer_epochs']==a['replay_epochs']==64 and a['epoch_records_exact']==65 and a['updates_per_trajectory']==4736
 assert sha(S/'operator/finalize_once.py')==a['original_once_scorer_sha256']==read(S/'REAL_RELEASE.json')['once_scorer_sha256']==SCORER
 assert read(M/'OBSERVER_TERMINATION.json')['exit_code']==read(M/'FULL_ARCHIVE_BUILD_TERMINATION.json')['exit_code']==0
 for n in ['CONTROLLER_IDENTITY.json','run/execution/train/IDENTITY.json','run/execution/replay/IDENTITY.json','run/execution/output/IDENTITY.json']:
  old=read(S/n);now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['pid','startticks','boot_id'])
 members=read(d/'MEMBERS.json')['members'];keys={r['path'] for r in members};assert any(p.startswith('initial_backup/') and '.part' in p for p in keys)
 inventory=dict(state='complete_original64_OWNfresh64_native_allRNG_all4736steps_all65records_selected_outputs_verified',producer_epochs=64,replay_epochs=64,updates_each=4736,selected_epoch=a['selected_epoch'],canonical_native_terminal_sha256=a['native_final_sha256'],payloads=m['tracked_payload_files'],initial_native_copies=[r for r in members if r['path'].endswith('/initial.pt')],initial_raw_guards=[p for p in sorted(keys) if p.startswith('initial_backup/') and '.part' in p],raw_radar_RMS_runtime_external=True)
 write(M/'MODELSET_INVENTORY.json',inventory)
 for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
  p=M/'full_backup'/n;p.parent.mkdir(exist_ok=True);shutil.copyfile(d/n,p)
 shutil.copyfile(S/'STATE.json',M/('POST_SCORE_STATE.json' if stage=='final' else 'PRE_SCORE_STATE.json'))
 (M/'FULL_RESTORE_GUIDE.md').write_text('# Complete fixed CORAL pilot recovery\n\n'+CONFIG+''' fixed20science. Original64 and independently freshOWN64, all65records/all4736steps/native model-optimizer-BN-Python-NumPy-Torch-orderRNG/validationselected model/frozenunscored outputs, actualinitialrawguards/native49/allfive namespacedrefs/licenses/privateIDs/roles/operators/consumedpriorFocal summary-state-lock included. Rawradar/RMS/runtime external via existing durable indexes, never manual uploads. Use latest four Git authorities and source layout, never overwrite healthy or newer state.

Owner-onlymetadata first, raw(download_raw_file=true,include_base64=false), immediate size/SHA/whole/everymember, max3completepipelines. Reuse knownIDs. Unknown failure requires successful exactfolder AND globalexactname zero/no-morepage proofs and durable marker BEFORE at mostone retry; match/searchfailure/priorretry stops. No authURLs.

FULL is PRE-SCORE. Preserve newer SUMMARY/postscoreSTATE/FINALIZE_ONCE.lock before recovery and NEVER rescore consumed gates. Immutable originalCORALscorer085f666b3561488b281e172fde504a8a6b762fe46c2fcb81fd956f850e14d697. Fixed source1 gate .9841666666666667+.005/tol1e-12; exactfailure closes gamma1/last128/pooledB64/GAP64/64/seed20261029 without coefficient/layer/domainbatch/grouprule/epoch/seed rescue; positive only separately registered SAMEprotocolsource6 confirmation. Users7–8 labels forbidden; repeated development exploratory, accuracy/originality evidence separate. Component transfer with native pooledsingleBN/BCE/order, not full per-domain authorimagebenchmark/originalUDA/cov-only scaling; fullprimarypapertext not reviewed, no alignmentinnovation claim. Original789/visibility/Driveprivacy unchanged.
''')
 total=a['actual_terminations']['output']['cumulative']
 common=f"Fixed published {CONFIG} completed original64 AND OWNindependentlyfresh64, all4736steps/65epochrecords/native model-optimizer-allRNG-BN/validation-selected epoch{a['selected_epoch']} outputs exact. Actualtrain/replay/outputexit0, canonical terminal{a['native_final_sha256']}, all20science unchanged{a['scientific_manifest_sha256']}. Cumulative{total['wall']:.6f}wall/{total['cpu']:.6f}CPU seconds under43200wall/CPU/12GiB, no duplicate/restart/budgetreset. Published DomainBed component transfer, nativepooledforward/BN/globalBCE/order vs authorper-domain protocol; not full imagebenchmark or originalUDA/cov-only scaling/novelalignment/fullprimarypaperreview claim.\n\nFULL owner-only private modelset{len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads SHA{m['archive_sha256']}, membersSHA{m['members_manifest_sha256']}: metadata/raw(download_raw_file=true,include_base64=false)/immediate sizeSHA/whole/everymember verified,max3completepipelines. Actualinitialrawguards/native49/allfive namespacedrefs/licenses/privateIDs/roles/operators/consumedpriorFocal lineage included; rawradar/RMS/runtime external. No private arrays/models/actualIDs inGit. Persistentindex results/cloud_20261106_coral_gap64_{stage}/full_backup/CLOUD_DOWNLOAD_INDEX.json. PRE-SCORE archive: preserve newestsummary/STATE/consumedlock before recovery, NEVER rescore.\n\n"
 files=['PILOT_AUDIT.json','OBSERVER_TERMINATION.json','FULL_ARCHIVE_BUILD_TERMINATION.json','MODELSET_INVENTORY.json','FULL_RESTORE_GUIDE.md']+['full_backup/'+n for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']]
 ins=dict(producer_epochs=64,replay_epochs=64,epoch_records_exact=65,updates_per_trajectory=4736,selected_epoch=a['selected_epoch'],native_final_sha256=a['native_final_sha256'],actual_terminations=a['actual_terminations'],archive_parts=len(m['chunks']),archive_bytes=m['archive_bytes'],archive_payloads=m['tracked_payload_files'],archive_sha256=m['archive_sha256'],complete_private_owner_only_raw_every_member_verified=True,original_once_scorer_sha256=SCORER,scientific20_unchanged=True)
 scalar=dict(last_completed_update_utc=datetime.now(timezone.utc).isoformat())
 if stage=='full_archive':
  assert not(S/'SUMMARY.json').exists() and not(S/'FINALIZE_ONCE.lock').exists();files.append('PRE_SCORE_STATE.json')
  action='Invoke original immutable release-bound CORAL once scorer exactly once after this exactFULLprivateproof publication. Exactfailure closes gamma1/last128/pooledB64/GAP64/64/seed20261029 without rescue then proceed next separately fixed literature-backed method; positive separatelyregister SAMEprotocolsource6 confirmation.'
  note='## 2026-11-06: CORAL original64/OWNfresh64 and complete private modelset verified before ORIGINALonce\n\n'+common+action+'\n\nDiagnostic remains UNSCORED; no active CORALmodel. Users7–8excluded, repeated development exploratory, accuracy/originality separate/unresolved. No newaccuracyresult or independent/original/legacy8fold claim. Goal unresolved/automaticcontinuation continues, cloud-only/original789/privacy unchanged.\n\n'
  state='CORAL_original64_OWNfresh64_FULL_private_modelset_verified_ready_for_original_once';ins.update(diagnostic_labels_read=False,once_gate_consumed=False,new_accuracy_result=False)
 else:
  q=read(S/'SUMMARY.json');st=read(S/'STATE.json');assert q['finalizer_invocations']==1 and st['finalizer_consumed'] and (S/'FINALIZE_ONCE.lock').exists() and st['summary_sha256']==sha(S/'SUMMARY.json') and read(M/'SCORING_TERMINATION.json')['exit_code']==0
  shutil.copyfile(S/'SUMMARY.json',M/'SUMMARY.json');write(M/'ONCE_GATE_CONSUMED.json',dict(state='original_once_consumed_preserve_before_pre_score_restore',summary_sha256=sha(S/'SUMMARY.json'),original_once_scorer_sha256=SCORER,finalizer_invocations=1,exclusive_consumed_lock_exists=True))
  files+=['SUMMARY.json','POST_SCORE_STATE.json','ONCE_GATE_CONSUMED.json','SCORING_TERMINATION.json']
  action='Separately preregister SAME fixed CORAL same-protocol source6 confirmation before any additional held-user outcomes.' if q['predeclared_gate_passed'] else 'Close exact CORALgamma1/last128/pooledB64/fullGAP64/B64/64epochs/seed20261029 without coefficient/layer/domainbatch/grouprule/epoch/seed rescue; proceed next separately fixed literature-backed method, no diagnostic sweep.'
  note='## 2026-11-06: Original CORAL once diagnostic completed; exact registered route retained\n\n'+common+f"ORIGINAL release-bound scorer executed exactly once after FULLprivateproof, actualexit0. Exploratorysource1 macro{q['diagnostic_macro']*100:.9f}% vs matchednoMixupGAP{q['reference_noMixup_GAP64_macro']*100:.9f}%, delta{q['difference']*100:+.9f}percentagepoints. Fixed+.5point/tol1e-12gate={q['predeclared_gate_passed']}, epoch{q['selected_epoch']} chosenbyvalidationonly. Route={q['route']}. {action}\n\nScorerconsumed; preserve summary/postscoreSTATE/consumedlock, NEVER rescore. Users7–8labels excluded, repeatedsource/eight-userdevelopment exploratory. Accuracy and originality evidence separate; no independent-validation/original-contribution/legacy8fold claim. Goal unresolved/automaticcontinuation continues; no activeCORALfit. Original789/visibility/Driveprivacy unchanged, cloud-only.\n\n"
  state=q['route'];ins.update(diagnostic_labels_read=True,once_gate_consumed=True,new_accuracy_result=True,summary=q);scalar.update(latest_source_diagnostic='results/cloud_20261106_coral_gap64_final/SUMMARY.json')
 for p in sorted(M.glob('*REPAIR*.json')):files.append(p.name)
 if (M/'FULL_UPLOAD_RECOVERY.json').exists():files.append('FULL_UPLOAD_RECOVERY.json')
 if (M/'NEXT_CONDITIONAL_MMD_CONFIG.json').exists():files.append('NEXT_CONDITIONAL_MMD_CONFIG.json')
 if (M/'NEXT_CONDITIONAL_MMD_SCIENTIFIC_HASHES.json').exists():files.append('NEXT_CONDITIONAL_MMD_SCIENTIFIC_HASHES.json')
 scalar['next_candidate']=action
 write(M/(stage+'_SPEC.json'),dict(parent=parent,stage=stage,note=note,include_source=False,metadata_files=files,state=state,running=False,pid=None,status_note=action,scalar_updates=scalar,status_insert=ins));print(json.dumps(dict(stage=stage,metadata_files=len(files),parts=len(m['chunks']),archive_sha256=m['archive_sha256'])))
if __name__=='__main__':main()
