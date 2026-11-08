"""Small post-score result/once-marker recovery supplement; no model or truth access."""
from pathlib import Path
import sys,json
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from stream_archive import build,verify
NAMES=['SUMMARY.json','FINAL_RESULT_STATE.json','FINALIZE_ONCE.lock','INFERENCE_ONCE.lock','FULL_PRIVATE_RELEASE.json','FINAL_REPORT.md','FINAL_ANALYSIS.json','NUMERICAL_TOLERANCE_AUDIT.json','PARTICIPANT_METRICS.csv','CLASS_METRICS.csv','COST_LEDGER.json','INFERENCE_COST.json','INFERENCE_COST_EXECUTION.json','COHORT_SCORE_EXECUTION.json','cohort_score.stdout.log','cohort_score.stderr.log','inference_cost.stdout.log','inference_cost.stderr.log','FOLLOWUP_SCOPE.md','SWA_SOURCE_CHECK_RECEIPT.json','OUTER8_ROLE_PLANNER_GENERATED_CHECKS.json','operations/outer_subject_roles.py','operations/render_cohort_report.py','operations/final_result_archive.py','canonical_private_PUBLICATION_VERIFIED.json','ALL_SIX_PREDICTIONS_FROZEN.json','FROZEN_HASHES.json']
def create():
 s=read(P/'SUMMARY.json');assert s['finalizer_invocations']==1 and read(P/'FINAL_RESULT_STATE.json')['finalizer_consumed']
 assert (P/'FINALIZE_ONCE.lock').exists() and (P/'INFERENCE_ONCE.lock').exists()
 files={'terminal/'+n:P/n for n in NAMES};before={n:sha(f) for n,f in files.items()}
 m=build(P/'result_backup','swad_native_loso6_scored_terminal_20261108',files,'terminal_scores_costs_reports_and_consumed_once_markers_recovery_supplement_requires_existing_complete_canonical_models',chunk_size=7340032,limit=2097152)
 m.update(complete_modelset=False,terminal_metadata_only=True,full_models_remain_in_prior_canonical_archive=True,summary_sha256=sha(P/'SUMMARY.json'));write(P/'result_backup/MANIFEST.json',m)
 assert before=={n:sha(f) for n,f in files.items()};return dict(parts=len(m['chunks']),archive_bytes=m['archive_bytes'],payload=m['tracked_payload_files'],summary_unchanged=True)
def check():
 b=P/'result_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
 for r in rows:
  d=r['metadata'];assert not d['shared'] and d['permissions']==[dict(type='user',role='owner')] and d['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and r['raw_fetch'] and r['download_verified'] and not r['include_base64']
 v=verify(m,rows);write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
 idx=dict(state='scored_terminal_metadata_and_consumed_once_markers_private_raw_allmember_verified',folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload=m['tracked_payload_files'],summary_sha256=sha(P/'SUMMARY.json'),complete_modelset=False,requires_canonical_model_archive_sha256=read(P/'full_backup/MANIFEST.json')['archive_sha256'],raw_fetch_options=dict(download_raw_file=True,include_base64=False),chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows],restore_terminal_summary_and_once_markers_before_any_finalizer_consideration=True,whole_job_resume_authorized=False)
 write(b/'CLOUD_DOWNLOAD_INDEX.json',idx);write(P/'FINAL_SCORED_RECOVERY_INDEX.json',dict(state='complete_canonical_models_plus_terminal_scored_metadata_recovery_chain_no_retraining_or_rescoring',canonical_models_and_original_raw_GAP_chain=read(P/'FULL_COMPOSITE_RECOVERY_INDEX.json'),terminal_result_supplement=idx,terminal_raw_allmember_proof=v,finalizer_consumed=True,never_repeat_existing_fits_or_cohort_scoring=True,summary_sha256=sha(P/'SUMMARY.json'),source_frozen_sha256=sha(P/'FROZEN_HASHES.json'),whole_job_resume_authorized=False,restore_to_isolated_destination_before_live_use=True));return v
if __name__=='__main__':print(json.dumps(create() if sys.argv[1:]==['build'] else check()))
