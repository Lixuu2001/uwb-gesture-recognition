"""Cumulative index for already verified private fold recoveries; never a score release."""
from pathlib import Path
import sys,json,time
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready

def main(expected_last):
 assert expected_last in range(1,7);ready();base=P/'held1_production64_backup';bm=read(base/'MANIFEST.json');bv=read(base/'READBACK_VERIFIED.json');assert bv['archive_sha256']==bm['archive_sha256'] and bv['chunks_verified']==len(bm['chunks'])
 folds={};first=read(P/'HELD1_COMPLETE_PRIVATE_RECEIPT.json');physical=read(P/'HELD1_COMPLETION_PHYSICAL_RECOVERY_VERIFIED.json');assert physical['restored_files']==first['source_files'] and physical['restored_bytes']==first['source_bytes']
 folds['1']=dict(base_key='held1_production22',additional_archive=read(P/'held1_completion_backup/CLOUD_DOWNLOAD_INDEX.json'),recipe=read(P/'HELD1_COMPLETION_DEDUP_RECIPE.json'),physical_recovery=physical,private_receipt_sha256=sha(P/'HELD1_COMPLETE_PRIVATE_RECEIPT.json'))
 for h in range(2,7):
  f=P/f'HELD{h}_COMPLETE_PRIVATE_RECEIPT.json'
  if not f.exists():continue
  receipt=read(f);d=P/f'held{h}_completed_backup';m=read(d/'MANIFEST.json');v=read(d/'READBACK_VERIFIED.json');recipe=read(P/f'HELD{h}_COMPLETE_OBJECT_RECIPE.json');physical=read(P/f'HELD{h}_COMPLETE_PHYSICAL_RECOVERY_VERIFIED.json')
  assert receipt['held']==h and receipt['archive_sha256']==v['archive_sha256']==m['archive_sha256'] and v['chunks_verified']==len(m['chunks'])
  assert receipt['source_files']==physical['restored_files']==recipe['files'] and receipt['source_bytes']==physical['restored_bytes']==recipe['bytes'] and recipe['base_archive_sha256']==bm['archive_sha256']
  assert read(P/f'folds/held{h}/run/PREDICTIONS_FROZEN.json')['full_training_replayed']
  folds[str(h)]=dict(base_key='held1_production22',object_archive=read(d/'CLOUD_DOWNLOAD_INDEX.json'),recipe=recipe,physical_recovery=physical,private_receipt_sha256=sha(f),audit_sha256=sha(P/f'HELD{h}_COMPLETE_OBJECT_AUDIT.json'))
 found=sorted(map(int,folds));assert found==list(range(1,expected_last+1)),found
 index=dict(state='cumulative_completed_SWAD_fold_private_raw_and_physical_recovery_registry_NOT_cohort_score_release',completed_private_verified_held=found,initial_raw_strong_GAP_all6_initial_chain=read(P/'INITIAL_COMPOSITE_RECOVERY_INDEX.json'),bases=dict(held1_production22=read(base/'CLOUD_DOWNLOAD_INDEX.json')),base_existing_raw_allmember_proof=bv,folds=folds,all_listed_folds_native64_independent64_all4_audits_and_physical_recovery_verified=True,all_six_fold_private_entries_present=(found==list(range(1,7))),canonical_final_all6_archive_and_exact_release_still_required=True,whole_job_resume_authorized=False,singlefold_or_cohort_scoring_authorized=False,do_not_retrain_or_overwrite_completed_models=True,preserve_original_phase_identity_and_elapsed_budget=True,source_frozen_sha256=sha(P/'FROZEN_HASHES.json'),scientific_source_or_numeric_config_changed=False,generated_by_operator_sha256=sha(__file__),observed_unix=time.time())
 f=P/f'VERIFIED_COMPLETED_HELD1_TO_HELD{expected_last}_RECOVERY_INDEX.json';assert not f.exists();write(f,index)
 print(json.dumps(dict(path=str(f),completed_private_verified_held=found,all_six=expected_last==6,score_authorized=False)))
if __name__=='__main__':assert len(sys.argv)==2;main(int(sys.argv[1]))
