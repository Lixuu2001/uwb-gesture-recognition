"""Preserve a finished producer while independent replay runs; never score labels."""
from pathlib import Path
import argparse,json,sys
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',default='/workspace/uwb-shufflenetv2-x1-recovery-pilot-20261103');p=a.parse_args();B=Path(p.root).resolve();sys.path.insert(0,str(B))
 from common import ready,read,write,sha
 from stream_archive import build
 c=ready();term=read(B/'supervisor01/TERMINATION.json')
 assert type(term['exit_code']) is int and term['exit_code']==0
 assert term['wall_timeout'] is False and term['resource_stop_reason'] is None and term['cgroup_oom_kill_delta']==0
 out=B/'run01';assert read(out/'STATE.json')['state']=='producer_complete_diagnostic_unscored'
 selection=read(out/'TRAIN_SELECTION.json');assert selection['epochs']==40 and selection['updates']==12560
 assert selection['diagnostic_labels_read'] is False and selection['users7_8_labels_used'] is False
 hashes=read(out/'FILES.json')
 required=['initial12_head.pt','best_model.pt','final_full.pt','split.npz','steps.jsonl','TRAIN_SELECTION.json','validation_unscored.npz','diagnostic_unscored.npz','IDENTITY.json','STATE.json']
 required += [f'epoch{i:02d}.json' for i in range(1,41)]+[f'validation_epoch{i:02d}.npz' for i in range(1,41)]
 assert all(n in hashes for n in required)
 for name,h in hashes.items():assert sha(out/name)==h,name
 target=B/'producer_complete_insurance';assert not target.exists(),'Prior response/IDs require inspection, no automatic rebuild'
 files={'study/'+name:B/name for name in read(B/'FROZEN_HASHES.json')}
 for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json','LAUNCH_INTENT.json','LAUNCH_OBSERVATION.json']:files['study/'+name]=B/name
 for name in [*hashes,'FILES.json']:files['run01/'+name]=out/name
 for path in sorted((B/'supervisor01').iterdir()):
  if path.is_file():files['supervisor01/'+path.name]=path
 source=Path(c['source_preparation'])
 for name in ['native_shufflenet.py','check_full_native.py','NATIVE_CHECK.json','NATIVE_CHECK.log','PAPER_REVIEW.json','WEIGHTS_SOURCE.json','local_author_sources/shufflenetv2.py','local_author_sources/LICENSE','weights/shufflenetv2_x1-5666bf0f80.pth']:files['source/'+name]=source/name
 files['support/native_finetune.py']=Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py')
 support=Path('/workspace/uwb-resnet18-imagenet-source-20261102')
 for name in ['native_resnet18.py','local_author_sources/resnet.py','local_author_sources/LICENSE']:files['support/base/'+name]=support/name
 files['dependencies/source_ids.npz']=Path(c['source_manifest'])
 for key in ['reference_predictions','reference_summary']:files['reference/'+Path(c[key]).name]=Path(c[key])
 cost=Path(c['cost_workspace'])
 for name in ['POLICY.json','COST_HASHES.json','VERIFIED_COST_RESULT.json','profile01/COST_RESULT.json','supervisor01/TERMINATION.json','guard.py','cost_math.py','profile_cost.py','run_screen.py']:files['cost/'+name]=cost/name
 files['operations/preserve_completed_producer.py']=Path(__file__).resolve()
 m=build(target,'shufflenetv2_x1_recovery40_producer_complete_unaudited_20261103',files,'producer_complete_ShuffleNetV2_x1_recovery40_without_numeric_audit',chunk_size=7340032,limit=c['archive_upper_bytes'])
 m.update(state='complete_producer_unscored_not_numeric_audited_insurance_streamed',complete_modelset=False,full_training_complete=True,numeric_audit_certified=False,diagnostic_labels_scored=False);write(target/'MANIFEST.json',m)
 for name,h in hashes.items():assert sha(out/name)==h,name
 print(json.dumps(dict(directory=str(target),archive_bytes=m['archive_bytes'],parts=len(m['chunks']),members=m['tracked_payload_files'],producer_updates=12560,full_numeric_audit_complete=False,diagnostic_labels_scored=False)))
if __name__=='__main__':main()
