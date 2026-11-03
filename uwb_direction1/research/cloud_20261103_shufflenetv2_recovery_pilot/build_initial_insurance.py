"""Build once: private untrained seed/source/ID insurance, never learned results."""
from pathlib import Path
import json
from common import B,ready,read,write,sha
from stream_archive import build
def main():
 c=ready(marker=False);out=B/'initial_insurance';assert not out.exists()
 from engine import make,torch,semantic
 seed=B/'seed_model';seed.mkdir(exist_ok=False);m=make()
 torch.save(m.fc.state_dict(),seed/'initial12_head.pt')
 info=dict(state='untrained_full_original_encoder_plus_seeded12_head',seed=20261029,model_state_sha256=semantic(m.state_dict()),head_checkpoint_sha256=sha(seed/'initial12_head.pt'),weights_sha256=c['weights_sha256'],frozen_sha256=sha(B/'FROZEN_HASHES.json'),actual_UWB_fit_started=False,diagnostic_labels_scored=False)
 write(seed/'IDENTITY.json',info)
 files={'seed/initial12_head.pt':seed/'initial12_head.pt','seed/IDENTITY.json':seed/'IDENTITY.json','study/FROZEN_HASHES.json':B/'FROZEN_HASHES.json'}
 for name in read(B/'FROZEN_HASHES.json'):files['study/'+name]=B/name
 source=Path(c['source_preparation'])
 for name in ['native_shufflenet.py','check_full_native.py','NATIVE_CHECK.json','NATIVE_CHECK.log','WEIGHTS_SOURCE.json','local_author_sources/shufflenetv2.py','local_author_sources/LICENSE','weights/shufflenetv2_x1-5666bf0f80.pth']:files['native/'+name]=source/name
 files['support/native_finetune.py']=Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py')
 support=Path('/workspace/uwb-resnet18-imagenet-source-20261102')
 for name in ['native_resnet18.py','local_author_sources/resnet.py','local_author_sources/LICENSE']:files['support/base/'+name]=support/name
 files['dependencies/source_ids.npz']=Path(c['source_manifest'])
 for key in ['reference_predictions','reference_summary']:files['reference/'+Path(c[key]).name]=Path(c[key])
 cost=Path(c['cost_workspace'])
 for name in ['POLICY.json','COST_HASHES.json','VERIFIED_COST_RESULT.json','profile01/COST_RESULT.json','supervisor01/TERMINATION.json']:files['cost/'+name]=cost/name
 manifest=build(out,'shufflenetv2_x1_recovery40_initial_20261103',files,'initial_seed_only_ShuffleNetV2_x1_SGDM40_B15_recovery_20261103',chunk_size=7340032,limit=67108864)
 manifest['state']='untrained_seed_and_scientific_source_insurance_streamed'
 manifest['complete_modelset']=False
 manifest['trained_model_or_audit_included']=False
 write(out/'MANIFEST.json',manifest);print(json.dumps(manifest))
if __name__=='__main__':main()
