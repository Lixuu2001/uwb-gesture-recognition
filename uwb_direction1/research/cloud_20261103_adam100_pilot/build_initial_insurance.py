"""Prospective initial seed insurance; no learned model/automatic resume."""
from trial_common import B,ready,read,write,sha
from stream_archive import build
from backup import scientific_files

def main():
 c=ready();assert not (B/'initial_insurance').exists()
 from adam_engine import make,torch,semantic
 seed=B/'seed_model';seed.mkdir(exist_ok=False);model=make()
 torch.save(model.fc.state_dict(),seed/'initial12_head.pt')
 write(seed/'IDENTITY.json',dict(state='untrained_full_initial_encoder_seeded12head',
  seed=20261029,model_state_sha256=semantic(model.state_dict()),
  head_checkpoint_sha256=sha(seed/'initial12_head.pt'),weights_sha256=c['weights_sha256'],
  frozen_sha256=sha(B/'FROZEN_HASHES.json'),actual_UWB_fit_started=False,
  diagnostic_labels_scored=False))
 files=scientific_files(c)
 files['seed/initial12_head.pt']=seed/'initial12_head.pt'
 files['seed/IDENTITY.json']=seed/'IDENTITY.json'
 m=build(B/'initial_insurance','shufflenetv2_x1_adam100_initial_20261103',files,
  'initial_seed_only_ShuffleNetV2_x1_Adam100_B20_20261103',
  chunk_size=c['streaming_chunk_bytes'],limit=c['archive_upper_bytes'])
 m.update(state='untrained_seed_source_insurance_streamed',complete_modelset=False,
  trained_model_or_audit_included=False)
 write(B/'initial_insurance/MANIFEST.json',m)
if __name__=='__main__':main()
