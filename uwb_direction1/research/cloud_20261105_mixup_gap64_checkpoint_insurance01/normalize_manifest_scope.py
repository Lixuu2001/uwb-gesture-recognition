"""Specialize generic archive-writer flags to intermediate checkpoint scope."""
from pathlib import Path
import json
D=Path(__file__).resolve().parent;p=D/'private_backup/MANIFEST.json';m=json.loads(p.read_text())
assert m['scope']=='intermediate_Mixup_GAP64_producer_native_checkpoint_not_complete_pilot_or_scoring_authority'
before={k:m[k] for k in ['state','complete_modelset','shared_initial_encoder_included_once']}
m.update(state='intermediate_native_checkpoint_streamed_not_complete_pilot',complete_modelset=False,final_scoring_authority=False,full64_training_complete=False)
m.pop('shared_initial_encoder_included_once')
p.write_text(json.dumps(m,indent=2)+'\n')
(D/'MANIFEST_SCOPE_CORRECTION.json').write_text(json.dumps(dict(state='generic_writer_default_flags_specialized_before_whole_archive_verification_and_publication',original_generic_flags=before,complete_modelset=False,final_scoring_authority=False,shared_initial_flag_removed_because_seed_forensics_can_repeat=True,archive_bytes_unchanged=True,archive_sha256=m['archive_sha256'],scientific_or_model_change=False,first_normalization_attempt_syntax_error_repaired=True,chunk_transfer_already_in_progress_but_only_immutable_bytes_and_chunk_rows_used=True),indent=2)+'\n')
print(json.dumps(dict(state=m['state'],complete_modelset=m['complete_modelset'],archive_sha256=m['archive_sha256'])))
