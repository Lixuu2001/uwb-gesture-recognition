"""Build once, immutable interrupted input45 and newly registered sources only."""
from pathlib import Path
from trial_common import B,ready,read,write,sha
from stream_archive import build

def main():
    c=ready();old=Path(c['operational_recovery']['original_root'])
    files={'study/FROZEN_HASHES.json':B/'FROZEN_HASHES.json',
           'study/REMOTE_REAL_PREREGISTRATION.json':B/'REMOTE_REAL_PREREGISTRATION.json'}
    for name in read(B/'FROZEN_HASHES.json'):files['study/'+name]=B/name
    for p in sorted((old/'run01').iterdir()):
        if p.is_file():files['original/run01/'+p.name]=p
    for name in ['FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json']:
        files['original/'+name]=old/name
    m=build(B/'recovery_input_insurance','adam100_checkpoint45_recovery_input_20261103',
            files,'interrupted_Adam100_checkpoint45_and_registered_continuation_sources',
            chunk_size=7340032,limit=67108864)
    m.update(state='interrupted_checkpoint45_input_source_insurance',complete_modelset=False,
             numeric_audit_certified=False,full100_training_complete=False,diagnostic_labels_scored=False)
    write(B/'recovery_input_insurance/MANIFEST.json',m)
if __name__=='__main__':main()
