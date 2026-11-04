from pathlib import Path
import sys,json,shutil
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));from common import read,write,sha,require_initial
from checks import verify_audits
from controller import valid_fold
from stream_archive import build
require_initial();assert valid_fold(2);verify_audits(2)
fold=ROOT/'folds/held2';frozen=read(fold/'fusion/FOLD_FROZEN.json');assert not frozen['diagnostic_labels_read'] and not frozen['users7_8_labels_used']
files={}
for path in fold.rglob('*'):
 if path.is_file() and not path.name.endswith(('.lock','.tmp')) and '__pycache__' not in path.parts:files['folds/held2/'+str(path.relative_to(fold))]=path
for path in (ROOT/'attempts/attempt000/held2').rglob('*'):
 if path.is_file():files['attempt000/held2/'+str(path.relative_to(ROOT/'attempts/attempt000/held2'))]=path
for n in ['CONFIG.json','METHOD_LOCK.json','FROZEN_HASHES.json','REMOTE_REAL_PREREGISTRATION.json','INITIAL_PRIVATE_INSURANCE.json','SEED_FILES.json','ASSET_FILES.json','DEPENDENCIES.json']:files['source_lineage/'+n]=ROOT/n
for n in ['CLOUD_DOWNLOAD_INDEX.json','DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json']:files['initial_archive_lineage/'+n]=ROOT/'initial_backup'/n
for n in read(ROOT/'FROZEN_HASHES.json'):
 if n.endswith(('.py','.md','.json','.log')):files['registered_source/'+n]=ROOT/n
out=B/'held2_complete_backup'
m=build(out,'softvote_source6_held2_complete_20261104',files,'complete_held2_protocol_audits_three_arms_unscored_no_delta_dependency_not_final_source6_scoring_authority',chunk_size=7340032,limit=536870912)
m.update(state='complete_unscored_held2_all_protocol_audits_and_outputs_streamed',held=2,delta_dependency=False,initial_data_runtime_source_dependencies_required=True,source6_all5newfolds_complete=False,final_scoring_authority=False);write(out/'MANIFEST.json',m);print(json.dumps(m))
