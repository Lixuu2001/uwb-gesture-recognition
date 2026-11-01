"""Metadata/hash guard for the completed, privately preserved source6 failure."""
from pathlib import Path
import hashlib,importlib.util,json
B=Path('/workspace/uwb-hinception-sixfold-20261031')
BACKUP=Path('/workspace/uwb-hinception-sixfold-complete-20261101')
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prior_closed(base):
 if Path(base).resolve()!=B:raise RuntimeError('Wrong source6 predecessor')
 if sha(CORE)!='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229':raise RuntimeError('Changed completion guard')
 spec=importlib.util.spec_from_file_location('disjoint_source6_completion',CORE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.check_complete(B)
 receipt=read(BACKUP/'DRIVE_BACKUP_RECEIPT.json');proof=read(BACKUP/'READBACK_VERIFIED.json');manifest=read(BACKUP/'MANIFEST.json')
 if receipt.get('state')!='private_readback_verified' or not receipt.get('standalone_complete_modelset_recoverable') or not receipt.get('all_six_full_modelsets_included') or not proof.get('complete_modelset_verified'):raise RuntimeError('Complete six-model private readback required')
 if not (receipt['archive_sha256']==proof['archive_sha256']==manifest['archive_sha256']) or proof['chunks_verified']!=41 or proof['archive_members']!=478:raise RuntimeError('Inconsistent complete private scope')
 members=read(BACKUP/'payload/MEMBERS.json')
 if sha(BACKUP/'payload/MEMBERS.json')!=manifest['members_manifest_sha256']:raise RuntimeError('Changed private member manifest')
 bindings={r['path']:r['sha256'] for r in members['members']}
 for name in ['run01/SUMMARY.json','run01/ALL_NEW_PREDICTIONS_FROZEN.json']+[f'run01/held{h}/COMPLETE_AUDIT.json' for h in range(2,7)]:
  if bindings.get(name)!=sha(B/name):raise RuntimeError('Current closure not in preserved archive: '+name)
 summary=read(B/'run01/SUMMARY.json')
 if summary['predeclared_gate_passed'] or summary['route']!='close_exact_CPU100_source_confirmation_no_rescue' or summary['users7_8_labels_used'] or summary['independent_validation']:raise RuntimeError('Predecessor does not authorize next candidate')
 return True
