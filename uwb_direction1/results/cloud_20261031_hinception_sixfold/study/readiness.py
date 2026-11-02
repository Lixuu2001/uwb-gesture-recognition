"""Pure metadata guards before new source-fold value/label access."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def ready(cfg):
 marker=read(B/'REMOTE_PREREGISTRATION.json')
 if marker.get('state')!='remote_source_confirmation_exact_readback_verified' or marker.get('protocol_sha256')!=sha(B/'PROTOCOL.json') or marker.get('frozen_sha256')!=sha(B/'FROZEN_HASHES.json'):raise RuntimeError('No exact source-confirmation registration')
 for n,d in read(B/'FROZEN_HASHES.json').items():
  if sha(B/n)!=d:raise RuntimeError('Changed frozen prerequisite: '+n)
 up=Path(cfg['upstream_workspace']);audit=read(up/'COMPLETE_AUDIT.json');receipt=read(up/'complete_backup_20261031/DRIVE_BACKUP_RECEIPT.json');proof=read(up/'complete_backup_20261031/READBACK_VERIFIED.json')
 if sha(up/'PROTOCOL.json')!=cfg['upstream_protocol_sha256'] or sha(up/'COMPLETE_AUDIT.json')!=cfg['upstream_audit_sha256'] or sha(up/'complete_backup_20261031/DRIVE_BACKUP_RECEIPT.json')!=cfg['upstream_private_receipt_sha256']:raise RuntimeError('Changed reused held1 evidence')
 if sha(Path(cfg['reused_held1']['directory'])/'diagnostic_predictions_unscored.npz')!=cfg['reused_held1']['prediction_sha256']:raise RuntimeError('Changed reused held1 predictions')
 if audit['state']!='pass' or not audit['known_baseline_expansion_eligible'] or audit['training_updates_replayed_exact']!=7400 or audit['users7_8_labels_used'] is not False:raise RuntimeError('Source-held1 expansion was not authorized by original gate')
 if receipt['state']!='private_readback_verified' or not receipt['standalone_complete_modelset_recoverable'] or not receipt['all_completed_models_included'] or proof['archive_sha256']!=receipt['archive_sha256']:raise RuntimeError('Reused modelset not privately preserved')
 for n in ['supervisor01/TERMINATION.json','audit_supervisor01/TERMINATION.json']:
  t=read(up/n)
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Reused trajectory not successfully complete')
 if cfg['epochs']!=100 or cfg['seed']!=20261029 or cfg['batch_size']!=64 or cfg['threads']!=1 or cfg['parameters']!=851434 or cfg['trainable_parameters']!=601006:raise RuntimeError('Changed fixed method/settings')
 if not cfg['cost_gate_passed'] or cfg['maximum_training_cost_estimate_seconds']>43100 or cfg['maximum_audit_estimate_seconds']>14400:raise RuntimeError('No bounded resource pass')
 for prefix in ['localcnn','gap']:
  if sha(cfg[prefix+'_summary'])!=cfg[prefix+'_summary_sha256'] or sha(cfg[prefix+'_audit'])!=cfg[prefix+'_audit_sha256']:raise RuntimeError('Changed fixed comparator')
 for group in cfg['comparator_predictions'].values():
  for row in group.values():
   if sha(row['path'])!=row['sha256']:raise RuntimeError('Changed comparator prediction')
 return True
