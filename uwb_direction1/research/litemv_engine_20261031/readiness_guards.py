"""Fail closed before future cost/config/UWB entry; no experiment defaults."""
import hashlib,json
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prior_closed(base):
 base=Path(base);p=base/'COMPLETE_AUDIT.json'
 if not p.exists():raise RuntimeError('Previous OS-CNN full replay audit is not complete; no new configuration/cost/fit')
 audit=read(p)
 if audit.get('state')!='pass':raise RuntimeError('Previous audit not passing; preserve and repair first')
 if audit.get('training_updates_replayed_exact')!=29400 or audit.get('epochs_replayed')!=100 or audit.get('selection')!='source_final_last' or audit.get('users7_8_labels_used') is not False:raise RuntimeError('Wrong or partial previous audit scope')
 for n in ['supervisor01/TERMINATION.json','audit_supervisor01/TERMINATION.json']:
  t=read(base/n)
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Previous execution incomplete; repair first')
 backup=base/'complete_backup_20261031';receipt=read(backup/'DRIVE_BACKUP_RECEIPT.json')
 if receipt.get('state')!='private_readback_verified' or not receipt.get('standalone_complete_modelset_recoverable') or not receipt.get('all_chunks_raw_authenticated_readback_and_members_verified') or not receipt.get('all_completed_models_included'):raise RuntimeError('Previous complete private modelset not yet verified')
 manifest=read(backup/'MANIFEST.json');verified=read(backup/'READBACK_VERIFIED.json');members=read(backup/'payload/MEMBERS.json')
 if verified.get('state')!='remote_archive_and_all_members_verified' or not (verified['archive_sha256']==receipt['archive_sha256']==manifest['archive_sha256']) or sha(backup/'payload/MEMBERS.json')!=manifest['members_manifest_sha256'] or verified['tracked_payload_members_verified']!=manifest['tracked_payload_files']:raise RuntimeError('Previous complete backup evidence inconsistent')
 rows=[r for r in members['members'] if r['path']=='study/COMPLETE_AUDIT.json']
 if len(rows)!=1 or rows[0]['sha256']!=sha(p):raise RuntimeError('Previous audited result absent from verified complete backup')
 # Deliberately do not open SUMMARY until full audit/termination/private proof.
 summary=read(base/'run01/SUMMARY.json')
 if summary['predeclared_route']!='close_exact_configuration' or summary['known_baseline_expansion_eligible'] or audit['known_baseline_expansion_eligible']:raise RuntimeError('Prior pass requires registered source confirmation; no replacement method')
 return True
def real_entry_ready(protocol_path,marker_path,prior_base):
 prior_closed(prior_base);cfg=read(protocol_path)
 for k in ['seed','epochs','batch_size']:
  if type(cfg.get(k)) is not int or (k!='seed' and cfg[k]<=0):raise RuntimeError('Unset fixed experiment configuration')
 if cfg.get('variant')!='author_LITEMV_full' or cfg.get('cost_gate_passed') is not True or cfg.get('input_preprocessing') not in ['explicit_RMS_adaptation','explicit_RMS_then_author_znormalization_adaptation']:raise RuntimeError('Missing fixed source variant/preprocessing or prospective cost pass')
 marker=read(marker_path)
 if marker.get('state')!='remote_preregistration_exact_readback_verified' or marker.get('protocol_sha256')!=sha(protocol_path):raise RuntimeError('No matching remote preregistration readback')
 return cfg
