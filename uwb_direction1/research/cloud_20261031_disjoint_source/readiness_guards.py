"""Block new configuration/cost/fit until current H-Inception fully closes."""
import json,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prior_closed(base):
 base=Path(base);p=base/'COMPLETE_AUDIT.json'
 if not p.exists():raise RuntimeError('Current H-Inception full replay audit not complete; no next configuration/cost/fit')
 audit=read(p)
 if audit.get('state')!='pass':raise RuntimeError('Current audit not passing; preserve and repair first')
 selected=audit.get('selected_epoch_verified')
 if audit.get('training_updates_replayed_exact')!=7400 or audit.get('epochs_replayed')!=100 or type(selected) is not int or not 1<=selected<=100 or audit.get('users7_8_labels_used') is not False:raise RuntimeError('Wrong or partial current audit scope')
 for n in ['supervisor01/TERMINATION.json','audit_supervisor01/TERMINATION.json']:
  t=read(base/n)
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Current execution incomplete; repair first')
 b=base/'complete_backup_20261031';receipt=read(b/'DRIVE_BACKUP_RECEIPT.json')
 if receipt.get('state')!='private_readback_verified' or not receipt.get('standalone_complete_modelset_recoverable') or not receipt.get('all_chunks_raw_authenticated_readback_and_members_verified') or not receipt.get('all_completed_models_included'):raise RuntimeError('Current complete private modelset not verified; first insurance is insufficient')
 manifest=read(b/'MANIFEST.json');proof=read(b/'READBACK_VERIFIED.json');members=read(b/'payload/MEMBERS.json')
 if proof.get('state')!='remote_archive_and_all_members_verified' or not (proof['archive_sha256']==receipt['archive_sha256']==manifest['archive_sha256']) or sha(b/'payload/MEMBERS.json')!=manifest['members_manifest_sha256'] or proof['tracked_payload_members_verified']!=manifest['tracked_payload_files']:raise RuntimeError('Current full private evidence inconsistent')
 matched=[r for r in members['members'] if r['path']=='study/COMPLETE_AUDIT.json']
 if len(matched)!=1 or matched[0]['sha256']!=sha(p):raise RuntimeError('Current audited result not covered by verified private archive')
 # Only after complete audit/private preservation may a gate summary be opened.
 summary=read(base/'run01/SUMMARY.json')
 if summary['known_baseline_expansion_eligible'] or audit['known_baseline_expansion_eligible'] or summary['predeclared_route']!='close_exact_configuration':raise RuntimeError('Current pass requires separately registered source confirmation, not replacement')
 return True
def real_entry_ready(protocol,marker,prior):
 prior_closed(prior);cfg=read(protocol)
 if not all(type(cfg.get(k)) is int for k in ['seed','epochs','batch_size']) or cfg['epochs']<=0 or cfg['batch_size']<=0:raise RuntimeError('Unset fixed configuration')
 if cfg.get('variant')!='author_default_DCNN_2L' or cfg.get('cost_gate_passed') is not True or not cfg.get('input_preprocessing'):raise RuntimeError('Missing exact variant/input adaptation/cost pass')
 m=read(marker)
 if m.get('state')!='remote_preregistration_exact_readback_verified' or m.get('protocol_sha256')!=sha(protocol):raise RuntimeError('No matching remote preregistration readback')
 return cfg
