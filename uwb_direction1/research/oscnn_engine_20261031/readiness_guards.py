"""Fail closed before cost or UWB access until the existing ResNet gate closes.

The source engine is not an executable real-data entry and this is not an
experiment registration. No batch/seed/epoch/resource policy defaults are chosen.
"""
import json,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prior_closed(base):
 base=Path(base);p=base/'COMPLETE_AUDIT.json'
 if not p.exists():raise RuntimeError('Previous full replay audit is not complete; no new config/cost/fit')
 audit=read(p)
 if audit.get('state')!='pass':raise RuntimeError('Previous audit not passing; repair first')
 for n in ['supervisor01/TERMINATION.json','audit_supervisor01/TERMINATION.json']:
  t=read(base/n)
  if t['exit_code']!=0 or t['wall_timeout']:raise RuntimeError('Previous execution incomplete; repair first')
 receipt=read(base/'complete_backup_20261031/DRIVE_BACKUP_RECEIPT.json')
 if receipt.get('state')!='private_readback_verified' or not receipt.get('standalone_complete_modelset_recoverable') or not receipt.get('all_chunks_raw_authenticated_readback_and_members_verified'):raise RuntimeError('Previous complete private modelset not yet verified')
 # Read diagnostic summary only after the prior full audit and complete backup.
 summary=read(base/'run01/SUMMARY.json')
 if summary['predeclared_route']!='close_exact_configuration' or summary['known_baseline_expansion_eligible']:raise RuntimeError('Prior pass requires separately registered source confirmation; no replacement method')
 return True
def real_entry_ready(protocol_path,marker_path,prior_base):
 prior_closed(prior_base);cfg=read(protocol_path)
 if not all(k in cfg for k in ['seed','epochs','batch_size','variant','cost_gate_passed']):raise RuntimeError('No complete fixed experiment configuration')
 if type(cfg['epochs']) is not int or cfg['epochs']<=0 or type(cfg['seed']) is not int or type(cfg['batch_size']) is not int or cfg['batch_size']<=0:raise RuntimeError('Unset or invalid fixed configuration')
 if cfg['variant']!='author_multivariate_OS_CNN' or cfg['cost_gate_passed'] is not True:raise RuntimeError('Wrong variant or no predeclared cost pass')
 marker=read(marker_path)
 if marker['state']!='remote_preregistration_exact_readback_verified' or marker['protocol_sha256']!=sha(protocol_path):raise RuntimeError('No matching remote registration readback')
 return cfg
