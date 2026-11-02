"""Explicit, separately registered audit correction. Original failed run stays intact."""
from pathlib import Path
import sys,json,hashlib
O=Path(__file__).resolve().parent;B=Path('/workspace/uwb-resnet18-imagenet-pilot-20261102')
sys.path.insert(0,str(B))
from common import read,write,sha,load_inputs,ready as original_ready
SCOPE='complete_audited_ResNet18_ImageNetV1_frozen512_LinearSVC_C1_source_held1_IEEE754_pixel_audit'
def ready(marker=True):
 c=original_ready()
 if marker:
  r=read(O/'REMOTE_PREREGISTRATION.json');assert r['state']=='numerical_audit_correction_remote_exact_readback_verified' and r['correction_frozen_sha256']==sha(O/'CORRECTION_HASHES.json') and r['branch_head_verified']
 for n,h in read(O/'CORRECTION_HASHES.json').items():assert sha(O/n)==h,n
 assert sha(B/'FROZEN_HASHES.json')=='80e067862039bdbf03fc600c60600d4c06aeadea3864ab349dade34af5d8624b'
 assert sha(B/'audit_complete.py')=='3a4aa73ab558a41566afcad606df86ba123bb77cd5157f021423ab534b292d13'
 assert sha(B/'audit_supervisor01/process.log')=='fb80d5398ede9ec6e4acd096ec3afe2a981b54e0ca1f517f2800cc099bc5ffa9'
 assert sha(B/'audit_supervisor01/TERMINATION.json')=='834ea7dddb14bb786200b1dd75dec74318c7d61bff858a3c777da3a85da22965'
 p=read(B/'supervisor01/TERMINATION.json');a=read(B/'audit_supervisor01/TERMINATION.json');assert p['exit_code']==0 and a['exit_code']==1 and p['cgroup_oom_kill_delta']==a['cgroup_oom_kill_delta']==0 and not p['resource_stop_reason'] and not a['resource_stop_reason'] and not p['wall_timeout'] and not a['wall_timeout']
 assert not (B/'run01/SUMMARY.json').exists();return c
def complete():
 c=ready();t=read(O/'supervisor01/TERMINATION.json');r=read(O/'COMPLETE_AUDIT.json')
 assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 assert r['state']=='pass' and r['original_strict_FP64_pixel_contract_failed_preserved'] and r['audit_roundoff_correction_preregistered'] and r['scientific_pipeline_outputs_unchanged'] and r['all6900_pixels_analytic_roundoff_bound_verified'] and not r['diagnostic_labels_read'] and not r['users7_8_labels_used']
 for p,h in r['artifact_hashes'].items():assert sha(p)==h,p
 return c
def private_scope(m,v,r,bound):
 assert m['scope']==v['scope']==r['scope']==SCOPE and m['complete_modelset'] and m['shared_frozen_encoder_included_once'] and m['original_failed_audit_included'] and m['corrected_complete_audit_included']
 assert v['complete_modelset_verified'] and r['complete_modelset_verified'] and r['numeric_audit_certified'] and r['state']=='private_readback_verified' and m['archive_sha256']==v['archive_sha256']==r['archive_sha256']
 assert bound['source_preparation/weights/resnet18-f37072fd.pth']=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
