"""Source-only split/metadata API; actual UWB entry requires prior full closure.

GPL-3.0-only. No data paths, preprocessing choice, epoch/batch/seed default or CLI.
"""
from pathlib import Path
import sys
sys.path.insert(0,'/workspace/uwb-disjoint-source-probe-20261031')
from readiness_guards import real_entry_ready
import numpy as np

def role_metadata(meta,expected):
 if set(meta)!=set(expected) or set(expected)!={'training','validation','diagnostic'}:raise ValueError('Three registered roles required')
 seen=set();seen_users=set()
 for role in ['training','validation','diagnostic']:
  users=expected[role]
  if not isinstance(users,list) or not users or any(type(u) is not int or not 1<=u<=6 for u in users) or len(set(users))!=len(users) or seen_users.intersection(users):raise ValueError('Only disjoint source users1..6 allowed')
  seen_users.update(users);row=meta[role]
  if set(row)!={'ids','users'}:raise ValueError('Only label-free ID/user metadata allowed')
  ids=row['ids'];us=row['users']
  if not all(isinstance(a,np.ndarray) and a.dtype.kind in 'iu' and a.ndim==1 for a in [ids,us]) or len(ids)<1 or len(ids)!=len(us) or (ids<0).any():raise ValueError('Invalid ID/user metadata')
  if set(us.tolist())!=set(users) or len(set(ids.tolist()))!=len(ids) or seen.intersection(ids.tolist()):raise ValueError('Wrong source users or duplicate/overlapping IDs')
  seen.update(ids.tolist())
 return meta

def load_registered_source(protocol,marker,prior,metadata_loader,values_loader,labels_loader):
 """Refusal happens before any supplied loader; diagnostic labels never loaded."""
 cfg=real_entry_ready(protocol,marker,prior)
 meta=role_metadata(metadata_loader(),cfg.get('role_users',{}))
 # Values/labels/preprocessing correctness still require the future registered
 # loader and independent raw-map audit. This API does not choose them.
 values={role:values_loader(role,meta[role]['ids']) for role in meta}
 labels={role:labels_loader(role,meta[role]['ids']) for role in ['training','validation']}
 return cfg,meta,values,labels
