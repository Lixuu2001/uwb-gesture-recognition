"""Role-limited label capability. Held-user truth has no API in a worker."""
from pathlib import Path
import numpy as np
from lazy_outer_data import OuterInputs,sha
TRUTH_SHA='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
class RoleLabels:
    def __init__(self,roles,provider,probe=False):
        self.roles=roles;self._provider=provider;self.probe=probe;self.reads=[]
    def get(self,role,ids):
        ids=np.asarray(ids)
        if role not in ('train','validation'):raise PermissionError('Held truth is sealed')
        if self.probe and role!='train':raise PermissionError('Cost probe is train-only')
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid label request')
        if not set(ids.tolist())<=set(self.roles['sample_ids'][role]):raise PermissionError('Out-of-role truth request')
        if self.probe and not set(self.roles['users']['train'])<=set([3,4,5,6]):raise PermissionError('Probe subject restriction')
        y=np.array(self._provider(ids),copy=True)
        if y.shape!=ids.shape or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=12):raise ValueError('Invalid labels')
        self.reads.append(dict(role=role,count=len(ids)))
        return y
class RealData:
    def __init__(self,held,probe=False,production_release=None):
        from core import ROOT,read
        p=read(ROOT/'PROTOCOL.json')
        if not probe:
            if production_release is None:raise PermissionError('Production not admitted')
            r=read(production_release)
            if r.get('production_admitted') is not True or r.get('protocol_sha256')!=sha(ROOT/'PROTOCOL.json'):raise PermissionError('No exact production release')
        root=Path(p['training_data_root'])
        self.inputs=OuterInputs(root/'cache/X_clutterremoved.npy',root/'cache/subj_clutterremoved.npy',p['source_ids'])
        self.held=held;self.roles=self.inputs.roles(held);self.probe=probe
        if probe:
            if held!=1:raise PermissionError('Only frozen source-user cost probe')
            ids=self.inputs.ids[np.isin(self.inputs.users,[3,4,5,6])]
            self.roles={**self.roles,'sample_ids':{'train':ids.tolist(),'validation':[],'test':[]},'users':{'train':[3,4,5,6],'validation':[],'test':[]}}
        path=root/'cache/y_clutterremoved.npy'
        if sha(path)!=TRUTH_SHA:raise ValueError('Truth file bytes changed')
        self._truth=None
        def provider(ids):
            if self._truth is None:self._truth=np.load(path,mmap_mode='r',allow_pickle=False)
            return self._truth[ids]
        self.labels=RoleLabels(self.roles,provider,probe)
    def batch(self,role,ids):
        from lazy_outer_data import rms_batch
        if role not in self.roles['sample_ids'] or not set(np.asarray(ids).tolist())<=set(self.roles['sample_ids'][role]):raise PermissionError('Out-of-role input')
        return rms_batch(self.inputs.raw,np.asarray(ids))
