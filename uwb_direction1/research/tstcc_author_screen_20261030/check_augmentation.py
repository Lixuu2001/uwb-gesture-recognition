"""Synthetic source behavior/NumPy2 diagnostic; no corrected training method."""
from pathlib import Path
import json,hashlib,numpy as np,torch
B=Path(__file__).resolve().parent
def check():
 src=(B/'vendor/TS-TCC/dataloader/augmentations.py').read_text();ns={};exec(compile(src,'original_author_augmentation','exec'),ns)
 x=torch.arange(4*3*90).reshape(4,3,90).float();np.random.seed(20261029)
 try:ns['permutation'](x,max_segments=8)
 except ValueError as e:error=str(e)
 else:raise AssertionError('expected ragged-list numpy2 compatibility error absent')
 needle='np.random.permutation(splits)';assert src.count(needle)==1
 compat=src.replace(needle,'[splits[j] for j in np.random.permutation(len(splits))]');space={};exec(compile(compat,'only_ragged_list_compatibility','exec'),space)
 np.random.seed(20261029);ret=space['permutation'](x,max_segments=8)
 collapsed=[i for i in range(4) if torch.equal(ret[i,0],ret[i,1]) and torch.equal(ret[i,0],ret[i,2])];assert collapsed
 result=dict(state='source_augmentation_semantics_checked_synthetic_only',original_numpy2_error=error,original_exact_source_sha256=hashlib.sha256(src.encode()).hexdigest(),temporary_analysis_adapter='onlyraggedpermutation listordering changed to indexordering; original ret[i]=pat[0,warp] preserved',collapsed_all_channels_record_indices=collapsed,total_synthetic_records=4,observation='For permuted records pinned source broadcasts channel0 into all channels. Not a full multichannel time-permutation implementation. No silent repair or real-data decision.',real_UWB_values_read=False,real_experiment_started=False)
 (B/'AUGMENTATION_SEMANTICS_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':check()
