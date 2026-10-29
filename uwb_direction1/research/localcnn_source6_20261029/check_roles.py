"""Metadata-role and synthetic validation-selection checks; no pixel/label reads."""
from train import fold_indices,selected_epoch
from pathlib import Path
import json,hashlib
import numpy as np
import torch
B=Path(__file__).resolve().parent
def main():
 cfg=json.loads((B/'PROTOCOL.json').read_text())
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 checked={}
 for h in range(1,7):
  val,tr,va,te=fold_indices(users,h)
  assert set(users[tr])==set(range(1,7))-{h,val} and set(users[va])=={val} and set(users[te])=={h}
  assert not(set(ids[tr])&set(ids[va]) or set(ids[tr])&set(ids[te]) or set(ids[va])&set(ids[te]))
  checked[str(h)]=[len(tr),len(va),len(te)];assert checked[str(h)]==cfg['counts'][str(h)]
 rows=[dict(epoch=1,validation_macro=.8,validation_loss=.5,diagnostic_macro=1.),dict(epoch=2,validation_macro=.8,validation_loss=.4,diagnostic_macro=0.),dict(epoch=3,validation_macro=.8,validation_loss=.4,diagnostic_macro=1.)]
 assert selected_epoch(rows)==2
 # Verify identical constructor/seed to the reused trajectory, without pixels/labels.
 from model import MVGLF
 torch.manual_seed(cfg['seed']);m=MVGLF('local')
 original=torch.load('/workspace/uwb-mvglf-20261029/run01/local/initial.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,original['model'][k]) for k,v in m.state_dict().items())
 r=dict(state='pass',source_metadata_only=True,pixels_or_gesture_labels_read=False,fold_counts=checked,role_disjointness=True,validation_only_selection_ties=True,initial_model_matches_reused_held1=True,source_manifest_sha256=hashlib.sha256(Path(cfg['source_manifest']).read_bytes()).hexdigest())
 p=B/'PREFLIGHT.json';assert not p.exists();p.write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':main()
