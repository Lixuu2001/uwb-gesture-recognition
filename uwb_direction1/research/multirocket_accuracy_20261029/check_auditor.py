"""Synthetic verifier checks, including deliberate tampering and label boundary."""
import json,tempfile
from pathlib import Path
import numpy as np
from adapter import fit_readout
import audit_result as a
def main():
 rng=np.random.default_rng(901);f=rng.normal(size=(72,23));y=np.arange(72)%4;u=np.arange(72)%3
 r=fit_readout(f,y,u);residual=a.check_readout(f,y,u,r)
 altered=dict(r);altered['coef']=r['coef'].copy();altered['coef'][0,0]+=.01
 rejected=False
 try:a.check_readout(f,y,u,altered)
 except AssertionError:rejected=True
 assert rejected
 import copy
 altered=copy.deepcopy(r);altered['scaler'].mean_[0]+=.01
 rejected=False
 try:a.check_readout(f,y,u,altered)
 except AssertionError:rejected=True
 assert rejected
 original_base=a.B;original_load=a.np.load;calls=[]
 try:
  with tempfile.TemporaryDirectory() as tmp:
   a.B=Path(tmp);(a.B/'run01').mkdir();(a.B/'run01/STATE.json').write_text('{"state":"training"}')
   def forbidden(*args,**kwargs):calls.append(True);raise RuntimeError('array access forbidden')
   a.np.load=forbidden
   try:a.audit(a.B/'unused.json')
   except AssertionError:pass
   else:raise AssertionError('unfinished run accepted')
   assert not calls and not (a.B/'unused.json').exists()
 finally:a.B=original_base;a.np.load=original_load
 result=dict(state='pass',synthetic_only=True,uwb_arrays_or_labels_read=False,
  correct_weighted_readout_accepted=True,coefficient_tamper_rejected=True,
  scaler_tamper_rejected=True,unfinished_run_rejected_before_array_or_label_access=True,
  synthetic_objective_residual=residual,completed_real_audit_executed=False)
 p=Path(__file__).with_name('AUDITOR_SYNTHETIC_CHECK.json');assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
