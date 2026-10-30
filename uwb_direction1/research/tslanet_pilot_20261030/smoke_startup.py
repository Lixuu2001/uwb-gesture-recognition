"""Run the actual startup to intercepted synthetic np.load; zero real values."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import run_pilot as run
class Intercept(RuntimeError):pass
with tempfile.TemporaryDirectory() as d:
 b=Path(d);cfg=dict(cost_gate_passed=True,pretraining_epochs=50,classification_epochs_per_arm=100,batch_size=16)
 for key,value in dict(previous_state=dict(state='complete',pid=None),previous_audit=dict(state='pass'),previous_private_backup=dict(state='private_readback_verified'),previous_termination=dict(exit_code=0,wall_timeout=False)).items():
  p=b/(key+'.json');p.write_text(json.dumps(value));cfg[key]=str(p)
 manifest=b/'synthetic_manifest.npz';manifest.write_bytes(b'synthetic_only');cfg.update(source_manifest=str(manifest),source_manifest_sha256=run.sha(manifest))
 reference=b/'reference.json';reference.write_text(json.dumps(dict(scores=dict(timemil=.98,gap=.97),cnn_reference=dict(full=.96,local=.95))));cfg.update(prior_held1_summary=str(reference),reference_held1_macro=.98)
 (b/'PROTOCOL.json').write_text(json.dumps(cfg));(b/'FROZEN_HASHES.json').write_text(json.dumps({'PROTOCOL.json':run.sha(b/'PROTOCOL.json')}));run.B=b
 with patch.object(run.np,'load',side_effect=Intercept('expected synthetic array boundary')) as loader:
  try:run.main()
  except Intercept:pass
  else:raise AssertionError('did not reach guarded boundary')
  assert loader.call_count==1 and loader.call_args.args[0]==str(manifest)
 state=json.loads((b/'run01/STATE.json').read_text());assert state['state']=='incomplete' and 'Intercept' in state['error']
 assert (b/'run01/IDENTITY.json').exists() and not list((b/'run01').glob('*/*.pt'))
 print(json.dumps(dict(state='pass',actual_startup_to_synthetic_np_load_exercised=True,real_UWB_values_or_labels_read=False,real_model_instantiations=0,real_optimization_steps=0)))
