"""Operational read-only verifier harness; clean pinned-Keras executor at exit.

The preregistered verifier and active training files stay byte-identical. This
only closes CallbackList's idle executor after the original verification, before
Python tears down its queue module. It redirects the repeated proof to a new
file and preserves the original first-insurance proof/log unchanged.
"""
import gc,hashlib,json,runpy
from pathlib import Path
from unittest.mock import patch
import run_pilot
B=Path(__file__).resolve().parent
if __name__=='__main__':
 p=B/'verify_first_update.py';assert run_pilot.sha(p)=='adf16fdcedece115e5151090899c94496fd56b88cbf7ad1e7d75014d5227d55a'
 original_write=run_pilot.write
 def redirected(target,value):
  assert Path(target)==B/'FIRST_UPDATE_VERIFIED.json'
  original_write(B/'FIRST_UPDATE_CLEAN_EXIT_VERIFIED.json',value)
 with patch.object(run_pilot,'write',redirected):namespace=runpy.run_path(str(p),run_name='__main__')
 cb=namespace['cb']
 if cb._executor is not None:cb._executor.shutdown(wait=True);cb._executor=None
 namespace.clear();del cb;gc.collect()
 proof=run_pilot.read(B/'FIRST_UPDATE_CLEAN_EXIT_VERIFIED.json');assert proof['state']=='pass'
 original_write(B/'VERIFIER_CLEANUP_NOTE.json',dict(state='read_only_verifier_operational_cleanup_completed',original_attempt_exit_code=0,original_warning='Keras CallbackList.__del__ invoked executor.shutdown after Python queue module teardown; unraisable AttributeError after passing proof, retained FIRST_UPDATE_VERIFIER.log',repair='Execute byte-identical frozen verifier; explicitly shut down idle CallbackList executor before interpreter teardown. Repeated proof redirected to separate file; original insurance proof/checkpoints and active scientific training/code remain unchanged.',frozen_verifier_sha256=run_pilot.sha(p),new_harness_sha256=run_pilot.sha(__file__),real_values_labels_or_new_forward_fit=False,active_training_or_frozen_code_modified=False))
