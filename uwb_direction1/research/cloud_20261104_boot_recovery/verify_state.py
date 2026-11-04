from pathlib import Path
import sys,json
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT));from common import read,write,sha
import engine as e
arm=sys.argv[1];e.setup(arm);from journal import load,StepLog
out=ROOT/'folds/held3'/arm;state,p=load(out);m,o,g=e.make(arm);e.restore(state,m,o,g)
restored=e.capture(m,o,g,state['epoch'],state['updates'],state['best_key'],state['best_model'],state['best_epoch']);assert e.semantic(restored)==e.semantic(state)
log=StepLog(out/'steps.jsonl',state['updates'],p['prefix_bytes'],p['prefix_sha256']);rows=log.original;log.close()
result=dict(state='native_model_optimizer_sampler_Torch_Python_Numpy_RNG_restore_exact',arm=arm,committed_epoch=state['epoch'],committed_updates=state['updates'],original_log_rows=rows,uncommitted_suffix_rows=rows-state['updates'],semantic_sha256=e.semantic(state),checkpoint_sha256=sha(out/p['slot']),new_training_update_performed=False,diagnostic_labels_read=False)
write(Path('/workspace/uwb-source6-boot-recovery-20261104')/(arm+'_NATIVE_RESTORE_VERIFIED.json'),result);print(json.dumps(result))
