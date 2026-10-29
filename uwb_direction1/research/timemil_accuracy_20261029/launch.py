"""Explicit one-shot CPU/wall supervisor for the prospectively frozen pilot."""
import importlib.util,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent
def main():
 assert sys.argv[1:]==['--run']
 assert not (B/'run01').exists(),'inspect existing artifacts; never restart a trial automatically'
 p=Path('/workspace/uwb-multirocket-20261029/launch.py')
 spec=importlib.util.spec_from_file_location('existing_bounded_supervisor',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.bounded_run([sys.executable,'-u',str(B/'run_pilot.py'),'--run'],B/'supervisor01',43100,43000,43100)
 state=B/'run01/STATE.json'
 if state.exists():
  v=json.loads(state.read_text())
  if result['exit_code']!=0 and v.get('state')!='complete':v.update(state='incomplete',pid=None,supervisor=result,scientific_negative=False);state.write_text(json.dumps(v,indent=2))
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
if __name__=='__main__':main()
