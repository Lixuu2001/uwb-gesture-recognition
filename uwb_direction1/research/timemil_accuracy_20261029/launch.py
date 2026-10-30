"""Explicit one-shot CPU/wall supervisor for the prospectively frozen pilot."""
import importlib.util,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent
def main():
 assert sys.argv[1:]==['--run']
 cfg=json.loads((B/'PROTOCOL.json').read_text());run_name=cfg.get('run_directory','run01');supervisor_name=cfg.get('supervisor_directory','supervisor01')
 assert not (B/run_name).exists(),'inspect existing artifacts; never restart a trial automatically'
 p=Path('/workspace/uwb-multirocket-20261029/launch.py')
 spec=importlib.util.spec_from_file_location('existing_bounded_supervisor',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.bounded_run([sys.executable,'-u',str(B/'run_pilot.py'),'--run'],B/supervisor_name,43100,43000,43100)
 prior=cfg.get('prior_predata_attempt_budget',{});cumulative=dict(current=result,prior_predata_attempt_budget=prior,cumulative_wall_seconds=result['supervisor_wall_seconds']+prior.get('wall_seconds',0),cumulative_cpu_seconds=result['child_cpu_seconds']+prior.get('cpu_seconds',0),scientific_negative=False)
 (B/'CUMULATIVE_TERMINATION.json').write_text(json.dumps(cumulative,indent=2));assert cumulative['cumulative_wall_seconds']<=43200 and cumulative['cumulative_cpu_seconds']<=43200
 state=B/run_name/'STATE.json'
 if state.exists():
  v=json.loads(state.read_text())
  if result['exit_code']!=0 and v.get('state')!='complete':v.update(state='incomplete',pid=None,supervisor=result,scientific_negative=False);state.write_text(json.dumps(v,indent=2))
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
if __name__=='__main__':main()
