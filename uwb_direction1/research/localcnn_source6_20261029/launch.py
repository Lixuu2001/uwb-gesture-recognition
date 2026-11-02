"""One-shot bounded launch; retain terminal receipt even after hard CPU kill."""
import importlib.util,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent
def main():
 assert sys.argv[1:]==['--run'] and not (B/'run01').exists()
 spec=importlib.util.spec_from_file_location('bounded_existing',Path('/workspace/uwb-multirocket-20261029/launch.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result=m.bounded_run([sys.executable,'-u',str(B/'train.py'),'--run'],B/'supervisor01',43100,129400,129500)
 p=B/'run01/STATE.json'
 if p.exists():
  state=json.loads(p.read_text())
  if result['exit_code']!=0 and state.get('state')!='complete':state.update(state='incomplete',pid=None,supervisor=result,scientific_negative=False);p.write_text(json.dumps(state,indent=2))
 print(json.dumps(result));raise SystemExit(0 if result['exit_code']==0 else 1)
if __name__=='__main__':main()
