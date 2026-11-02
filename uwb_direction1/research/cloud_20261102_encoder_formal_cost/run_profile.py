"""Single bounded synthetic profile; refuses existing outputs or absent receipt."""
from pathlib import Path
import importlib.util,json,sys
from guard import ready
B=Path(__file__).resolve().parent
def main():
 ready();assert not (B/'profile01').exists() and not (B/'supervisor01').exists()
 spec=importlib.util.spec_from_file_location('Encoder_cost_supervisor',B/'independent_supervisor.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 r=s.bounded_run([sys.executable,'-u',str(B/'profile_cost.py')],B/'supervisor01',620,590,600);print(json.dumps(r));raise SystemExit(0 if r['exit_code']==0 else 1)
if __name__=='__main__':main()
