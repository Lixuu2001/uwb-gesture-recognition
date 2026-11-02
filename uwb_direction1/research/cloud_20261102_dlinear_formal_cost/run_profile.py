from pathlib import Path
import importlib.util,sys,json
from guard import ready
B=Path(__file__).resolve().parent
def main():
 ready();assert not (B/'profile01').exists() and not (B/'supervisor01').exists()
 s=importlib.util.spec_from_file_location('DLinear_cost_supervisor',B/'independent_supervisor.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 r=m.bounded_run([sys.executable,'-u',str(B/'profile_cost.py')],B/'supervisor01',620,590,600);print(json.dumps(r));raise SystemExit(0 if r['exit_code']==0 else 1)
if __name__=='__main__':main()
