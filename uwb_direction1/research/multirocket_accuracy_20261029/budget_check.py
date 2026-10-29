"""Controlled wall/CPU timeout checks without UWB file access."""
import json,sys
from pathlib import Path
from launch import bounded_run
B=Path(__file__).resolve().parent
out=B/'synthetic_budget';out.mkdir(exist_ok=False)
wall=bounded_run([sys.executable,'-c','import pathlib,time;pathlib.Path("'+str(out/'partial.txt')+'").write_text("partial");time.sleep(20)'],out/'wall',.25,3,4)
cpu=bounded_run([sys.executable,'-c','while True: pass'],out/'cpu',5,1,2)
assert wall['wall_timeout'] and wall['exit_code']!=0 and wall['supervisor_wall_seconds']<4
assert cpu['exit_code']!=0 and cpu['child_cpu_seconds']<3 and not cpu['wall_timeout']
assert (out/'partial.txt').read_text()=='partial'
assert (out/'wall/TERMINATION.json').exists() and (out/'cpu/TERMINATION.json').exists()
r=dict(state='pass',synthetic_only=True,uwb_arrays_or_labels_read=False,
    wall_timeout_terminal_receipt=True,cpu_limit_terminal_receipt=True,
    partial_artifact_retained=True,scientific_negative=False,wall_case=wall,cpu_case=cpu)
(B/'BUDGET_CHECK.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
