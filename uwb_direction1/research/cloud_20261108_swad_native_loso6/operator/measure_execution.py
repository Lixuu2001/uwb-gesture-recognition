"""Record one explicit subprocess execution without external time(1); no retry."""
from pathlib import Path
import sys,json,time,subprocess,resource,datetime
output=Path(sys.argv[1]);assert not output.exists();command=sys.argv[2:];assert command
start=time.monotonic();parent=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
child=subprocess.Popen(command);exit_code=child.wait();after=resource.getrusage(resource.RUSAGE_CHILDREN)
r=dict(command=command,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,user_CPU_seconds=after.ru_utime-before.ru_utime,system_CPU_seconds=after.ru_stime-before.ru_stime,parent_CPU_seconds=time.process_time()-parent,max_RSS_KiB=after.ru_maxrss,exit_code=exit_code,measurement='stdlib resource.RUSAGE_CHILDREN for waited process family; no external time utility',automatic_retries=0)
output.write_text(json.dumps(r,indent=2)+'\n');sys.exit(exit_code)
