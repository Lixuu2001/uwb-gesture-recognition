"""One bounded child at a time; no retry, duplicate fold or automatic admission."""
from pathlib import Path
import os,sys,time,signal,subprocess,fcntl,resource
from core import ROOT,read,write
def proc(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text();q=s[s.rfind(')')+2:].split();raw=Path(f'/proc/{pid}/status').read_text()
        return dict(pid=pid,startticks=int(q[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=q[0],CPU=(int(q[11])+int(q[12]))/os.sysconf('SC_CLK_TCK'),RSS_KiB=next((int(x.split()[1]) for x in raw.splitlines() if x.startswith('VmRSS:')),0))
    except FileNotFoundError:return None
def bounded(name,command,caps):
    out=ROOT/'execution'/name;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);reason=None
    with (out/'stdout.log').open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(out/'IDENTITY.json',proc(p.pid))
        while p.poll() is None:
            st=proc(p.pid)
            if time.monotonic()-start>caps['wall'] or (st and st['CPU']>caps['CPU']) or (st and st['RSS_KiB']>caps['RSS']):
                reason='fixed_resource_cap';os.killpg(p.pid,signal.SIGTERM);break
            time.sleep(.5)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(exit_code=p.returncode,resource_stop=reason,wall=time.monotonic()-start,CPU=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,max_RSS_KiB=after.ru_maxrss,automatic_retries=0)
    write(out/'TERMINATION.json',result)
    if p.returncode or reason:raise RuntimeError('Execution failed; preserve logs and repair explicitly: '+name)
    return result
def main():
    p=read(ROOT/'PROTOCOL.json');mode=sys.argv[1]
    if mode=='generated':bounded('generated',[sys.executable,str(ROOT/'generated_checks.py')],dict(wall=180,CPU=180,RSS=6*1024**2));return
    if mode=='probe':
        from admission import require_probe
        require_probe();bounded('probe',[sys.executable,str(ROOT/'resource_probe.py')],dict(wall=p['probe_wall_cap_seconds'],CPU=p['probe_CPU_cap_seconds'],RSS=p['probe_RSS_cap_KiB']));return
    raise PermissionError('Full phase supervisor admission and private interfold checkpoint handshake pending; no production launcher in this preparation')
if __name__=='__main__':main()
