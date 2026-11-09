"""Once-only detached launch after exact production admission, no task creation."""
import os,sys,time,subprocess,fcntl
from pathlib import Path
from core import ROOT,read,write,sha
from admission import require_production
from supervisor import proc
def main():
    held=int(sys.argv[1]);require_production(held)
    guard=(ROOT/'launch.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from phase_supervisor import verify_previous_private
    verify_previous_private(held)
    record=ROOT/'launches'/f'held{held}';record.mkdir(parents=True,exist_ok=False)
    if (ROOT/'folds'/f'held{held}').exists():raise PermissionError('Fold output already exists; explicit recovery required')
    if held==1:
        if (ROOT/'PHASE_LEDGER.json').exists() or (ROOT/'PRELAUNCH_ORIGIN.json').exists():raise PermissionError('Phase already started')
        write(ROOT/'PRELAUNCH_ORIGIN.json',dict(start_monotonic=time.monotonic(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_release_sha256=sha(ROOT/'PRODUCTION_RELEASE.json')))
    else:
        l=read(ROOT/'PHASE_LEDGER.json')
        if l['next_held']!=held or l['state']!='waiting_private_backup':raise PermissionError('No serial handoff')
    log=(record/'stdout.log').open('w')
    p=subprocess.Popen([sys.executable,str(ROOT/'phase_supervisor.py'),str(held)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    identity=proc(p.pid);write(record/'LAUNCH_IDENTITY.json',dict(**identity,held=held,source_commit=read(ROOT/'PRODUCTION_RELEASE.json')['source_commit'],new_automation_created=False,automatic_retries=0));log.close();guard.close()
    print(identity)
if __name__=='__main__':main()
