"""Execute the released original scorer once after verified full private preservation publication."""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess,signal
M=Path(__file__).resolve().parent;S=Path('/workspace/uwb-sd-gap64-20261105')
sys.path.insert(0,str(S))
from common import read,write,sha
from controller import identity
def main():
    proof=read(M/'full_archive_PUBLICATION_VERIFIED.json');assert proof['branch_head_verified'] and proof['force'] is False and proof['exact_readback_verified']==len(read(M/'full_archive_PUBLICATION.json')['files'])
    assert read(S/'full_backup/READBACK_VERIFIED.json')['complete_modelset_verified']
    assert sha(S/'operator/finalize_once.py')==read(S/'REAL_RELEASE.json')['once_scorer_sha256']=='9321449c2a3924b1059fc0c051352eb88d06497a3ee393a76076f75188b9f528'
    assert not(S/'SUMMARY.json').exists() and not(S/'FINALIZE_ONCE.lock').exists()
    prior=read(S/'CONTROLLER_IDENTITY.json');now=identity(prior['pid']);assert now is None or now['state']=='Z' or any(now[k]!=prior[k] for k in ['pid','startticks','boot_id'])
    marker=M/'SCORING_STARTED.json'
    with marker.open('x') as f:json.dump(dict(original_scorer_sha256=sha(S/'operator/finalize_once.py'),full_private_publication=proof['commit'],observed_unix=time.time(),maximum_original_scorer_invocations=1),f,indent=2)
    p=subprocess.Popen([sys.executable,str(S/'operator/finalize_once.py')],stdin=subprocess.DEVNULL,stdout=(M/'SCORING.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    write(M/'SCORING_PID.json',dict(pid=p.pid,identity=identity(p.pid)));start=time.monotonic();reason=None
    while p.poll() is None:
        who=identity(p.pid);rss=0
        if who and who['state']!='Z':
            try:
                raw=Path(f'/proc/{p.pid}/status').read_text();rss=next((int(line.split()[1]) for line in raw.splitlines() if line.startswith('VmRSS:')),0)
            except FileNotFoundError:rss=0
        if time.monotonic()-start>600 or (who and who['cpu']>600) or rss>12582912:
            reason='bounded_original_once_scorer_execution_cap';os.killpg(p.pid,signal.SIGTERM);break
        time.sleep(.5)
    p.wait();result=dict(exit_code=p.returncode,resource_stop_reason=reason,wall_seconds=time.monotonic()-start,original_scorer_invocations=1,original_scorer_unchanged=True,original_lock_exists=(S/'FINALIZE_ONCE.lock').exists(),summary_exists=(S/'SUMMARY.json').exists(),never_automatically_retry=True)
    write(M/'SCORING_TERMINATION.json',result);assert p.returncode==0 and reason is None,result;print(json.dumps(result))
if __name__=='__main__':main()
