"""Bind the exact verified release and start one registered real CORAL controller."""
from pathlib import Path
import sys,json,subprocess,os,time
S=Path('/workspace/uwb-coral-gap64-20261106');M=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import read,write,sha,ready
from controller import identity
def main():
    proof=read(M/'release_PUBLICATION_VERIFIED.json');assert proof['branch_head_verified'] and proof['force'] is False and proof['exact_readback_verified']==len(read(M/'release_PUBLICATION.json')['files'])
    assert not (S/'REAL_RELEASE.json').exists() and not(S/'run').exists() and not(S/'CONTROLLER_IDENTITY.json').exists()
    q=read(S/'RELEASE_INTENT.json');assert q['initial_archive_raw_allmember_verified'] and q['frozen_sha256']==sha(S/'FROZEN_HASHES.json')
    actual=dict(q,state='exact_actual_release_verified_initial_private_ready_before_single_real_launch',branch_head_verified=True,commit=proof['commit'],release_exact_readback_files=proof['exact_readback_verified'],release_force=False)
    write(S/'REAL_RELEASE.json',actual);ready(True)
    with (M/'LAUNCH_REQUEST.json').open('x') as f:json.dump(dict(exact_release_commit=proof['commit'],frozen_sha256=q['frozen_sha256'],initial_archive_sha256=q['initial_archive_sha256'],observed_unix=time.time(),maximum_new_controllers=1),f,indent=2)
    p=subprocess.Popen([sys.executable,str(S/'controller.py'),'--run'],stdin=subprocess.DEVNULL,stdout=(S/'controller.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(pid=p.pid,identity=identity(p.pid),release_commit=proof['commit'],frozen_sha256=q['frozen_sha256'],real_training_process_started=True,first_actual_update_native_verification_pending=True,diagnostic_labels_read=False,users7_8_labels_used=False)
    write(S/'LAUNCH_STARTED.json',result);write(M/'LAUNCH_STARTED.json',result);print(json.dumps(result))
if __name__=='__main__':main()
