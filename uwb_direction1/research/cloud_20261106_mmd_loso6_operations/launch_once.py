"""Operational single launch, exact published release only; never restarts or scores."""
from pathlib import Path
import os,sys,json,fcntl,subprocess,time
P=Path('/workspace/uwb-mmd-loso6-20261106');M=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity

def main():
    lock=(M/'launch_once.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    v=read(M/'release_PUBLICATION_VERIFIED.json');r=read(P/'REAL_RELEASE.json')
    assert r['commit']==v['commit'] and v['branch_head_verified'] and v['force'] is False
    assert r['frozen_sha256']==sha(P/'FROZEN_HASHES.json')
    assert not any((P/n).exists() for n in ['LAUNCH_REQUEST.json','PHASE_IDENTITY.json','STATE.json'])
    assert all(not (P/f'folds/held{h}/run').exists() for h in range(2,7))
    targets={str(P/'phase_controller.py'),str(P/'shared/controller.py'),str(P/'shared/worker.py')}
    for d in Path('/proc').iterdir():
        if not d.name.isdecimal():continue
        try:args=set(d.joinpath('cmdline').read_bytes().decode().split('\0'))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not args.intersection(targets),'A matching controller/worker already exists'
    original=Path('/workspace/uwb-mmd-gap64-20261106')
    for n in ['CONTROLLER_IDENTITY.json']:
        q=original/n
        if q.exists():
            old=read(q);now=identity(old['pid']);assert not now or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
    ready(True)
    marker=dict(state='one_launch_requested_after_exact_release_private_initial_and_all5_native_tests',release_commit=r['commit'],frozen_sha256=r['frozen_sha256'],unix=time.time(),old_fold1_unscored=True,no_individual_test_gate=True)
    with (P/'LAUNCH_REQUEST.json').open('x') as f:json.dump(marker,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    try:
        with (P/'phase_controller.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(P/'phase_controller.py'),'--run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        actual=identity(child.pid);assert actual and actual['state']!='Z';write(P/'LAUNCH_RECEIPT.json',dict(state='phase_controller_process_created_not_yet_proof_of_model_training',identity=actual,release_commit=r['commit'],unix=time.time(),no_restart=True))
        print(json.dumps(read(P/'LAUNCH_RECEIPT.json')))
    except BaseException as ex:write(P/'LAUNCH_FAILURE.json',dict(error=repr(ex),automatic_retry=False,launch_marker_retained=True));raise
if __name__=='__main__':main()
