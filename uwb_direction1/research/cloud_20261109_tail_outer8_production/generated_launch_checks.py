"""Exercise exact launch helper only inside generated marker fixtures."""
from pathlib import Path
import sys,subprocess,time,shutil
import core as e
import generated_phase_checks as g
def main():
    # Versioned isolated fixtures retain the failed v1 evidence.
    g.BASE=e.ROOT/'generated_phase_launch_v2';g.BASE.mkdir(exist_ok=False)
    root=g.fixture('once');g.dump(root/'PRODUCTION_RELEASE.json',dict(source_commit='generated_fixture_no_real_release'));shutil.copyfile(e.ROOT/'launch_fold.py',root/'launch_fold.py')
    def run(h,ok):
        r=subprocess.run([sys.executable,str(root/'launch_fold.py'),str(h)],capture_output=True,text=True,timeout=20)
        (root/f'launch{h}_{time.time_ns()}.log').write_text(r.stdout+r.stderr)
        assert (r.returncode==0)==ok
    run(1,True);run(1,False)
    start=time.monotonic()
    while not(root/'PHASE_LEDGER.json').exists() or e.read(root/'PHASE_LEDGER.json')['state']=='fold_running':
        assert time.monotonic()-start<15;time.sleep(.1)
    assert e.read(root/'PHASE_LEDGER.json')['completed']==[1]
    run(2,False);assert not(root/'launches/held2').exists() and not(root/'folds/held2').exists()
    e.write(e.ROOT/'GENERATED_LAUNCH_CHECK.json',dict(passed=True,exact_detached_launch_in_isolated_fixture=True,duplicate_launch_rejected=True,missing_private_handoff_rejected_before_launch_marker=True,real_UWB_data_read=False,real_phase_started=False,sources={n:e.sha(e.ROOT/n) for n in ['launch_fold.py','phase_supervisor.py','generated_launch_checks.py']}))
    print('PASS: exact once-only detached launch and private preflight guard; generated only')
if __name__=='__main__':main()
