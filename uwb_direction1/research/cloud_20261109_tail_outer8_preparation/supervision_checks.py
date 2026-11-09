"""Generated caps/duplicate-run/production-rejection checks; no UWB labels."""
import sys
from pathlib import Path
import core as e
from supervisor import bounded
from role_data import RealData
def main():
    result=bounded('generated_cap_success',[sys.executable,'-c','print("generated child complete")'],dict(wall=10,CPU=10,RSS=256*1024))
    assert result['exit_code']==0
    blocked=[]
    try:bounded('generated_cap_stop',[sys.executable,'-c','while True: pass'],dict(wall=.2,CPU=.2,RSS=256*1024))
    except RuntimeError:blocked.append('busy_generated_child_resource_cap')
    else:raise AssertionError('resource cap did not stop child')
    stopped=e.read(e.ROOT/'execution/generated_cap_stop/TERMINATION.json');assert stopped['resource_stop']=='fixed_resource_cap' and stopped['exit_code']!=0
    try:bounded('generated_cap_success',[sys.executable,'-c','print("duplicate")'],dict(wall=10,CPU=10,RSS=256*1024))
    except FileExistsError:blocked.append('duplicate_execution_directory')
    else:raise AssertionError('duplicate allowed')
    try:RealData(1)
    except PermissionError:blocked.append('production_labels_without_release')
    else:raise AssertionError('unadmitted label access')
    from admission import require_production
    try:require_production(1)
    except FileNotFoundError:blocked.append('fullworker_without_production_release')
    else:raise AssertionError('production admitted by prep')
    e.write(e.ROOT/'SUPERVISION_CHECK.json',dict(passed=True,blocked=blocked,generated_bounded_child_complete=result,generated_forced_stop=stopped,real_UWB_labels_read=False,production_started=False,phase_supervisor_syntax_only_no_live_phase_exercised=True,scope='bounded child/resource/duplicate and fail-closed production checks; full phase/private handoff still needs generated end-to-end coverage before production admission'))
    print('PASS: bounded generated children, cap termination, duplicate rejection and no-production guards')
if __name__=='__main__':main()
