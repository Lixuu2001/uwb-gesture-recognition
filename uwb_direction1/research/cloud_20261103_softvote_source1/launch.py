from common import B,ready
from independent_supervisor import bounded_run
import sys
def main():
    p=ready();assert len(sys.argv)==2 and sys.argv[1] in ['validate','audit']
    phase=sys.argv[1]
    assert not (B/(phase+'_supervisor01')).exists()
    r=bounded_run([sys.executable,'-u',str(B/(phase+'.py'))],B/(phase+'_supervisor01'),p['wall_seconds'],p['CPU_seconds']-20,p['CPU_seconds'],p['RSS_stop_KiB'])
    assert r['exit_code']==0 and r['resource_stop_reason'] is None and r['cgroup_oom_kill_delta']==0
if __name__=='__main__':main()
