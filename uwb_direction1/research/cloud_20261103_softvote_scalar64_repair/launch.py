from common import B,ready,read
from independent_supervisor import bounded_run
import sys
def main():
    p=ready();assert len(sys.argv)==2 and sys.argv[1] in ['validate','audit']
    phase=sys.argv[1]
    assert not (B/(phase+'_supervisor01')).exists()
    wall=p['wall_seconds'];cpu=p['CPU_seconds']
    if phase=='audit':
        t=read(__import__('pathlib').Path(p['operational_repair']['original_root'])/'audit_supervisor01/TERMINATION.json')
        wall-=t['supervisor_wall_seconds'];cpu-=__import__('math').ceil(t['child_cpu_seconds'])
    assert wall>0 and cpu>20
    r=bounded_run([sys.executable,'-u',str(B/(phase+'.py'))],B/(phase+'_supervisor01'),wall,cpu-20,cpu,p['RSS_stop_KiB'])
    assert r['exit_code']==0 and r['resource_stop_reason'] is None and r['cgroup_oom_kill_delta']==0
if __name__=='__main__':main()
