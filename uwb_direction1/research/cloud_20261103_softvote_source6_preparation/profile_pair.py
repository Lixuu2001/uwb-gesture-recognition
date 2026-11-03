"""Bounded concurrent CPU1+4 synthetic preflight. Not a real fit launcher."""
from pathlib import Path
import sys,json,hashlib,subprocess,time,resource
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def events():return {x.split()[0]:int(x.split()[1]) for x in Path('/sys/fs/cgroup/memory.events').read_text().splitlines()}
def main():
    marker=json.loads((B/'REMOTE_PREREGISTRATION.json').read_text())
    assert marker['all_changed_remote_bytes_exact'] and marker['branch_head_verified']
    assert marker['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    for n,h in json.loads((B/'FROZEN_HASHES.json').read_text()).items():assert sha(B/n)==h
    for n,h in json.loads((B/'DEPENDENCIES.json').read_text())['files'].items():assert sha(n)==h
    p=json.loads((B/'SOURCE6_METHOD_LOCK.json').read_text());out=B/'profile01';out.mkdir(exist_ok=False)
    before=resource.getrusage(resource.RUSAGE_CHILDREN);ev0=events();start=time.monotonic();children=[];logs=[];peak=0;reason=None
    def limit():resource.setrlimit(resource.RLIMIT_CPU,(2980,3000))
    try:
        for arm in ['timemil','shuffle']:
            log=(out/(arm+'.log')).open('x');logs.append(log)
            child=subprocess.Popen([sys.executable,'-u',str(B/'profile_worker.py'),arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit);children.append(child)
        (out/'PIDS.json').write_text(json.dumps(dict(pids=[x.pid for x in children]))+'\n')
        while any(x.poll() is None for x in children):
            rss=0
            for x in children:
                try:rss+=sum(int(line.split()[1]) for line in Path(f'/proc/{x.pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
                except FileNotFoundError:pass
            peak=max(peak,rss)
            if peak>p['combined_RSS_stop_KiB']:reason='combined_RSS_limit'
            if time.monotonic()-start>p['profile_wall_seconds']:reason='wall_limit'
            if any(x.poll() not in [None,0] for x in children):reason='worker_failure'
            if reason:break
            time.sleep(.2)
    finally:
        for x in children:
            if x.poll() is None:
                x.terminate()
                try:x.wait(timeout=3)
                except subprocess.TimeoutExpired:x.kill();x.wait()
        for log in logs:log.close()
    after=resource.getrusage(resource.RUSAGE_CHILDREN);ev1=events();codes=[x.returncode for x in children]
    result=dict(state='complete_concurrent_full_batch_synthetic_preflight',exit_codes=codes,resource_stop_reason=reason,wall_seconds=time.monotonic()-start,total_child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,peak_combined_RSS_KiB=peak,cgroup_OOM_kill_delta=ev1['oom_kill']-ev0['oom_kill'],new_UWB_fit_started=False)
    result['resource_gate_passed']=len(codes)==2 and codes==[0,0] and reason is None and result['cgroup_OOM_kill_delta']==0
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['resource_gate_passed']:raise SystemExit(1)
if __name__=='__main__':main()
