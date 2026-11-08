"""Run one unchanged registered operator with durable full-process cost/failure evidence."""
from pathlib import Path
import sys,json,subprocess,time,resource
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import read,ready,write,sha

def main():
    stage=sys.argv[1];assert stage in ['inference','cohort'];ready(False)
    release=read(P/'FULL_PRIVATE_RELEASE.json');assert release['branch_head_verified'] and release['force'] is False
    assert read(P/'full_backup/READBACK_VERIFIED.json')['complete_modelset_verified']
    if stage=='inference':
        assert not(P/'INFERENCE_ONCE.lock').exists() and not(P/'INFERENCE_COST.json').exists()
        script=P/'operator/inference_cost.py';args=['--run']
    else:
        assert (P/'INFERENCE_COST.json').exists() and not(P/'FINALIZE_ONCE.lock').exists() and not(P/'SUMMARY.json').exists()
        script=P/'operator/finalize_cohort_once.py';args=[];assert sha(script)==release['once_scorer_sha256']
    assert sha(script)==read(P/'FROZEN_HASHES.json')[str(script.relative_to(P))]
    stdout=P/(stage.upper()+'_OPERATOR_STDOUT.json');stderr=P/(stage.upper()+'_OPERATOR_STDERR.log');cost=P/(stage.upper()+'_EXECUTION_COST.json')
    assert not cost.exists();before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();unix=time.time()
    with stdout.open('xb') as out,stderr.open('xb') as err:
        result=subprocess.run([sys.executable,str(script),*args],stdout=out,stderr=err)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt=dict(state='unchanged_registered_operator_process_completed' if result.returncode==0 else 'operator_failed_preserve_once_lock_and_partial_artifacts_no_automatic_retry',stage=stage,exit_code=result.returncode,started_unix=unix,ended_unix=time.time(),wall_seconds=time.monotonic()-started,CPU_user_seconds=after.ru_utime-before.ru_utime,CPU_system_seconds=after.ru_stime-before.ru_stime,CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,max_RSS_KiB=after.ru_maxrss,operator_sha256=sha(script),original_once_protocol_unchanged=True,scientific_frozen45_unchanged=True,new_fit_started=False)
    write(cost,receipt);print(json.dumps(receipt));assert result.returncode==0,'Inspect saved cost/stderr/partial outputs; never clear a consumed once lock'

if __name__=='__main__':main()
