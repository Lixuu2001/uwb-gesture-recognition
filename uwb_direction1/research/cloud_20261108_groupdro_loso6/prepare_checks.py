"""Run the preregistered native tests for all six subject-role sets, plus fixed cost and toy stop audits."""
from pathlib import Path
import sys,os,subprocess,time,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'shared'))
from common import read,write,ready

def main():
    ready();assert not(P/'SYNTHETIC_SUMMARY.json').exists() and not(P/'PHASE_IDENTITY.json').exists()
    started=time.monotonic();rows=[]
    try:ready(True)
    except (FileNotFoundError,AssertionError):pass
    else:raise AssertionError('Missing new release must prevent real fitting')
    assert not(P/'PHASE_IDENTITY.json').exists()
    subprocess.run([sys.executable,str(P/'test_lifecycle.py')],check=True)
    assert read(P/'LIFECYCLE_TEST_RESULT.json')['state'].startswith('pass_')
    for h in range(1,7):
        fold=P/f'folds/held{h}';env=dict(os.environ,UWB_GROUPDRO_LOSO_FOLD_ROOT=str(fold))
        r=subprocess.run([sys.executable,str(P/'shared/controller.py'),'synthetic'],env=env,check=True)
        a=read(fold/'SYNTHETIC_RESULT.json');assert a['state']=='pass' and all(a['checks'].values());rows.append(dict(held=h,result=a,termination=read(fold/'synthetic_execution/TERMINATION.json')))
    from loso_roles_metrics import synthetic_checks
    role_metrics=synthetic_checks()
    r=subprocess.run([sys.executable,str(P/'test_safety.py')],capture_output=True,text=True,check=True);safety=json.loads(r.stdout)
    write(P/'SAFETY_RESULT.json',safety)
    subprocess.run([sys.executable,str(P/'shared/controller.py'),'profile'],check=True)
    gate=read(P/'RESOURCE_GATE.json')
    result=dict(all_passed=all(all(r['result']['checks'].values()) for r in rows) and safety['state']=='pass' and role_metrics['state']=='pass',perfold=rows,role_metric_checks=role_metrics,safety=safety,full_production_lifecycle_fixture=read(P/'LIFECYCLE_TEST_RESULT.json'),missing_release_blocks_fitting=True,resource_gate_passed=gate['gate_passed'],wall=time.monotonic()-started,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False)
    write(P/'SYNTHETIC_SUMMARY.json',result);print(json.dumps(dict(all_passed=result['all_passed'],resource_gate_passed=gate['gate_passed'],reserve_seconds=gate['reserve_seconds'],ratio=gate['GroupDRO_native_ratio'],wall=result['wall'])))
if __name__=='__main__':main()
