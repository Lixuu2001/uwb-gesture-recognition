"""Run the exact phase controller on isolated, explicitly generated filesystem fixtures."""
from pathlib import Path
import subprocess,sys,json,shutil,hashlib,os
import core as e
from stream_archive import build,verify
BASE=e.ROOT/(sys.argv[1] if len(sys.argv)>1 else 'generated_phase')
STUB_CORE='''from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,sort_keys=True));t.replace(p)
'''
STUB_WORKER='''from pathlib import Path
import sys,time
from core import ROOT,read,write,sha
assert (ROOT/'GENERATED_ONLY').exists()
held=int(sys.argv[sys.argv.index('--held')+1]);out=ROOT/'folds'/f'held{held}';mode=read(ROOT/'FIXTURE_CONFIG.json')['mode']
if mode=='busy':
 end=time.process_time()+.16
 while time.process_time()<end:pass
elif mode=='fail':raise RuntimeError('intentional generated child failure')
if Path(__file__).name=='fold_output_audit.py':
 f=out/'generated_dummy_payload.txt';f.write_text('generated-only, no UWB array or model fit')
 write(out/'PREDICTIONS_FROZEN.json',dict(all3_arms_prod64_fresh64_audited_unscored=True,files={str(f):sha(f)}))
'''
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def fixture(name,mode='normal',wall=100,CPU=100,disk=0):
    root=BASE/name;root.mkdir(parents=True,exist_ok=False)
    for n in ['phase_supervisor.py','supervisor.py']:shutil.copyfile(e.ROOT/n,root/n)
    (root/'core.py').write_text(STUB_CORE);(root/'GENERATED_ONLY').write_text('no real data or production admission')
    (root/'admission.py').write_text("from core import ROOT\ndef require_production(held):\n assert (ROOT/'GENERATED_ONLY').exists() and ROOT.parent.name.startswith('generated_phase') and 1<=held<=8\n")
    for n in ['worker.py','fold_output_audit.py']:(root/n).write_text(STUB_WORKER)
    dump(root/'PROTOCOL.json',dict(phase_wall_cap_seconds=wall,phase_CPU_cap_seconds=CPU,phase_RSS_cap_KiB=256*1024,disk_reserve_bytes=disk))
    dump(root/'FROZEN_HASHES.json',{});dump(root/'FIXTURE_CONFIG.json',dict(mode=mode));dump(root/'EXTERNAL_PHASE_CHARGES.json',[])
    return root
def run(root,h,success):
    attempt=len(list(root.glob('attempt*.stdout')))
    r=subprocess.run([sys.executable,str(root/'phase_supervisor.py'),str(h)],capture_output=True,text=True,timeout=30)
    (root/f'attempt{attempt:02}.stdout').write_text(r.stdout);(root/f'attempt{attempt:02}.stderr').write_text(r.stderr)
    if (r.returncode==0)!=success:raise AssertionError(dict(root=str(root),held=h,expected=success,returncode=r.returncode,stderr=r.stderr[-1200:]))
    return r
def release(root,h):
    import fold_backup as fb,register_backup_costs as rc
    fb.ROOT=root;rc.ROOT=root;fold=root/'folds'/f'held{h}'
    fb.archive(h);b=fold/'private_backup';m=e.read(b/'MANIFEST.json')
    for i,c in enumerate(m['chunks']):
        row=dict(**c,path=str(b/c['name']),file_id='generated'+str(h),raw_fetch=True,download_verified=True,include_base64=False,metadata=dict(shared=False,permissions=[dict(role='owner')]))
        dump(b/f'READBACK_{i:03}.json',row)
    fb.verify_raw(h)
    for n in ['BACKUP_BUILD_COST.json','BACKUP_VERIFY_COST.json']:dump(fold/n,dict(exit_code=0,user_CPU_seconds=.001,system_CPU_seconds=.001,parent_CPU_seconds=0.,generated_fixture_receipt_not_a_real_measurement=True))
    old=sys.argv;sys.argv=['generated-register',str(h)]
    try:rc.main()
    finally:sys.argv=old
    fb.finalize(h)
def main():
    BASE.mkdir(exist_ok=False);checks=[];root=fixture('normal8')
    run(root,1,True);first=e.read(root/'PHASE_LEDGER.json');assert first['completed']==[1] and first['state']=='waiting_private_backup';checks.append('first_fold_waits_unscored_for_private_backup')
    run(root,1,False);run(root,2,False);assert not(root/'folds/held2').exists();checks.append('duplicate_and_missing_private_handoff_blocked_before_next_worker')
    release(root,1);b=root/'folds/held1/private_backup';rows=e.read(b/'READBACK_INPUT.json');rows[0]['metadata']['shared']=True;dump(b/'READBACK_INPUT.json',rows)
    r=e.read(root/'folds/held1/PRIVATE_FOLD_RELEASE.json');r['readback_input_sha256']=e.sha(b/'READBACK_INPUT.json');dump(root/'folds/held1/PRIVATE_FOLD_RELEASE.json',r)
    run(root,2,False);assert not(root/'folds/held2').exists();checks.append('public_private_part_metadata_rejected')
    rows[0]['metadata']['shared']=False;dump(b/'READBACK_INPUT.json',rows);r['readback_input_sha256']=e.sha(b/'READBACK_INPUT.json');dump(root/'folds/held1/PRIVATE_FOLD_RELEASE.json',r)
    previous=e.read(root/'PHASE_LEDGER.json')['CPU']
    for h in range(2,9):
        run(root,h,True);now=e.read(root/'PHASE_LEDGER.json');assert now['CPU']>previous and now['external_CPU']>=.01*(h-1)-1e-12;previous=now['CPU'];release(root,h)
    final=e.read(root/'PHASE_LEDGER.json');assert final['completed']==list(range(1,9)) and final['state']=='all8_audited_unscored_waiting_private_backup'
    checks.append('all8_serial_three_stages_and_CPU_wall_external_ledger_cumulative');checks.append('actual_archive_verify_CPU_register_finalize_helpers_exercised_on_generated_folds')
    run(root,8,False);checks.append('completed_eighth_fold_cannot_repeat')
    root=fixture('CPU_cap',mode='busy',CPU=.35);run(root,1,False);r=e.read(root/'PHASE_LEDGER.json');assert 'fixed_phase_CPU_cap' in r['error'] and not(root/'folds/held1/execution/output_audit').exists();checks.append('cumulative_CPU_cap_across_successful_children')
    root=fixture('wall_cap',wall=.2);run(root,1,False);r=e.read(root/'PHASE_LEDGER.json');assert 'fixed_phase_wall_cap' in r['error'];checks.append('cumulative_wall_cap_before_next_stage')
    root=fixture('disk_cap',disk=10**18);run(root,1,False);assert not(root/'folds').exists();checks.append('disk_reserve_rejects_before_child_start')
    root=fixture('child_failure',mode='fail');run(root,1,False);r=e.read(root/'PHASE_LEDGER.json');assert r['state']=='execution_incomplete_explicit_recovery_required';checks.append('child_failure_retains_logs_and_refuses_automatic_restart');run(root,1,False)
    root=fixture('held_wrap');run(root,2,False);assert not(root/'folds').exists();checks.append('cannot_skip_first_held')
    sources={n:e.sha(e.ROOT/n) for n in ['phase_supervisor.py','supervisor.py','fold_output_audit.py','admission.py','fold_backup.py','register_backup_costs.py','generated_phase_checks.py']}
    e.write(e.ROOT/(BASE.name.upper()+'_CHECK.json'),dict(passed=True,checks=checks,sources=sources,exact_phase_controller_executed_in_isolated_fixtures=True,all8_dummy_folds=True,real_UWB_arrays_or_labels_read=False,real_production_admission=False,new_automations_or_Drive_objects_created=False,CPU_and_wall_do_not_establish_accuracy=True))
    print('PASS: exact phase controller8fold private handoffs, cumulative budgets, duplicate/privacy/failure guards; generated only')
if __name__=='__main__':main()
