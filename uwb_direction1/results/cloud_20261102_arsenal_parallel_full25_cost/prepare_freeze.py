"""New execution policy;never edits/relaunches the closed CPU1 profile."""
from pathlib import Path
import json,hashlib,subprocess,sys
B=Path(__file__).resolve().parent;S=Path('/workspace/uwb-arsenal-parallel-source-probe-20261102');O=Path('/workspace/uwb-arsenal-full25-cost-20261102')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (B/'COST_HASHES.json').exists();e=json.loads((S/'EQUIVALENCE_CHECK.json').read_bytes());assert e['all_kernel_scaler_readout_alpha_LOOaccuracy_weight_arrays_exact'] and e['native_parallel_jobs']==4
    p=json.loads((O/'POLICY.json').read_bytes());p.update(method='same_full_native_Arsenal25x2000_new_jobs4_execution_policy',native_parallel_jobs=4,threads=4,BLAS_threads_per_member=1,Numba_threads_per_Rocket_member=1,native_NumBa_parallel_lock_preserved=True,profile_wall_seconds=3600,profile_CPU_soft_seconds=14380,profile_CPU_hard_seconds=14400,CPU_budget_seconds=172800,previous_CPU1_screen_incomplete_not_full12h_rejection=True,no_old_screen_retry=True,scientific_model_seed_scaler_alpha_vote_unchanged=True,synthetic_serial_parallel_equivalence_proof=str(S/'EQUIVALENCE_CHECK.json'))
    (B/'POLICY.json').write_text(json.dumps(p,indent=2)+'\n')
    for n in ['cost_math.py','test_cost_math.py','guard.py','profile_cost.py','run_screen.py','verify_result.py']:
        s=(O/n).read_bytes().decode().replace('/workspace/uwb-arsenal-source-probe-20261102','/workspace/uwb-arsenal-parallel-source-probe-20261102')
        s=s.replace("==(25,2000,20261029,1,4700)","==(25,2000,20261029,4,4700)")
        if n=='test_cost_math.py':s=s.replace('threads=1','threads=4')
        if n=='profile_cost.py':s=s.replace('full4700_native_25x2000_Arsenal_fit','full4700_native_25x2000_Arsenal_jobs4_fit').replace("r=select(p,fit,ev,saves,loads,rss,max(sizes),free);","r=select(p,fit,ev,saves,loads,rss,max(sizes),free);")
        if n=='run_screen.py':s=s.replace('Bounded1CPU','Bounded4CPU').replace('(1780,1800)','(14380,14400)').replace('>1800','>3600').replace('cpu_soft_seconds=1780,cpu_hard_seconds=1800','cpu_soft_seconds=14380,cpu_hard_seconds=14400')
        if n=='guard.py':
            a=s.index('    previous=json.loads(');z=s.index("    return json.loads",a)
            s=s[:a]+"    previous=json.loads(Path('/workspace/uwb-arsenal-full25-cost-20261102/VERIFIED_EXECUTION_CLOSURE_20261102.json').read_bytes())\n    assert previous['no_old_profile_retry'] and not previous['actual_UWB_values_labels_fit']\n"+s[z:]
        if n=='verify_result.py':s=s.replace('independently_verified_full4700_native_Arsenal25x2000_cost','independently_verified_full4700_native_Arsenal25x2000_jobs4_cost').replace("next_action='separately_register_fixed_real_pilot_runner_full_auditor_private_preservation'","native_parallel_jobs=4,scientific_configuration_unchanged=True,next_action='separately_register_fixed_real_jobs4_pilot_runner_full_auditor_private_preservation'")
        (B/n).write_text(s)
    # Add four-core conservative CPU accounting without altering old gate code.
    pth=B/'cost_math.py';s=pth.read_text().replace('resource_gate_passed=estimate<=43200 and memory<=14680064 and disk<=free,','resource_gate_passed=estimate<=43200 and 4*estimate<=172800 and memory<=14680064 and disk<=free,estimated_CPU_seconds=4*estimate,');pth.write_text(s)
    registration='''# Same full Arsenal25x2000: separately registered CPU4 execution policy

The preceding CPU1/1800wall synthetic screen at a6de5e2f579fdc06f2c36b1692e4c6425e2c8fa9 ended by supervisorTERM at1800.2048931729514wall/1799.577349CPUseconds,peak3852048KiB,cgroupOOMdelta0,without fullnative4700fit completion. Its exactscreen is closed,allsource/tests/logs/partialstateunchanged,no retry. It supplies no full12h estimate oraccuracy result;do not reject the scientificmodel merely because a30minute costwindow was too short. Independentexecutionclosure is retained. No realUWB values/labels/fit occurred.

This is an execution adjustment of the same published Arsenal baseline,NOT a new method or accuracy-selected rescue. Native aeon1.6 sourceb694b1c03c0a753cc844dcb9ee357d77ca6bda6b remainsunchanged;full25x2000/50000kernels/4000features/member,seed20261029,rawsignedRMS/N567/T90,class12,normaliseTrue,with_meanFalse/unweightedRidgeCVfixed10alphas/trainingLOOaccuracy directweights/contract0 unchanged. Only native Arsenal n_jobs=4 uses current4CPUquota. EachRockettransform andBLAS uses1thread. OriginalNumba parallel lock remains,so transforms mayserialize;no guaranteed4xspeedup,no thread/runtime/methodgrid basedon UWBlabels. HIVE-COTE2.0 paper Algorithm4/Table2/source citations and modernlibrary-weightedvote vs papermajority difference remainexplicit,not originalJava parity/novelty.

New syntheticexecutiontest(not rerunning oldfrozen tests) fittedfull25x2000 jobs1 andjobs4 onN24/T90/M567/12classfixedsinusoid+noise;allkernels/25scalers/readouts/classes/alphas/LOOaccuracyweights/proba/nativecanonicaltiepredictions bitexact. Jobs4stagedprobas/picklereload exact;Numba-lock retained. This is fixture equivalence,not proofallfloatingpoint kernels identical for everypossibledata. OriginalindependentLOO/scaler/vote tests remaintheir prior scope. No actualUWB data used.

Afterallchangedfiles exactremote readback/headverification,newfull4700synthetic end-to-end nativejobs4fit once;threefull25eval64/save/reload measurements andprobability exactchecks. Onehour profilewall3600/CPUsoft14380hard14400/12GiBexactchildRSS/.2s/cgroupOOM independentmonitor;4corequota4/memory16GiB. No smaller member/kernel/feature/scaler/alpha/seed/vote/view/precision fallback. Conservative3*fullfit+2*2200maxeval64/64+4maxsave+2maxload+180 <=12wallhours;4*estimate<=48CPUhours;peak+2GiBmaps<=14GiB;8checkpoints+64MiB<=remainingdisk. FullN avoids smallN cubic/SVD extrapolation. This newresourcepolicy is preregistered beforeformal timing;noactualUWBfit authorized bycostpass alone.

Ifsuccessfulcostpasses: separatelyadapt/freeze/register realjobs4runner/auditor/privatepreservation,train3–6/4700,val2/1000,diag1/1200,canonicalIDs. Nativealphas/CVweights trainingonly;val/diaglabels sealeduntilfullnumericaudit+completeprivatechunk/allmemberverification thenonescore. FixedTimeMIL .9866666666666667/+ .005/tol1e-12,knownvalsupport0..9/diag0..11. Failedaccuracy closes exactscientificconfiguration,noseed/kernel/member/alpha/vote/scaler/role rescue/rescore;pass onlyseparatelyregisteredsource6confirmation. Users7–8labels forbidden,alleight historicaldevelopment exploratory,noindependentvalidation/8foldgain/originalcontribution yet. Goalnotmet. OldCPU1realrunner draft remainsunstarted/unfrozen andmustnotbelaunchedforjobs4withoutnewregistration. ExistingHybrid95.5833%closed/private991members,original789/rootREADME/usercleanup/currentvisibility/Driveprivacy/automationenabled preserved.
'''
    (B/'PREREGISTRATION.md').write_text(registration)
    q=subprocess.run([sys.executable,str(B/'test_cost_math.py')],capture_output=True,text=True);(B/'COST_TESTS.log').write_text(q.stdout+q.stderr);assert q.returncode==0
    deps=json.loads((S/'DEPENDENCIES.json').read_bytes())['files']
    for pth in S.iterdir():
        if pth.is_file():deps[str(pth)]=sha(pth)
    for name in ['VERIFIED_EXECUTION_CLOSURE_20261102.json','COST_HASHES.json','supervisor01/TERMINATION.json']:
        deps[str(O/name)]=sha(O/name)
    for name,digest in deps.items():assert sha(Path(name))==digest,name
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(files=deps,scope='Nativeimported scientific Python/binaryclosure/newjobs4executionproof/closedCPU1evidence;notwholeOS'),indent=2)+'\n')
    names=['cost_math.py','test_cost_math.py','guard.py','profile_cost.py','run_screen.py','verify_result.py','POLICY.json','PREREGISTRATION.md','DEPENDENCIES.json','COST_TESTS.log']
    (B/'COST_HASHES.json').write_text(json.dumps({n:sha(B/n) for n in names},indent=2)+'\n')
    from guard import ready
    try:ready();raise AssertionError('Missingremote marker mustblock')
    except FileNotFoundError:pass
    (B/'PRE_ACCESS_GUARD_CHECK.json').write_text(json.dumps(dict(missing_remote_marker_blocks=True,cost_files=10,dependencies=len(deps),actual_UWB_values_labels_fit=False,formal_timing_started=False),indent=2)+'\n');print(json.dumps(dict(cost_files=10,dependencies=len(deps),cost_hashes_sha256=sha(B/'COST_HASHES.json'))))
if __name__=='__main__':main()
