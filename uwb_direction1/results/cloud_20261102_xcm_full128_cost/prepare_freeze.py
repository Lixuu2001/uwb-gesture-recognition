from pathlib import Path
import json,hashlib,subprocess,sys,shutil,datetime
B=Path(__file__).resolve().parent;S=Path('/workspace/uwb-xcm-source-probe-20261102')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (B/'COST_HASHES.json').exists()
    checked=json.loads((S/'SOURCE_CHECK.json').read_bytes());assert checked['parameters']==2621070
    policy=dict(method='full_official_XCM128_BasicMotions_released_single_configuration_transfer',source_commit='3d70a544a8c0ae11d10f71968d5a5cd7e57d5928',parameters=2621070,trainable_parameters=2620302,filters=128,window_fraction=.2,kernel_length=18,timepoints=90,variables=567,classes=12,epochs=100,batch_size=32,seed=20261029,threads=4,loss='unweighted_categorical_crossentropy',optimizer=checked['optimizer'],checkpoint_selection='future_val2_micro_accuracy_latest_tie_over100epochs',original_final_epoch_and_5fold_grid_not_reproduced=True,native_backend='Keras3.11.3_Torch2.15.0_CPU',historical_TensorFlow_parity=False,synthetic_only=True,actual_UWB_values_labels_fit_authorized=False,warm_training_updates=1,timed_updates=3,timed_evals=3,timed_full_checkpoint_saves=3,timed_full_checkpoint_loads=3,full_pair_cost_formula='2*(14700*max_train+3485*max_eval+101*max_save+2*max_load)+180',CPU_cost_formula='4*estimated_wall',wall_budget_seconds=43200,CPU_budget_seconds=172800,RSS_stop_KiB=12582912,map_margin_KiB=2097152,memory_budget_KiB=14680064,checkpoint_workspace_formula='8*max_checkpoint_bytes+67108864',profile_wall_seconds=1200,profile_CPU_soft_seconds=1180,profile_CPU_hard_seconds=1200,source_train_users=[3,4,5,6],validation_user=2,diagnostic_user=1,fixed_reference=.9866666666666667,minimum_gain=.005,tolerance=1e-12,users7_8_labels_forbidden=True,all_exploratory=True,no_seed_width_window_epoch_batch_backend_or_view_rescue=True,novelty_claim=False)
    (B/'POLICY.json').write_text(json.dumps(policy,indent=2)+'\n')
    source=Path('/workspace/uwb-moderntcn-fullwave-cost-screen-20261102/run_screen.py').read_bytes().decode().replace('profile_screen.py','profile_cost.py')
    (B/'run_screen.py').write_text(source)
    review=dict(paper='Fauvel et al.2021 XCM,Mathematics9(23):3137',DOI='10.3390/math9233137',paper_url='https://arxiv.org/pdf/2009.04796',reviewed='Original paper sections3.1architecture/4.2algorithm/4.3hyperparameter protocol,pp6–10;official fullmodel/main/config/README/LICENSE',mechanism='Parallel pervariable 2D and jointvariable 1D branches plus final1D/GAP',selection='One released example configuration window.2/B32/100E/default128;not paper fivefold window/batch grid and not selected on UWB labels',source_import_repairs='KevinFauvel/XCM guessed404 resolved paper official XAIseries/XCM;models/XCM.py404 resolved main.py import models/xcm.py;no data restoration implied',compatibility='Unchanged function AST and layer defaults bound to native Keras3 imports instead of removed keras.layers.convolutional;Torch backend, no TensorFlow installation/parity claim',pre_freeze_failure='First RNG snapshot constructed Keras global generator after capturing Python state;synthetic continuation detected one Python RNG index drift. Attempts01/02 source+failurelogs retained;now generator materialized before all RNG capture,fullweights/Adam/metrics/RNG nextupdate exact. Topological branch order asserted by names,not source ordering.',actual_UWB_values_labels_fit=False)
    (S/'PAPER_AND_RUNTIME_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
    text='''# Full XCM128 released configuration: preregistered synthetic cost

Fauvel et al.2021 XCM,DOI10.3390/math9233137,official XAIseries/XCM commit3d70a544a8c0ae11d10f71968d5a5cd7e57d5928. Original architecture and evaluation sections plus complete author code reviewed. Apache2.0 originals byte preserved. Native Keras3/Torch binding retains complete author function AST,128filters/2D+1Dparallelbranches/reductions/final1D/GAP/softmax,window.2→kernel18,100E/B32/defaultAdam.001/betas.9/.999/eps1e-7/unweightedCE. This is one released example configuration,not authors' fivefold window/batch search or historical TensorFlow numerical reproduction. Input metadata90×567×1/classes12,projectseed20261029/CPU4 are explicit transfer parameters. User7–8 labels forbidden; bounded latestfourrecords XCM tokensearch foundzero,not an exhaustive repository proof.

Source synthetic checks passed independent toy graph forward/inputgradients with sameweights and even padding,allnativeAdam/metrics/RNG nextupdate snapshot replay,and serialized fullKerascheckpoint prediction/Adam nextupdate. Fullsymbolicshape[N,90,567,1]→[N,12],2621070params/2620302trainable;no full numerical targetshape forward or actualUWB values/labels/fit yet. Preparation caught and repaired first Keras global RNG initialization snapshot order beforefreeze;both failed logs/sources retained. No TensorFlow dependency or proxy stub and no shrinking to rescue.

After allfiles exact remote readback,one fullB32 warm update thenthree fulltraining/eval/checkpointsave+load timings using seeded syntheticinput. Full .keras weights/Adam retained in three local files;serialized predictions+optimizer exact. Conservative full100E source4700train/147updatesperE plus entirefullreplay,each3485evalbatches (100*32val+69held+216rawaudit),101fullsaves/2loads plus180seconds. Gate <=12wallhours/48CPUhours,peak+2GiBmapmargin<=14GiB,and8fullcheckpointsets+64MiB<=currentremainingdisk. Three timings usemaxima,not median. Independent supervisor1200wall/1180soft1200hardCPU/12GiBRSS/.2spoll/cgroupOOM evidence;stopped/error not scientificnegative,no retry or seed/width/window/batch/epochs/views/backend fallback.

Cost pass alone does not authorize real UWBfit. It requires separately implemented/preregistered runner/fullreplay/private preservation,source3–6train/2val/1diag canonicalIDs,signedperviewRMS input,validationcheckpointselection latesttie,once diagnostic aftercompleteaudit/private recovery. FixedTimeMIL.9866666666666667,+.005/tol1e-12. Fail exactconfigclosed,nogrid/rescore;pass separately source6 confirmation. All historically exposed eightusers exploratory,no independentvalidation or noveltyclaim. Published XCM contribution belongs authors;accuracy transfer evidence remains separate. RootREADME/usercleanup/original789/currentrepo visibility/Driveprivate unchanged.
'''
    (B/'PREREGISTRATION.md').write_text(text)
    p=subprocess.run([sys.executable,str(B/'test_cost_math.py')],capture_output=True,text=True)
    (B/'COST_TESTS.log').write_text(p.stdout+p.stderr);assert p.returncode==0
    deps={}
    for src in ['/workspace/uwb-moderntcn-fullwave-cost-screen-20261102/DEPENDENCIES.json','/workspace/uwb-tlenet-formal-cost-20261102/DEPENDENCIES.json']:
        for name,h in json.loads(Path(src).read_bytes())['files'].items():
            path=Path(name)
            if not name.startswith('/workspace/') or '/uwb-resnet-keras-screen-20261030/deps/' in name:
                assert sha(path)==h;deps[name]=h
    for p in sorted(S.rglob('*')):
        if p.is_file() and '__pycache__' not in str(p):deps[str(p)]=sha(p)
    (B/'DEPENDENCIES.json').write_text(json.dumps(dict(scope='Selected runtime Python/Torch/NumPy and isolated Keras source;not wholeOS/dependencybinarysnapshot',files=[dict(path=k,sha256=v) for k,v in sorted(deps.items())]),indent=2)+'\n')
    frozen=['cost_math.py','test_cost_math.py','guard.py','profile_cost.py','run_screen.py','verify_result.py','POLICY.json','PREREGISTRATION.md','DEPENDENCIES.json','COST_TESTS.log']
    (B/'COST_HASHES.json').write_text(json.dumps(dict(files=[dict(path=str(B/n),sha256=sha(B/n)) for n in frozen]),indent=2)+'\n')
    try:
        from guard import ready
        ready()
        raise AssertionError('Missingmarker must block')
    except FileNotFoundError:pass
    (B/'PRE_ACCESS_GUARD_CHECK.json').write_text(json.dumps(dict(missing_remote_marker_blocks=True,cost_files=len(frozen),dependencies=len(deps),full_target_shape_numerical_timing_started=False,actual_UWB_values_labels_fit=False),indent=2)+'\n')
    print(json.dumps(dict(cost_files=len(frozen),dependencies=len(deps),hash=sha(B/'COST_HASHES.json'))))
if __name__=='__main__':main()
