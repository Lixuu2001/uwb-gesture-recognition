"""New execution-only policy; never mutates/relaunches the closed native screen."""
from pathlib import Path
import json,hashlib,subprocess,sys,py_compile
B=Path(__file__).resolve().parent;A=Path('/workspace/uwb-drcif500-full-cost-20261102');S=Path('/workspace/uwb-drcif500-batched-spectrum-source-20261102')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,s):(B/n).write_text(s)
assert not (B/'COST_HASHES.json').exists() and not (B/'profile01').exists()
e=json.loads((S/'EQUIVALENCE_CHECK.json').read_bytes());assert e['all500_trees_intervals_feature_subsets_node_states_leaf_probabilities_exact'] and e['native_spectrum_full_vs64batch_exact_shapes']==[[131,3,90],[67,567,90]]
closed=json.loads((A/'VERIFIED_EXECUTION_CLOSURE_20261102.json').read_bytes());assert closed['no_old_screen_retry'] and not closed['full_fit_completed']
p=json.loads((A/'POLICY.json').read_bytes());p.update(method='same_DrCIF500_CIT20_separately_registered_batched_native_spectrum_execution',spectrum_execution_batch_size=64,original_spectrum_kernel_unmodified=True,prior_native_fullbatch_screen_closed=True,scientific_configuration_and_float64_precision_unchanged=True,old_fullbatch_screen_retry=False,execution_equivalence_source=str(S/'EQUIVALENCE_CHECK.json'))
put('POLICY.json',json.dumps(p,indent=2)+'\n')
for n in ['cost_math.py','test_cost_math.py','guard.py','run_screen.py']:put(n,(A/n).read_text())
s=(A/'profile_cost.py').read_text().replace('/workspace/uwb-drcif500-source-probe-20261102','/workspace/uwb-drcif500-batched-spectrum-source-20261102').replace('from native_drcif import','from native_drcif_batched import').replace('full4700_float64_promoted_native_DrCIF500_CIT20_fit','full4700_float64_DrCIF500_CIT20_batched_native_spectrum_fit').replace("state='complete_full4700_DrCIF500_CIT20_synthetic_cost'","state='complete_full4700_DrCIF500_CIT20_batched_native_spectrum_synthetic_cost',spectrum_batch_size=64,scientific_configuration_unchanged=True")
put('profile_cost.py',s)
s=(A/'verify_result.py').read_text().replace('independently_verified_full4700_DrCIF500_CIT20_cost','independently_verified_full4700_DrCIF500_CIT20_batched_spectrum_cost').replace("v.update(registration_commit=","v.update(spectrum_execution_batch_size=64,scientific_configuration_and_precision_unchanged=True,prior_native_screen_not_retried=True,registration_commit=")
put('verify_result.py',s)
put('PREREGISTRATION.md','''# Same DrCIF500/CIT20,float64: separately registered native spectrum batching

Previous fullbatch resource screen c8872803679ad80449dd62b862d95d5edcb39c80/worker89168/controller89167 stopped at12.925227639032528wall/12.911614CPU/12737228KiB by registered12GiBRSS supervisorTERM−15,cgroupOOMdelta0. It did not complete fit or produce accuracy/full12hcost evidence. Exact oldscreen andallfrozen source/tests/logs remainunchanged;independent terminal closure retained. No oldscreen retry and no family accuracy rejection. Native fullbatch float64FFT has large whole-dataset temporary arrays;memory pressure arose beforefullfit completion. Batching is an execution repair,not an accuracy-selected tree/feature/seed/precision/view rescue.

New adapter subclasses unchanged nativePeriodogramTransformer only to preallocate complete output and call originalsuper()._transform for64caseblocks,thenconcatenateinoriginalorder. Samefloat64 input,constantzero padding90to128,first64fullcomplexFFT magnitudes,allchannels,allintervals,all500CIT20/10of29/seed20261029/jobs4threading/BLAS1Numba1/nativeNaNpolicy unchanged. No rfft replacement,feature approximation,precision reduction,chunkwiselearning or changedFFTkernel. Original aeon sources andpreviousnativewrapper notmodified. Disclose new execution subclass,modernlibrary differences andnotoriginalJava parity/nooriginality claim. A prefreezeintegration attempt detected that publicfit resetsfactorytransformerassignments;its adapter/test/log are retained. The execution subclass now installsbatchedspectrum inside_fitafterresetbeforedelegatingunchangednativeforest;actualfittedtransformerclass is asserted. This repairs execution lifecycle without changing forest mathematics;newformal costwasnotstartedbyfailedattempt.

New numerical checks(not rerunning oldfrozen tests): originalwholebatch vs64casebatched spectrum bitexact for131x3x90 and67x567x90 includingconstantchannel;full500CIT models on24x3x90 haveallintervals/feature-subsets/nodes/leafprobabilities exact,probabilities/predictions exact,picklefullstate+predictions exact. End-to-end24casefit uses onechunk;componenttest explicitlycrosses64boundary/full567width. These fixtures are not all-input proof or full4700 throughput. Firstsourcefloat32NumbaCatch22fault andfloat64inputrepair remaintruthfullyretained;do not claimoriginalfloat32pipeline worked.

Fullpaper-scaleDrCIF: Middlehurstetal2021 HIVE-COTE2.0,DOI10.1007/s10994-021-06057-9;primarysection3.2/experimentalstructure/Table2configurationreview. Same nativeaeon1.6 sourceb694b1c03c0a753cc844dcb9ee357d77ca6bda6b,500CIT20,raw/diff/paddedperiodogram lengths90/89/64,567channels/requested79/78/67intervals,10attributes/representation,3min/.5maxinterval;native duplicate removal,per-representationattribute subsetting andprobabilityaverage preserved. Publishedbaseline transfer,not claimed original contribution/Java exactrandomdraws. Noaddednormalizer,sign/RMS/views unchanged. This exact new execution policy is frozen andremoteregisteredbeforeanyformal timing.

Afterallchangedfiles exactremote readback+branchheadverification,onefull4700syntheticfit float32fixture promotedfloat64,all500CIT20 with64caseoriginalFFT batches;fullfit includespromotion/alltransforms. No tinyN throughput extrapolation. Threefull500eval64/save/reload;alllearnedstate/probabilities/predictions exact. Modelserializedsize/SHA measuredfirstbynonstoringProtocol5sink;8fullmodels+64MiB>freeclosesstoragebeforewriting. Only disposableverified syntheticcheckpointcopies mayberemoved;noUWBmodel touched.

Same3600wall/CPUsoft14380hard14400/RSS12GiB/.2s+cgroupOOM screen. Conservative3fullfit+2*2200maxeval64/64+4maxsave+2maxload+180<=12h;4*estimate<=48CPUh;peak+2GiB<=14GiB;8models+64MiB<=free. Noauto retry or tree/threshold/interval/attribute/seed/CIT/vote/precision/viewfallback. Costpassneverauthorizesunregisteredrealfit. Failureclosesexactnewexecutionconfiguration andthenadvanceanotherliterature-supportedmethod.

Futureseparatelyregisteredsourceprotocol fixedtrain3–6/4700,val2/1000,diag1/1200/canonicalIDs;traininglabelsonly,noOOBselection;valclasses0..9/diag0..11 finitepresent-classmacro. Fullindependentnumeric audit+completeprivatechunk/allmemberrecovery beforeoncescore. Fixedreference.9866666666666667/minGain.005/tol1e-12. Accuracyfailclosesexactconfiguration,noseed/role/rescorerescue;passonlyseparatelyregisteredsource6confirmation. Users7–8labelsforbidden,historicalall8development exploratory,noindependentvalidation/new8foldgain/originality. Goalnotmet;automationenabled/original789/userREADME/currentvisibility/Driveowneronly/cloudonlypreserved.
''')
for path in B.glob('*.py'):py_compile.compile(str(path),doraise=True)
q=subprocess.run([sys.executable,str(B/'test_cost_math.py')],capture_output=True,text=True);put('COST_TESTS.log',q.stdout+q.stderr);assert q.returncode==0
deps=json.loads((A/'DEPENDENCIES.json').read_bytes())['files']
for path in S.iterdir():
 if path.is_file():deps[str(path)]=sha(path)
for n in ['verify_execution_closure_20261102.py','VERIFIED_EXECUTION_CLOSURE_20261102.json','supervisor01/TERMINATION.json','COST_HASHES.json']:
 path=A/n;deps[str(path)]=sha(path)
for path,h in deps.items():assert sha(Path(path))==h,path
put('DEPENDENCIES.json',json.dumps(dict(files=deps,scope='Priorpinnedscientificclosure+alloriginalsources+newoutputequivalentexecutionsubclass+closednativeRSSproof;notwholeOS'),indent=2)+'\n')
names=['prepare.py','POLICY.json','PREREGISTRATION.md','cost_math.py','test_cost_math.py','guard.py','run_screen.py','profile_cost.py','verify_result.py','COST_TESTS.log','DEPENDENCIES.json'];put('COST_HASHES.json',json.dumps({n:sha(B/n) for n in names},indent=2)+'\n')
from guard import ready
try:ready();raise AssertionError('Missingremote marker mustblock')
except FileNotFoundError:pass
put('PRE_ACCESS_GUARD_CHECK.json',json.dumps(dict(missing_remote_marker_blocks=True,cost_files=len(names),dependencies=len(deps),formal_timing_started=False,actual_UWB_values_labels_fit=False),indent=2)+'\n');print(json.dumps(dict(cost_files=len(names),dependencies=len(deps),cost_hashes_sha256=sha(B/'COST_HASHES.json'))))
