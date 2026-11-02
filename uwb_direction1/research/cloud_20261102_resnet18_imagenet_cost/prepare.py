from pathlib import Path
import json,hashlib,subprocess,sys,py_compile
B=Path(__file__).resolve().parent;S=Path('/workspace/uwb-resnet18-imagenet-source-20261102');A=Path('/workspace/uwb-drcif500-batched-spectrum-cost-20261102')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,s):(B/n).write_text(s)
assert not (B/'COST_HASHES.json').exists() and not (B/'profile01').exists()
e=json.loads((S/'INTEGRATION_CHECK.json').read_bytes());assert e['all_official1000class_weights_loaded_strict'] and e['independent_functional_residual_driver_logits_exact'] and e['linearSVC72case_12class_converged_and_restored_exact']
closed=json.loads((A/'VERIFIED_EXECUTION_CLOSURE_20261102.json').read_bytes());assert closed['no_further_same_configuration_resource_rescue'] and not closed['accuracy_measured']
weights=json.loads((S/'WEIGHTS_SOURCE.json').read_bytes())
p=dict(method='full_official_ResNet18_ImageNet1K_V1_frozen512_encoder_LinearSVC_C1_transfer',vision_commit='7a9db90e9aadea3e3e1fec35969601167e9efd4f',weights_sha256=weights['sha256'],encoder_bytes=weights['bytes'],full_original_parameters=11689512,frozen_encoder_parameters=11176512,embedding_width=512,encoder_BN_eval=True,input_shape=[3,90,189],image_shape=[3,224,224],channel_mapping='orderedthreephysicalradars_asthreeinputchannels',per_record_view_minmax_floor=1e-8,resize='bilinear_fullimage_align_cornersFalse_antialiasFalse_nocrop',ImageNet_mean=[.485,.456,.406],ImageNet_std=[.229,.224,.225],batch_size=16,seed=20261029,threads=4,head='LinearSVC',C=1.,loss='squared_hinge',penalty='l2',dual=False,tol=1e-4,max_iter=10000,fit_intercept=True,intercept_scaling=1.,class_weight=None,no_head_grid=True,synthetic_train_size=4700,synthetic_fixture_shape=[4700,3,90,189],budget_seconds=43200,CPU_budget_seconds=172800,RSS_stop_KiB=12582912,map_margin_KiB=2097152,memory_limit_KiB=14680064,profile_wall_seconds=3600,profile_CPU_soft_seconds=14380,profile_CPU_hard_seconds=14400,formula='3*full4700_encoder_plus_SVMfit+2*2200*maxeval64/64+4*maxheadsave+2*maxheadload+180',checkpoint_storage='2*shared_official_encoder_checkpoint+8*headcheckpoint+64MiB<=remainingdisk;frozenencoderweightsnotduplicatedperhead',actual_UWB_values_labels_fit_authorized=False,eventual_source_train_users=[3,4,5,6],eventual_validation_user=2,eventual_diagnostic_user=1,fixed_reference=.9866666666666667,minimum_gain=.005,tolerance=1e-12,users7_8_labels_forbidden=True,all_exploratory=True,novelty_claim=False,no_weight_architecture_view_preprocessing_C_seed_or_role_rescue=True)
put('POLICY.json',json.dumps(p,indent=2)+'\n')
for n in ['guard.py','run_screen.py']:put(n,(A/n).read_text())
put('cost_math.py','''import math
def select(p,fit,evaluation,saves,loads,rss,size,free):
 assert (p['embedding_width'],p['C'],p['seed'],p['threads'],p['batch_size'],p['synthetic_train_size'])==(512,1.,20261029,4,16,4700)
 assert len(evaluation)==len(saves)==len(loads)==3 and all(math.isfinite(v) and v>0 for v in [fit,rss]+evaluation+saves+loads)
 estimate=3*fit+2*2200*max(evaluation)/64+4*max(saves)+2*max(loads)+180;memory=rss+2097152;disk=2*p['encoder_bytes']+8*size+67108864
 return dict(resource_gate_passed=estimate<=43200 and 4*estimate<=172800 and memory<=14680064 and disk<=free,estimated_seconds=estimate,estimated_hours=estimate/3600,estimated_CPU_seconds=4*estimate,estimated_max_RSS_KiB=memory,required_checkpoint_workspace_bytes=disk,head_checkpoint_bytes=size,shared_encoder_checkpoint_bytes=p['encoder_bytes'],workspace_free_bytes=free,accuracy_measured=False,real_UWB_values_labels_fit=False,real_pilot_started=False,no_weight_architecture_view_preprocessing_C_seed_or_role_rescue=True)
''')
put('test_cost_math.py','''import unittest
from cost_math import select
P=dict(embedding_width=512,C=1.,seed=20261029,threads=4,batch_size=16,synthetic_train_size=4700,encoder_bytes=46830571)
class Test(unittest.TestCase):
 def test_pass_shared_encoder(self):self.assertTrue(select(P,100,[1]*3,[1]*3,[1]*3,1000,1000,200000000)['resource_gate_passed'])
 def test_large_cost(self):self.assertFalse(select(P,15000,[1]*3,[1]*3,[1]*3,1000,1000,200000000)['resource_gate_passed'])
 def test_memory_and_exact_shared_storage(self):
  self.assertFalse(select(P,100,[1]*3,[1]*3,[1]*3,13000000,1000,200000000)['resource_gate_passed'])
  self.assertFalse(select(P,100,[1]*3,[1]*3,[1]*3,1000,1000,100000000)['resource_gate_passed'])
 def test_scope_and_nonfinite(self):
  with self.assertRaises(AssertionError):select(dict(P,C=2),100,[1]*3,[1]*3,[1]*3,1000,1000,200000000)
  with self.assertRaises(AssertionError):select(P,float('nan'),[1]*3,[1]*3,[1]*3,1000,1000,200000000)
if __name__=='__main__':unittest.main(verbosity=2)
''')
put('profile_cost.py','''from pathlib import Path
import sys,time,json,resource,pickle,shutil,warnings
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
 (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,3,90,189]),indent=2)+'\\n')
 sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-source-20261102');from native_resnet18 import np,make,extract,make_head,encoder_digest
 from sklearn.exceptions import ConvergenceWarning
 class Fixture:
  def __init__(self,n):self.shape=(n,3,90,189)
  def __len__(self):return self.shape[0]
  def __getitem__(self,k):
   assert isinstance(k,slice);a,b,s=k.indices(len(self));assert s==1
   return np.stack([np.random.default_rng(20261029+i).standard_normal((3,90,189),dtype=np.float32) for i in range(a,b)])
 def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\\n')
 x=Fixture(4700);y=np.arange(4700)%12;state('full4700_complete_pretrained_ResNet18_frozen_feature_extraction_then_fixed_C1_SVM');t=time.monotonic();m=make();before=encoder_digest(m);features,hashes=extract(m,x,16);head=make_head()
 with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);head.fit(features,y)
 fit=time.monotonic()-t;assert head.n_iter_<10000 and features.shape==(4700,512) and encoder_digest(m)==before
 ev=[];saves=[];loads=[];sizes=[]
 for i in range(3):
  state('timing_complete_frozen_encoder_and_SVM_predict64',index=i);t=time.monotonic();evalf,_=extract(m,x[i*64:(i+1)*64],16);scores=head.decision_function(evalf);pred=head.predict(evalf);ev.append(time.monotonic()-t);assert np.isfinite(scores).all();np.testing.assert_array_equal(pred,head.classes_[scores.argmax(1)])
 for i in range(3):
  state('timing_weight_bound_head_checkpoint_save_reload',index=i);payload=dict(encoder_weights_sha256=p['weights_sha256'],encoder_state_sha256=before,head=head);path=P/f'head_synthetic_{i}.pkl';t=time.monotonic()
  with path.open('xb') as f:pickle.dump(payload,f,protocol=5)
  saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic()
  with path.open('rb') as f:q=pickle.load(f)
  loads.append(time.monotonic()-t);assert q['encoder_weights_sha256']==p['weights_sha256'] and q['encoder_state_sha256']==encoder_digest(m);np.testing.assert_array_equal(q['head'].coef_,head.coef_);np.testing.assert_array_equal(q['head'].intercept_,head.intercept_);np.testing.assert_array_equal(q['head'].decision_function(evalf),scores);np.testing.assert_array_equal(q['head'].predict(evalf),pred)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,fit,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_pretrained_ResNet18_frozen512_LinearSVC_C1_synthetic_cost',full_native_fit_seconds=fit,evaluation64_seconds=ev,complete_head_save_seconds=saves,complete_head_reload_seconds=loads,peak_RSS_KiB=rss,head_checkpoint_sizes=sizes,encoder_weights_sha256=p['weights_sha256'],encoder_state_sha256=before,full_encoder_frozen_unchanged=True,full_encoder_layers=[2,2,2,2],features=512,full_training_cases=4700,head_n_iter=int(head.n_iter_),head_converged=True,loaded_head_coefficients_decisions_predictions_exact=True,training_feature_batch_hashes=hashes,actual_UWB_values_labels_fit=False)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
''')
put('verify_result.py','''import json
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes());assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,3,90,189]
 assert r['features']==512 and r['full_training_cases']==4700 and r['full_encoder_layers']==[2,2,2,2] and r['encoder_weights_sha256']==p['weights_sha256'] and r['full_encoder_frozen_unchanged'] and r['head_converged'] and r['head_n_iter']<10000 and r['loaded_head_coefficients_decisions_predictions_exact'] and not r['actual_UWB_values_labels_fit']
 for i,n in enumerate(r['head_checkpoint_sizes']):assert (B/f'profile01/head_synthetic_{i}.pkl').stat().st_size==n
 v=dict(state='independently_verified_full4700_official_ResNet18_frozen512_LinearSVC_C1_cost',**select(p,r['full_native_fit_seconds'],r['evaluation64_seconds'],r['complete_head_save_seconds'],r['complete_head_reload_seconds'],max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB']),max(r['head_checkpoint_sizes']),r['workspace_free_bytes']),registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],real_fit_authorized=False)
 v['next_action']='separately_register_real_runner_full_audit_private_encoder_head_backup' if v['resource_gate_passed'] else 'close_exact_frozen_ResNet18_ImageNet_C1_configuration_no_rescue';(B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
if __name__=='__main__':main()
''')
put('PREREGISTRATION.md','''# Full official frozen ImageNet ResNet18 + fixed linear SVM: synthetic cost first

BothDrCIF500CIT20float64 resource screens areclosed: nativefullbatchRSS12737228KiB/12.925227639s,output-equivalent64casebatchspectrumRSS12636572KiB/16.554265165s,TERM−15/noOOM/no completefit/accuracy. Oldfrozenfiles/tests/logs unchanged,no further sameconfigurationrescues. QUANTfull539217/200trees onehourincompletealso closed,noaccuracy/familyrejection. Nextcandidatechosenfromprimaryliterature andnotfromnewheld-labelqueries.

PrimaryUWBtransferbasis: Cho/Park2022 Classificationofhumanbodymotionsusinganultra-widebandpulseradar,DOI10.3233/THC-212827,publisherprimarymethods3.1–3.3/Table1 checked (notallPDF/figures). PretrainedResNet18/224x224radarimagefine-tuningispriorart,bodymotionnotourhandgesturetask. FrozenCNNfeature+linearSVMbasisRazavianetal2014 CNNFeaturesOff-the-Shelf,DOI10.1109/CVPRW.2014.131,primaryabstract/methodexcerpts. We use fullmodernofficialResNet18,notthatpaper's4096dimensionnetworkorexactradarpaperfine-tuning. AdditionalgesturepaperSensors2021DOI10.3390/s21010259 indexedprimarybutdirectPMCchallenge/MDPI429,notfulltextreview. Publishedtransferbaseline,notoriginalityclaimormodulecombinationinnovation.

Officialpytorch/visionv0.24.0commit7a9db90e9aadea3e3e1fec35969601167e9efd4f,resnet.pyGitblob47067ec83175a97cc6f6a8721b342128a434d440. ExactoriginalASTconv3x3/conv1x1/BasicBlock/Bottleneck/ResNet classes used because torchvisionbinarypackageabsent;onlyAPIusageloggingstubbed,nolayersormathrewritten. Full18layerarchitecture[2,2,2,2],11689512originalparameters,strictfull1000classstate_dictload. OfficialImageNet1K_V1defaultcheckpoint46830571bytes/SHAf37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec,publicdownloadverifiedpublisherprefixandfullSHA. Replaceonly1000wayfinalfcwithidentityforpooled512representation;11176512encoderparametersallfrozen/BatchNormeval. FixedLinearSVC(C1,L2,squaredhinge,dualFalse,tol1e-4,max_iter10000,interceptTrue,scaling1,noclassweight,seed20261029),requireconvergence,noCgrid/scaler/weights/backbone/headrescue.

Projectradaradapterisexplicit:threeorderedphysicalradarsas3inputchannels,each90timeby189rangeimage. ExistingcanonicalRMSmaps iffutureactualpilotregistered. Perrecord/viewminmaxfloor1e-8,fullmapbilinearresizeto224x224/align_cornersFalse/antialiasFalse;nocrop/augmentation. ImageNetmean(.485,.456,.406)/std(.229,.224,.225). This differsfromnativeImageNetresize256/centercrop224to preservealltime/rangeextent;notRGBnaturalcolorororiginalradarpaperimageprocessingclaim. FreezeBN/encoder,no trainingdata-normalizationfit. Standardpublishedtransferbaseline,notnewmechanism. Fullarchitecture/fullofficialweights,notold1DfromscratchResNet/shrunkversion/frozenLTRfeatures.

SyntheticindependentFP64minmax/bilinear/referencechecksrtol=atol2e-5prechosen;fulloriginalASTforward vsfunctionalresidualdriver1000logits bitexact;allfrozenencoderstateunchanged/repeatedsamebatchfeatures/picklerestoreexact;72case12classfixedSVMconvergedandrestorecoefficients/predictsexact. SourcecopymaterializationextraterminalnewlinecaughtbyGitblobguardbeforemodelconstruction,exactUTF8rewritefixedit;badcopy/logretained,noauthoralgorithmchange. NoUWBvalues/labels/fit. FixedCPU4float32deterministicalgorithms/batch16;actualrolesmustusefixed16caseorder,notassumebatchsize-independencefromfixtures.

Afterallchangedfilesexactremotereadback/headverification,onefull4700synthetic90x189threeviewencoder+SVMfit (not smallNextrapolation);row-addressablefixturekeepsfullinputsmallRAMwithoutalteringall4700cases. Fullfit includesmodelload/pixeladapter/allfeatureextraction/headfit. Threefullencoder+SVMeval64measurements,threeencoder-weightSHA-boundheadcheckpoints save/reload withcoef/intercept/decision/predictsexact;publicfrozenencoderweightsareone shareddependency,never duplicatedperhead. Actualstudyprivatebackupmustincludesharedencoderonce+fullheads/IDs/predictions/logs/source,notheadonlymislabelledfullmodel.

3600wall/CPU14380soft14400hard/RSS12GiB/.2s/cgroupOOM;conservative3fullfit+2*2200maxeval64/64+4maxheadsave+2maxheadload+180<=12h;4*estimate<=48CPUh;peak+2GiB<=14GiB;2sharedencodercheckpointcopies+8heads+64MiB<=free. FailureclosesexactfrozenResNet18V1/3views224/minmax/C1/seed20261029policy,nosweep. Costpassalone doesnotstartactualUWBfit:separatelyfreeze/registerproducer/fullauditor/privatebackup/oncefinalizer first.

Futureprotocolfixedtrain3–6/4700,val2/1000,diag1/1200canonicalIDs. Encoderfrozen/eval/sharedweights,classifiertraininglabels3–6only. No testlabeltuning. Supportedmacroval0..9/diag0..11,TimeMILreference.9866666666666667/+ .005/tol1e-12. Fullindependentfit/semanticnumericreplay andcompleteprivatearchive/allmemberrecoverybeforeoncescore. Failclosesexactconfiguration;passonlyseparatelyregisteredsource6confirmation. Users7–8labelsforbidden,all8historicaldevelopmentexploratory,noindependentvalidation/new8foldgain/originalcontribution yet.Goalnotmet;automationenabled/original789/userREADME/currentvisibility/Driveprivate/cloudonlypreserved.
''')
for path in B.glob('*.py'):py_compile.compile(str(path),doraise=True)
q=subprocess.run([sys.executable,str(B/'test_cost_math.py')],capture_output=True,text=True);put('COST_TESTS.log',q.stdout+q.stderr);assert q.returncode==0
deps=json.loads((A/'DEPENDENCIES.json').read_bytes())['files']
for path in S.rglob('*'):
 if path.is_file() and '__pycache__' not in path.parts:deps[str(path)]=sha(path)
sys.path.insert(0,str(S));import native_resnet18
for mod in tuple(sys.modules.values()):
 name=getattr(mod,'__file__',None)
 if name and Path(name).is_file() and '__pycache__' not in str(name):deps[str(Path(name).resolve())]=sha(Path(name))
for n in ['verify_execution_closure_20261102.py','VERIFIED_EXECUTION_CLOSURE_20261102.json','supervisor01/TERMINATION.json']:
 path=A/n;deps[str(path)]=sha(path)
for path,h in deps.items():assert sha(Path(path))==h,path
put('DEPENDENCIES.json',json.dumps(dict(files=deps,scope='PinnedexistingPython/binaryclosure+officialsourceAST+verifiedpublicweights+syntheticchecks+closedDrCIFproof;notwholeOS'),indent=2)+'\n')
names=['prepare.py','POLICY.json','PREREGISTRATION.md','cost_math.py','test_cost_math.py','guard.py','run_screen.py','profile_cost.py','verify_result.py','COST_TESTS.log','DEPENDENCIES.json'];put('COST_HASHES.json',json.dumps({n:sha(B/n) for n in names},indent=2)+'\n')
from guard import ready
try:ready();raise AssertionError('Missingremote marker mustblock')
except FileNotFoundError:pass
put('PRE_ACCESS_GUARD_CHECK.json',json.dumps(dict(missing_remote_marker_blocks=True,cost_files=len(names),dependencies=len(deps),formal_timing_started=False,actual_UWB_values_labels_fit=False),indent=2)+'\n');print(json.dumps(dict(cost_files=len(names),dependencies=len(deps),cost_hashes_sha256=sha(B/'COST_HASHES.json'))))
