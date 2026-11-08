from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;G=Path('/workspace/uwb-cloud-current')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not(P/'PROTOCOL.json').exists()
q={
 'state':'generated_core_preparation_only_no_complete_LOSO6_runner_BN_calibration_or_fit_release',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'method':'published_SWAD_LossValley_default3_6_0p3_component_on_full_native_GAP',
 'parameters':617612,'input':'all3signed_RMS_Nx90x567','batch':64,'epochs':64,'seed':20261029,'threads':1,
 'native_loss':'same_global_onehot12BCE','native_optimizer':'AdamW.001wd.0001StableLookahead.5k6lazy_clip2',
 'default_windows':{'n_converge':3,'n_tolerance':6,'tolerance_ratio':.3},
 'dense_segments':'each post-native-update model, one full-epoch checkpoint',
 'fixed_tail_control':'same trajectory dense epochs33..64; no test-selected window',
 'strong_baseline':'original GAP64 sixfold, no refit to lower conditions',
 'BN_required_before_admission':'source-only seeded shuffled full-training-ID B64 calibration, same for SWAD/tail; separately selected-GAP BN-only control; validation and held forbidden; RNG/mode restoration required',
 'author_deviations':[
 'Native617612 BCE/AdamWLookahead/all3RMS replaces pretrainedResNet/CE/Adam/images',
 'Author default freeze_bn=True; keep native trainingBN and explicitly adapt source-only calibration, not yet implemented',
 'Author stops at valley death; complete fixed64 native trajectory/tail while valley stays frozen',
 'One-full-epoch checkpoints replace author dataset-specific steps including step0',
 'Independent validation user replaces train_out/oracle reporting; no held-selected window',
 'CPU-only final cuda redirect; no author arithmetic changes'],
 'author_full_paper_reviewed':False,'author_benchmark_reproduced':False,'original_contribution_claim':False,
 'gate':{'equal_user_mean_gain_min':.005,'strict_wins_min':4,'worst_paired_min':-.02,'tolerance':1e-12},
 'profile_before_measure':{
 'updates_per_arm':8,'warmup':3,'measured':5,'order':'native_then_SWAD_odd_reverse_even',
 'formula':'1.25*31189.589142517*max(1,SWAD/native_wall_ratio,SWAD/native_CPU_ratio)+1200<=43200',
 'fixed_additive_reserve_for_BN_and_dense_journal_seconds':1200,
 'hard_main_wall_CPU_seconds':43200,'RSS_KiB':12582912,'single_attempt_no_retiming':True},
 'admission':'complete6 loader/worker/BN/averager journal/fullfresh64/native reference/role/resource audits + exact source and privateinitial release all required',
 'new_UWB_fit_started':False,
 'prior_GroupDRO_exact_closed_summary_sha256':sha('/workspace/uwb-groupdro-loso6-20261108/SUMMARY.json'),
 'pre_GroupDRO_score_route_sha256':sha('/workspace/uwb-groupdro-loso6-20261108/CONTINGENT_NEXT_IMPLEMENTATION_PRE_SCORE.json')}
(P/'PROTOCOL.json').write_text(json.dumps(q,indent=2)+'\n')
(P/'HISTORY_CHECK.json').write_text(json.dumps(dict(old_SWAD_protocol=read(G/'uwb_direction1/research/SWAD_SOURCE_PROTOCOL.json'),old_failed_result_repository='uwb_direction1/results/cloud_20261003_swad/SUMMARY.json',old_protocol_sha256=sha(G/'uwb_direction1/research/SWAD_SOURCE_PROTOCOL.json'),scope='old24/seed2026/users3..8/legacyGN+cosine+decay.001 vs fullnative617612/BN64/seed20261029/users1..6/nativeconstantAdamWLookaheadwd.0001; preserve old failure, not authorbenchmark or novelty',prior_GroupDRO_gate_false=True,eta0_control_not_activated=True,no_test_user_specific_tuning=True),indent=2)+'\n')
(P/'METADATA_GENERATION_REPAIR.json').write_text(json.dumps(dict(state='inline_metadata_SyntaxError_fixed_before_any_test_timing_or_UWB_access',error='positional argument follows keyword argument; single_attempt_no_retiming had no =True',core_file_already_written=True,protocol_not_created_by_failed_attempt=True,failed_attempt_started_training_or_timing=False,repair='persist explicit dictionary builder with boolean assignment and run once'),indent=2)+'\n')
print(json.dumps(dict(metadata_written=True,tests_started=False,realfit_started=False)))
