"""Private native checkpoint insurance; never fit, infer or read held truth."""
from pathlib import Path
import sys,json,time,shutil,hashlib,re,subprocess
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');G=Path('/workspace/uwb-cloud-current');sys.path.insert(0,str(ROOT))
from common import read,write,sha,privacy,alive,process_identity,config
from checks import verify_member,verify_audits
from controller import valid_fold,budget_reason
from stream_archive import build,verify
PARENT='3613a9681f59a1bd93e9e7fc44869e3fd861ff81'
RP='uwb_direction1/results/cloud_20261105_held6_producer_insurance01';CP='uwb_direction1/research/cloud_20261105_held6_producer_insurance01'
CODE=['snapshot.py','verify_native.py','pipeline.py','README.md']
def proof(arm):return read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json')
def observe():
    ident=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(ident) and ident['pid']==573 and ident['startticks']==34322 and ident['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
    assert read(ROOT/'STATE.json')==dict(held=6,state='training_fixed_members_unscored',users7_8_labels_used=False)
    for held in [2,3,4,5]:assert valid_fold(held);verify_audits(held)
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
    _,complete=verify_member(6,'shuffle');assert complete['epochs']==complete['selected_epoch']==100 and complete['updates']==23000
    term=read(ROOT/'attempts/attempt002/held6/shuffle_train_TERM.json')
    assert term['exit_code']==0 and term['resource_stop_reason'] is None and not term['wall_timeout'] and term['cgroup_OOM_kill_delta']==0
    copied=B/'capture01/attempts/attempt002/held6/shuffle_train_TERM.json';assert copied.exists() and sha(copied)==sha(ROOT/'attempts/attempt002/held6/shuffle_train_TERM.json')
    procs=read(ROOT/'attempts/attempt002/held6/PROCESSES.json')['identities'];active=[process_identity(x['pid']) for x in procs if alive(x)];assert len(active)==1
    assert b'timemil' in Path(f'/proc/{active[0]["pid"]}/cmdline').read_bytes()
    ledger=read(ROOT/'RESOURCE_LEDGER.json');_,p=config();assert budget_reason(ledger['folds'],6,p,ledger['last_observed_combined_RSS_KiB']) is None
    assert ledger['folds']['3']==dict(wall=25984.039045241712,cpu=100021.76343385856)
    assert ledger['folds']['2']==dict(wall=22834.174004506,cpu=84851.68278)
    assert ledger['folds']['4']==dict(wall=22593.303525181997,cpu=83070.146821)
    assert ledger['folds']['5']==dict(wall=23589.402822912998,cpu=87193.339813)
    o=dict(state='same_registered_held6_shuffle100_producer_complete_TimeMIL_running_unscored',observed_unix=time.time(),controller=process_identity(573),active_children=active,resource_ledger=ledger,captured_epochs={a:proof(a)['committed_epoch'] for a in ['timemil','shuffle']},current_progress={a:read(ROOT/'folds/held6'/a/'PROGRESS.json') for a in ['timemil','shuffle']},shuffle_producer_completion_sha256=sha(ROOT/'folds/held6/shuffle/ARM_COMPLETE.json'),shuffle_actual_termination=term,completed_folds2_3_4_5_valid_frozen_unscored=True,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),full100_shuffle_replay_started=(ROOT/'folds/held6/shuffle/replay/PROGRESS.json').exists(),full100_shuffle_replay_complete=False,scientific_configuration_changed=False,new_process_started_by_preservation=False,budget_reset=False,new_accuracy_result=False,diagnostic_labels_read=False,users7_8_labels_used=False,original_contribution_validated=False,data_redownloaded=False)
    assert not o['full100_shuffle_replay_started'];return o
def archive():
    for a in ['timemil','shuffle']:
        q=proof(a);assert q['native_instantiation_restore_recapture_exact'] and not q['full_training_replayed']
    sh=proof('shuffle');assert sh['committed_epoch']==100 and sh['committed_updates']==23000 and sh['complete_member_manifest_verified'] and sh['selected_model_matches_native_committed_state'] and sh['uncommitted_suffix_rows']==0
    for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert sha(B/'capture01'/n)==h,n
    o=observe();write(B/'LIVE_OBSERVATION.json',o)
    names=CODE+['CAPTURE.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','LIVE_OBSERVATION.json'];write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
    files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
    for f in (B/'capture01').rglob('*'):
        if f.is_file():files['overlay/'+str(f.relative_to(B/'capture01'))]=f
    m=build(B/'held6_producer_backup',f'uwb_held6_TimeMIL{proof("timemil")["committed_epoch"]}_Shuffle100_insurance01_20261105',files,'held6_complete_shuffle100_and_running_TimeMIL_native_insurance_not_full_audit_or_scoring_authority',chunk_size=7340032,limit=268435456)
    m.update(state='held6_shuffle100_and_intermediate_TimeMIL_recovery_archive_streamed',complete_modelset=False,delta_dependency=False,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False)
    write(B/'held6_producer_backup/MANIFEST.json',m);return {k:v for k,v in m.items() if k!='chunks'}|dict(parts=len(m['chunks']))
def verify_backup():
    b=B/'held6_producer_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows);v=verify(m,rows)
    assert not v['complete_modelset_verified'];write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
    r=dict(state='held6_complete_shuffle100_and_running_TimeMIL_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=False,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False,diagnostic_labels_read=False,new_accuracy_result=False,full_training_audit_complete=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
    write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));return v
def publication():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==PARENT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
    o=observe();write(B/'POST_BACKUP_LIVE_OBSERVATION.json',o)
    m=read(B/'held6_producer_backup/MANIFEST.json');v=read(B/'held6_producer_backup/READBACK_VERIFIED.json');assert v['archive_sha256']==m['archive_sha256']
    for n,h in read(B/'SOURCE_HASHES.json').items():assert sha(B/n)==h,n
    r=G/RP;c=G/CP;r.mkdir(parents=True);c.mkdir(parents=True)
    for n in ['LIVE_OBSERVATION.json','POST_BACKUP_LIVE_OBSERVATION.json','SOURCE_HASHES.json']:shutil.copyfile(B/n,r/n)
    for a in ['timemil','shuffle']:shutil.copyfile(B/'capture01'/f'{a}_SNAPSHOT_VERIFIED.json',r/f'{a}_SNAPSHOT_VERIFIED.json')
    for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
        f=r/'held6_producer_backup'/n;f.parent.mkdir(exist_ok=True);shutil.copyfile(B/'held6_producer_backup'/n,f)
    for n in CODE:shutil.copyfile(B/n,c/n)
    write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in CODE})
    tm=proof('timemil');arc=m['archive_sha256'];lr=o['resource_ledger']['folds']['6']
    note=f'''## 2026-11-05 held6 Shuffle100 producer complete; first native checkpoint insurance verified

Same original bootad409b48/controller573/startticks34322/attempt002, no restart/duplicate fitting or scientific/config/seed/gate/budget change. Held6 Shuffle completed100epochs/23000updates with fixed final100 selection and actual exit0/no stop/no OOM; TimeMIL continues (captured epoch{tm['committed_epoch']}/{tm['committed_updates']}updates; live epoch{o['current_progress']['timemil']['epoch']}at observation). Both captured native model/optimizer/sampler/all RNG states were instantiated/restored/recaptured exactly, journals/log prefixes/TimeMIL selection checked; complete Shuffle payload hashes and selected model match native committed100state. This is not the pending independent full100replay/output audits/three-arm freeze. Completed held2/3/4/5 are reused unscored, not retrained or uploaded again.

Owner-only raw intermediate archive: {len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads, SHA{arc}; all raw parts/whole archive/every member verified. NO checkpoint-delta dependency, initial source/data/maps/runtime still required; not a completed fold/model set or final scoring authority. Actual IDs/models/optimizer/RNG/binary unscored predictions/full logs remain private. Details/index: `{RP.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. Held6 cost{lr['wall']:.3f}wall/{lr['cpu']:.3f}CPU seconds within unchanged43200/172800caps; prior completed-fold charges retained. Continue original all5newfolds/3arms/audits/full final recovery before once aggregate. Source1 reused/users7–8 excluded/exploratory repeated-user caveat unchanged. No new accuracy or originality claim.

'''
    report=f'''# Held6 Shuffle100 producer completed; original TimeMIL continues

{note}
## Actual execution and native checkpoint checks

Parent `{PARENT}` completed held5. The SAME original registered controller now has held6 Shuffle100/23000updates completed, fixed final100 selected, actual producer exit0/no timeout/no resource stop/no cgroup OOM. One actual TimeMIL trainer remains active; no new fitting or restart was launched by preservation. All60 scientific frozen files remain SHA-identical. Completed held2/3/4/5 models/audits/frozen outputs and terminal resource costs are verified unchanged.

The coherent captured TimeMIL checkpoint is epoch{tm['committed_epoch']}/{tm['committed_updates']}updates, with{tm['captured_complete_step_rows']}complete logged rows and{tm['uncommitted_suffix_rows']}uncommitted suffix rows retained for native exact replay on recovery. Shuffle's whole immutable completed producer is captured. The unchanged generic snapshot and native verifier preserve private IDs, model/optimizer/sampler/Torch/Python/NumPy RNG state, model selection/storage aliases, journals, first-update certificates and log-prefix/raw-cut forensics. Separate native-family processes instantiate the original models/optimizers, restore and recapture exactly without a forward pass, optimizer update or held-label access. Every complete Shuffle member payload and its selected model matches its native committed100state. This is NOT independent full100training replay; it and output audits/fusion remain pending until both producers finish. The actual controller already owns that sequence; do not start a duplicate auditor.

## Private recovery scope and reproducibility

Archive SHA256 `{arc}` contains{m['tracked_payload_files']}payloads plus member manifest, {len(m['chunks'])}parts and{m['archive_bytes']}bytes. Metadata establishes owner-only/nonshared original private folder, parent/MIME/size before download_raw_file=true/include_base64=false fetch; immediate materialization/size/SHA and whole archive/every member pass, at most3 complete concurrent pipelines. No authenticated URL is output or committed. Actual IDs, tensor states, binary unscored predictions and complete logs remain private; no raw radar arrays/runtime binaries are duplicated or put in Git. Original789source snapshot is untouched.

This is a complete Shuffle producer plus ONE coherent running TimeMIL checkpoint, not a completed three-arm fold/model set or final scoring authority. It has no checkpoint-delta dependency; initial source/seed/ID/data/maps/runtime and original registration are still required. Restore only into a NEW inspection directory after confirmed loss, all old identities inactive, no newer/live state and metadata/raw/archive/member/native/journal/suffix checks. Preserve original logs, all cumulative caps/charges and the original unknown elapsed wall+5CPU/wallsecond policy. Never overwrite live/newer state, refit completed models or compute held diagnostic accuracy to select a configuration.

## Remaining registered experiment and limits

Held6 cumulative wall{lr['wall']:.6f}/CPU{lr['cpu']:.6f}seconds is within original43200wall/172800CPU per-fold and216000wall/864000CPU five-newfold limits. The original controller must finish TimeMIL, independent Shuffle100replay, both output audits and three-arm freeze. This is the last new fold; full-fivefold recoverability remains pending. Only all five newfolds/three arms/audits/frozen outputs and FULL final private recoverability permit once-only source-six diagnostics and unchanged bounded accuracy gates. Source1 is reused without rescoring, users7–8 labels remain excluded from new-method selection and repeated source1/eight-user development is exploratory. Accuracy improvement, eight-fold generalization and original contribution remain unestablished by this producer-completion milestone.
'''
    (r/'RUN_REPORT.md').write_text(report)
    updates=dict(state='source6_held6_shuffle100_producer_complete_TimeMIL_running_privately_insured',latest_report=RP.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=573,background_training_observed_unix=o['observed_unix'],background_training_status_note='Same controller573/bootad409b48. Held6 Shuffle100/23000updates fixedfinal100 completed with actual exit0; TimeMIL continues. Native restore/recapture and complete selected-state checks plus private raw/allmember intermediate archive verified; full100replay/output audits/three-arm freeze remain pending. Held2/3/4/5 reused unscored. No diagnostic accuracy/originality claim or scientific/budget change.')
    key='source6_held6_producer_insurance01_20261105';insert=dict(report=updates['latest_report'],observation=RP.removeprefix('uwb_direction1/')+'/POST_BACKUP_LIVE_OBSERVATION.json',private_download_index=RP.removeprefix('uwb_direction1/')+'/held6_producer_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,checkpoint_delta_dependency=False,captured_epochs={'timemil':tm['committed_epoch'],'shuffle':100},shuffle_updates=23000,full100_shuffle_replay_complete=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
    paths=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
    for path in paths:
        f=G/path;body=f.read_text()
        if path.endswith('STATUS.json'):
            assert '"'+key+'"' not in body
            for k,val in updates.items():
                rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
                body,n=re.subn(rx,lambda m:m[1]+json.dumps(val,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
            body=body.rstrip();assert body.endswith('}');body=body[:-1]+',\n  "'+key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n';json.loads(body)
        else:body=note+body
        f.write_text(body)
    assert not subprocess.check_output(['git','diff','--name-only',PARENT,'--','uwb_direction1/original'],cwd=G,text=True).strip()
    files=[]
    for path in paths+[str(f.relative_to(G)) for f in sorted(r.rglob('*')) if f.is_file()]+[str(f.relative_to(G)) for f in sorted(c.rglob('*')) if f.is_file()]:
        raw=(G/path).read_bytes();assert not path.endswith(('.pt','.pth','.npz','.npy','.bin','.jsonl'))
        row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        if path in paths:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
        files.append(row)
    write(B/'PUBLICATION_MANIFEST.json',dict(root=str(G),parent=PARENT,files=files,note=note,status_updates=updates,status_insert_key=key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2)))
    return dict(state='held6_shuffle100_and_intermediate_TimeMIL_publication_prepared',files=len(files),archive_sha256=arc,captured_epochs=insert['captured_epochs'],new_accuracy_result=False)
if __name__=='__main__':
    assert len(sys.argv)==2;commands={'archive':archive,'verify':verify_backup,'publication':publication};print(json.dumps(commands[sys.argv[1]]()))
