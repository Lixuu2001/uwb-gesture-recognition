"""Preserve completed held5 producers and live auditor OWN state without fitting/scoring."""
from pathlib import Path
import sys,json,time,hashlib,shutil,subprocess,re
B=Path(__file__).resolve().parent
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
G=Path('/workspace/uwb-cloud-current')
sys.path.insert(0,str(ROOT))
from common import read,write,sha,privacy,alive,process_identity,config
from checks import verify_member,verify_audits
from controller import valid_fold,budget_reason
from stream_archive import build,verify
PARENT='acd13ef2678fdc415994d1aeba571a471848b4c0'
RP='uwb_direction1/results/cloud_20261104_held5_audit_insurance01'
CP='uwb_direction1/research/cloud_20261104_held5_audit_insurance01'
ARCHIVE='held5_audit_backup'
CODE=['snapshot.py','verify_native.py','augment_capture.py','pipeline.py','README.md']
TERMINAL={
 '2':dict(wall=22834.174004506,cpu=84851.68278),
 '3':dict(wall=25984.039045241712,cpu=100021.76343385856),
 '4':dict(wall=22593.303525181997,cpu=83070.146821),
}
def proof(arm):return read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json')
def observe():
    ident=read(ROOT/'CONTROLLER_IDENTITY.json')
    assert alive(ident) and ident['pid']==573 and ident['startticks']==34322 and ident['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
    state=read(ROOT/'STATE.json');assert state==dict(held=5,state='auditing_members_unscored')
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
    for held in [2,3,4]:assert valid_fold(held);verify_audits(held)
    complete={};terms={}
    for arm in ['timemil','shuffle']:
        _,q=verify_member(5,arm);complete[arm]={k:v for k,v in q.items() if k!='files'}
        t=read(ROOT/'attempts/attempt002/held5'/f'{arm}_train_TERM.json')
        assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
        assert sha(B/'capture01/attempts/attempt002/held5'/f'{arm}_train_TERM.json')==sha(ROOT/'attempts/attempt002/held5'/f'{arm}_train_TERM.json')
        terms[arm]=t
    assert complete['timemil']['updates']==4608 and complete['timemil']['selected_epoch']==24
    assert complete['shuffle']['updates']==23000 and complete['shuffle']['selected_epoch']==100
    cert=read(ROOT/'folds/held5/timemil/inference_audit/CERTIFICATE.json')
    term=read(ROOT/'folds/held5/timemil/inference_audit/TERMINATION.json')
    assert cert['state']=='pass' and not cert['full_training_replayed'] and not cert['diagnostic_labels_read']
    assert term['exit_code']==0 and term['resource_stop_reason'] is None and term['cgroup_OOM_kill_delta']==0
    for n,h in cert['artifact_hashes'].items():assert sha(n)==h,n
    procs=read(ROOT/'attempts/attempt002/held5/PROCESSES.json')['identities']
    active=[process_identity(x['pid']) for x in procs if alive(x)];assert len(active)==1
    assert b'replay_shuffle.py' in Path(f'/proc/{active[0]["pid"]}/cmdline').read_bytes()
    ledger=read(ROOT/'RESOURCE_LEDGER.json');_,p=config()
    assert budget_reason(ledger['folds'],5,p,ledger['last_observed_combined_RSS_KiB']) is None
    for h,cost in TERMINAL.items():assert ledger['folds'][h]==cost
    captured=read(B/'capture01/folds/held5/shuffle/replay/PROGRESS.json')
    current=read(ROOT/'folds/held5/shuffle/replay/PROGRESS.json')
    assert 1<=captured['epoch']<=current['epoch']<100
    return dict(state='held5_both_producers_complete_TimeMIL_output_audit_pass_independent_Shuffle_replay_running_unscored',observed_unix=time.time(),controller=process_identity(573),current_state=state,active_auditor=active[0],complete_producers=complete,actual_producer_terminations=terms,TimeMIL_output_certificate={k:v for k,v in cert.items() if k!='artifact_hashes'},TimeMIL_output_termination=term,current_own_replay=current,captured_own_replay=captured,resource_ledger=ledger,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),completed_newfolds_reused_unscored=[2,3,4],scientific_configuration_changed=False,budget_reset=False,new_process_started_by_preservation=False,diagnostic_labels_read=False,users7_8_labels_used=False,full100_Shuffle_replay_complete=False,new_accuracy_result=False,original_contribution_validated=False,data_redownloaded=False)
def archive():
    for arm in ['timemil','shuffle']:
        q=proof(arm)
        assert q['native_instantiation_restore_recapture_exact'] and q['complete_member_manifest_verified'] and q['selected_model_matches_native_committed_state']
        assert q['uncommitted_suffix_rows']==0
    assert proof('timemil')['committed_updates']==4608 and proof('timemil')['retained_output_audit_verified']
    sh=proof('shuffle');assert sh['committed_updates']==23000 and sh['own_checkpoint_native_restore_recapture_exact'] and sh['own_replay_prefix_of_producer_steps_verified']
    assert sh['own_epoch_records_exact']==sh['own_replay_epoch']+1
    prior=Path('/workspace/uwb-source6-held4-audit-delta02-20261104')
    for n in ['snapshot.py','verify_native.py']:assert sha(B/n)==sha(prior/n)
    write(B/'LIVE_OBSERVATION.json',observe())
    names=CODE+['CAPTURE.log','AUGMENT.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','LIVE_OBSERVATION.json']
    write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
    files={'preservation/'+n:B/n for n in names+['SOURCE_HASHES.json']}
    for f in (B/'capture01').rglob('*'):
        if f.is_file():files['overlay/'+str(f.relative_to(B/'capture01'))]=f
    e=sh['own_replay_epoch']
    m=build(B/ARCHIVE,f'uwb_held5_complete_producers_own{e}_insurance01_20261104',files,'held5_complete_producers_passed_TimeMIL_output_audit_and_own_Shuffle_checkpoint_not_full_audit_or_scoring_authority',chunk_size=7340032,limit=268435456)
    m.update(state='held5_complete_producers_own_auditor_checkpoint_insurance_streamed',complete_modelset=False,delta_dependency=False,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False)
    write(B/ARCHIVE/'MANIFEST.json',m)
    return {k:v for k,v in m.items() if k!='chunks'}|dict(parts=len(m['chunks']))
def verify_backup():
    b=B/ARCHIVE;m=read(b/'MANIFEST.json')
    rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows)
    v=verify(m,rows);assert not v['complete_modelset_verified'] and not m['delta_dependency']
    assert sha(b/'MEMBERS.json')==m['members_manifest_sha256']
    entries={x['path']:x for x in read(b/'MEMBERS.json')['members']}
    for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert entries['overlay/'+n]['sha256']==h
    for n,h in read(B/'SOURCE_HASHES.json').items():assert sha(B/n)==h,n
    write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
    r=dict(state='held5_complete_producers_own_auditor_checkpoint_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=False,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,full_training_audit_complete=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
    write(b/'DRIVE_BACKUP_RECEIPT.json',r)
    write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r))
    return v
def publication():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==PARENT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
    o=observe();write(B/'POST_BACKUP_LIVE_OBSERVATION.json',o)
    m=read(B/ARCHIVE/'MANIFEST.json');v=read(B/ARCHIVE/'READBACK_VERIFIED.json');assert v['archive_sha256']==m['archive_sha256']
    for n,h in read(B/'SOURCE_HASHES.json').items():assert sha(B/n)==h,n
    sh=proof('shuffle');tm=proof('timemil');e=sh['own_replay_epoch'];arc=m['archive_sha256'];cost=o['resource_ledger']['folds']['5']
    r=G/RP;c=G/CP;r.mkdir(parents=True);c.mkdir(parents=True)
    for n in ['LIVE_OBSERVATION.json','POST_BACKUP_LIVE_OBSERVATION.json','SOURCE_HASHES.json']:shutil.copyfile(B/n,r/n)
    for arm in ['timemil','shuffle']:shutil.copyfile(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json',r/f'{arm}_SNAPSHOT_VERIFIED.json')
    for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
        p=r/ARCHIVE/n;p.parent.mkdir(exist_ok=True);shutil.copyfile(B/ARCHIVE/n,p)
    for n in CODE:shutil.copyfile(B/n,c/n)
    write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in CODE})
    audit=o['active_auditor'];live=o['current_own_replay']['epoch']
    note=f'''## 2026-11-04 held5 producers complete; TimeMIL output audit passed; original Shuffle replay insured

Same original bootad409b48/controller573/startticks34322/attempt002. Held5 TimeMIL64/4608updates selected24 from fixed inner validation; Shuffle100/23000updates fixedfinal100. Both actual producer terminations exit0/no stop/no OOM. TimeMIL native/selection/all validation and unscored diagnostic output audit passed with actual exit0, NOT full64 training-trajectory replay. Independent Shuffle full100 replay is still running as PID{audit['pid']}/startticks{audit['startticks']}, live epoch{live}; full replay/output audit/three-arm freeze remain pending. No held accuracy read, no scientific/config/seed/gate/budget change or duplicate fitting. Completed newfolds2/3/4 reused unscored;6pending.

Private first held5 insurance preserves both complete producers, actual TimeMIL output certificate/termination and auditor OWN{e}/{230*e}updates. Both native models/optimizers/sampler/all RNG states restore and recapture exactly; own state, all{sh['own_epoch_records_exact']}producer-matching records and original-step-log prefix pass. Archive{len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads, SHA{arc}, raw whole-archive/every-member verification passed in owner-only original folder. NO earlier checkpoint-delta dependency; original initial/source/data/maps/runtime dependencies still required. Intermediate recovery only, not completed-fold or final scoring authority. Report/index: `{RP.removeprefix('uwb_direction1/')}/RUN_REPORT.md`.

Held5 cumulative cost{cost['wall']:.6f}wall/{cost['cpu']:.6f}CPU seconds is within original caps; all prior terminal costs/unknown-interruption charges retained. All5newfolds/3arms/audits/frozen outputs AND FULL final private recovery still precede once-only source-six diagnostics and unchanged bounded gates. Source1 reused without rescoring; users7–8 labels excluded from new-method selection; repeated source1/eight-user development exploratory. Accuracy improvement and original-contribution evidence remain separate and unresolved.

'''
    report=f'''# Held5 completed producers and passed TimeMIL selected-output audit; original independent Shuffle replay continues

{note}
## Evidence and recovery scope

Parent `{PARENT}` preserved complete held4 and live held5 training. This continuation observed the same original controller finish both held5 members and start its registered audits. Both producer ARM_COMPLETE manifests and all included payload SHAs pass. Exact training counts derive from the private split:4600training IDs,72TimeMIL or230Shuffle updates/epoch, yielding4608/23000updates. No new model, optimizer update, forward pass or diagnostic-label read was started by preservation. Generic snapshot/native verifier files are byte-identical to previously verified helpers; augmentation retains held5's actual TimeMIL output audit certificate and successful termination.

Both completed native states were instantiated through the original families, restored with complete optimizer/sampler/Torch/Python/NumPy RNG and recaptured exactly; full committed journals, original log prefixes, TimeMIL selection and storage aliases, and selected-model tensors pass. The captured Shuffle auditor OWN epoch{e}/{230*e}updates was separately restored/recaptured; registered initial-seed record and all{sh['own_epoch_records_exact']}records match the producer, and its checkpoint references the exact prefix of the original23000-row training log. These read-only checks do NOT constitute the unfinished independent100epoch replay. TimeMIL selected-output/native audit does NOT claim a64epoch training replay. Actual current identity/progress/cost is in POST_BACKUP_LIVE_OBSERVATION.json.

Archive SHA256 `{arc}` contains{m['tracked_payload_files']}payloads plus member manifest in{len(m['chunks'])}parts/{m['archive_bytes']}bytes. Both completed producers, selected/final/initial/full optimizer/RNG models, actual sample IDs, logs, epoch journals/validation arrays, actual producer terminations, TimeMIL output audit and coherent live auditor checkpoint are private. Every part passed owner-only/nonshared original private-parent/MIME/size metadata before download_raw_file=true/include_base64=false, immediate materialization/size/SHA verification and whole archive/every member verification; at most3 complete pipelines. No authenticated URL is output or committed. No raw radar arrays, runtime binaries, actual sample IDs, model tensors or prediction arrays enter Git. Original789source snapshot and60frozen scientific files remain unchanged.

There is NO earlier checkpoint-delta dependency. Original initial/source/seed/ID/data/maps/runtime/preregistration remain required. Restore overlay/ only into a NEW inspection directory after confirmed loss and inactive old identities; verify archive members/native states/auditor prefix/actual certificates first, preserve old/newer/live files and all cumulative caps/charges, and resume only the auditor's OWN state. Do not shortcut to the producer-final state, rerun completed fitting or score held outputs for selection. Preserve all history and charge unknown interruptions at wall+5CPU/wallsecond. This intermediate archive is not completed three-arm-fold or all5fold final scoring authority; after all audits/fusion create a non-delta complete-fold archive.

Completed held2/3/4 models, audits, frozen outputs and terminal resource costs remain verified unchanged. Held5 observed cost{cost['wall']:.6f}wall/{cost['cpu']:.6f}CPU seconds retains original43200wall/172800CPU per-fold and216000wall/864000CPU total caps without reset. The original controller continues the fixed protocol; full100Shuffle replay/output audit/fusion and held6 remain. Only all5newfolds/all3arms/audits/frozen outputs plus full final private recoverability permit once-only source-six diagnostics under original gates. Source1 pilot is reused, users7–8 excluded from new-method selection and prior repeated-user development exploratory. This milestone establishes no new accuracy estimate, eight-fold improvement or originality claim.
'''
    (r/'RUN_REPORT.md').write_text(report)
    updates=dict(state='source6_held5_producers_complete_TimeMIL_output_audit_pass_Shuffle_replay_insured',latest_report=RP.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=573,background_training_observed_unix=o['observed_unix'],background_training_status_note=f'Same controller573/bootad409b48; held5 TimeMIL64/4608 selected24 and Shuffle100/23000fixed100 complete actual exit0; TimeMIL output audit passed, not full64replay. Independent Shuffle replayPID{audit["pid"]} active, unfinished. Private non-delta intermediate archive preserves both complete producers/auditor OWN{e}; all raw parts/members/native state verified. Held2/3/4 unchanged unscored,6pending. No accuracy/originality or scientific/gate/seed/budget change.',compute='2026-11-04 bootad409b48-89ce-4704-beee-3658103d5856; CPU quota4/RAM16GiB; Python3.12.14 NumPy2.3.5 Torch2.15.0+cpu; original independent Shuffle replay4threads; both held5 producers complete; no GPU')
    key='source6_held5_audit_insurance01_20261104'
    insert=dict(report=updates['latest_report'],observation=RP.removeprefix('uwb_direction1/')+'/POST_BACKUP_LIVE_OBSERVATION.json',private_download_index=RP.removeprefix('uwb_direction1/')+'/'+ARCHIVE+'/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,checkpoint_delta_dependency=False,captured_own_replay_epoch=e,TimeMIL_output_audit_passed=True,TimeMIL_full64_training_replayed=False,Shuffle_full100_training_replayed=False,Shuffle_exact_producer_updates=23000,TimeMIL_exact_producer_updates=4608,completed_newfolds=[2,3,4],active_held=5,all5newfolds_complete=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
    paths=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
    for path in paths:
        f=G/path;body=f.read_text()
        if path.endswith('STATUS.json'):
            assert '"'+key+'"' not in body
            for k,val in updates.items():
                rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
                body,n=re.subn(rx,lambda m:m[1]+json.dumps(val,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
            body=body.rstrip();assert body.endswith('}')
            body=body[:-1]+',\n  "'+key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n';json.loads(body)
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
    return dict(state='held5_intermediate_private_insurance_publication_prepared',files=len(files),archive_sha256=arc,captured_own_replay_epoch=e,current_own_replay_epoch=live,new_accuracy_result=False)
if __name__=='__main__':
    assert len(sys.argv)==2
    print(json.dumps({'archive':archive,'verify':verify_backup,'publication':publication}[sys.argv[1]]()))
