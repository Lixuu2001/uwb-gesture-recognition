"""Completed producer / own-auditor delta insurance; no fitting or held scoring."""
from pathlib import Path
import sys,json,time,hashlib,shutil,subprocess,re
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');BASE=Path('/workspace/uwb-source6-held6-producer-insurance01-20261105');G=Path('/workspace/uwb-cloud-current');sys.path.insert(0,str(ROOT))
from common import read,write,sha,privacy,alive,process_identity,config
from delta import create,apply
from stream_archive import build,verify
from checks import verify_member,verify_audits
from controller import valid_fold,budget_reason
PARENT='9b4f21fc7371dd12b471d274124e7e4d03d2f85c';BASE_SHA='6cde909c0f442f4813d8511f6859cce857a89740caea8196ee45d66473bca80b'
RP='uwb_direction1/results/cloud_20261105_held6_audit_delta02';CP='uwb_direction1/research/cloud_20261105_held6_audit_delta02'
CODE=['snapshot.py','verify_native.py','delta.py','augment_capture.py','pipeline.py','README.md']
def proof(root,arm):return read(root/f'{arm}_SNAPSHOT_VERIFIED.json')
def prepare():
    for a in ['timemil','shuffle']:
        q=proof(B/'capture01',a);assert q['native_instantiation_restore_recapture_exact'] and q['complete_member_manifest_verified'] and q['selected_model_matches_native_committed_state']
    assert proof(B/'capture01','timemil')['retained_output_audit_verified']
    sh=proof(B/'capture01','shuffle');assert sh['own_replay_epoch']>=26 and sh['own_checkpoint_native_restore_recapture_exact'] and sh['own_replay_prefix_of_producer_steps_verified']
    archive=BASE/'held6_producer_backup';m=read(archive/'MANIFEST.json');v=read(archive/'READBACK_VERIFIED.json');rows=read(archive/'READBACK_INPUT.json');privacy(rows)
    assert m['archive_sha256']==v['archive_sha256']==BASE_SHA and not m['delta_dependency'] and verify(m,rows)==v
    assert sha(archive/'MEMBERS.json')==m['members_manifest_sha256']
    entry={x['path']:x for x in read(archive/'MEMBERS.json')['members']}['overlay/PAYLOAD_HASHES.json'];assert sha(BASE/'capture01/PAYLOAD_HASHES.json')==entry['sha256']
    p=create(BASE/'capture01',B/'capture01',B/'delta_payload',BASE_SHA)
    p.update(chain_depth=1,base_overlay_archive_member_prefix='overlay/',base_index_repository_path='uwb_direction1/results/cloud_20261105_held6_producer_insurance01/held6_producer_backup/CLOUD_DOWNLOAD_INDEX.json',base_reconstruction_required=True)
    write(B/'delta_payload/DELTA.json',p);shutil.copyfile(archive/'CLOUD_DOWNLOAD_INDEX.json',B/'delta_payload/BASE_ARCHIVE_INDEX.json')
    r=apply(BASE/'capture01',B/'delta_payload',B/'reconstructed01');write(B/'LOCAL_RECONSTRUCTION.json',r)
    assert sha(B/'capture01/PAYLOAD_HASHES.json')==sha(B/'reconstructed01/PAYLOAD_HASHES.json')
    prior=Path('/workspace/uwb-source6-held3-audit-delta04-20261104')
    for n in ['snapshot.py','verify_native.py','delta.py']:assert sha(B/n)==sha(prior/n),n
    write(B/'BASE_VERIFIED.json',dict(state='prior_private_full_checkpoint_raw_allmember_and_overlay_table_verified',archive_sha256=BASE_SHA,chain_depth=1,base_payload_table_sha256=p['base_payload_table_sha256'],current_payload_table_sha256=p['current_payload_table_sha256'],three_generic_helpers_unchanged=True))
    return dict(state=r['state'],current_files=r['files'],changed_files=r['changed_files'],captured_own_replay_epoch=sh['own_replay_epoch'])
def observe():
    ident=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(ident) and ident['pid']==573 and ident['startticks']==34322 and ident['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
    assert read(ROOT/'STATE.json')==dict(held=6,state='auditing_members_unscored')
    for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
    for held in [2,3,4,5]:assert valid_fold(held);verify_audits(held)
    complete={};terms={}
    for a in ['timemil','shuffle']:
        _,q=verify_member(6,a);complete[a]={k:v for k,v in q.items() if k!='files'}
        assert q['updates']==(4608 if a=='timemil' else 23000) and q['selected_epoch']==(63 if a=='timemil' else 100)
        t=read(ROOT/'attempts/attempt002/held6'/f'{a}_train_TERM.json');assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0;terms[a]=t
    cert=read(ROOT/'folds/held6/timemil/inference_audit/CERTIFICATE.json');term=read(ROOT/'folds/held6/timemil/inference_audit/TERMINATION.json')
    assert cert['state']=='pass' and not cert['full_training_replayed'] and not cert['diagnostic_labels_read'] and term['exit_code']==0 and term['resource_stop_reason'] is None and term['cgroup_OOM_kill_delta']==0
    for n,h in cert['artifact_hashes'].items():assert sha(n)==h,n
    procs=read(ROOT/'attempts/attempt002/held6/PROCESSES.json')['identities'];active=[process_identity(x['pid']) for x in procs if alive(x)];assert len(active)==1
    assert b'replay_shuffle.py' in Path(f'/proc/{active[0]["pid"]}/cmdline').read_bytes()
    ledger=read(ROOT/'RESOURCE_LEDGER.json');_,p=config();assert budget_reason(ledger['folds'],6,p,ledger['last_observed_combined_RSS_KiB']) is None
    assert ledger['folds']['2']==dict(wall=22834.174004506,cpu=84851.68278) and ledger['folds']['3']==dict(wall=25984.039045241712,cpu=100021.76343385856)
    assert ledger['folds']['4']==dict(wall=22593.303525181997,cpu=83070.146821) and ledger['folds']['5']==dict(wall=23589.402822912998,cpu=87193.339813)
    current=read(ROOT/'folds/held6/shuffle/replay/PROGRESS.json');captured=read(B/'capture01/folds/held6/shuffle/replay/PROGRESS.json');assert 26<=captured['epoch']<=current['epoch']<100
    return dict(state='same_registered_held6_both_producers_complete_TimeMIL_output_audit_pass_own_Shuffle_replay_running_unscored',observed_unix=time.time(),controller=process_identity(573),active_auditor=active[0],complete_producers=complete,actual_producer_terminations=terms,TimeMIL_output_certificate={k:v for k,v in cert.items() if k!='artifact_hashes'},TimeMIL_output_termination=term,current_own_replay=current,captured_own_replay=captured,resource_ledger=ledger,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),completed_folds2_3_4_5_reused_unscored=True,scientific_configuration_changed=False,budget_reset=False,new_training_or_trajectory_audit_started_by_preservation=False,diagnostic_labels_read=False,users7_8_labels_used=False,full100_Shuffle_replay_complete=False,new_accuracy_result=False,original_contribution_validated=False,data_redownloaded=False)
def archive():
    for a in ['timemil','shuffle']:
        original=proof(B/'capture01',a);restored=proof(B/'reconstructed01',a)
        for k in ['state_sha256','checkpoint_sha256','committed_epoch','committed_updates','own_replay_epoch','own_epoch_records_exact','native_instantiation_restore_recapture_exact','retained_output_audit_verified','own_checkpoint_native_restore_recapture_exact','own_replay_prefix_of_producer_steps_verified']:assert original[k]==restored[k],(a,k)
    write(B/'LIVE_OBSERVATION.json',observe());names=CODE+['CAPTURE.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','RECONSTRUCTED_TIMEMIL.log','RECONSTRUCTED_SHUFFLE.log','LOCAL_RECONSTRUCTION.json','BASE_VERIFIED.json','LIVE_OBSERVATION.json'];write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
    files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
    for d,prefix in [(B/'delta_payload','delta_payload'),(B/'capture01','capture_proofs'),(B/'reconstructed01','reconstructed_proofs')]:
        for f in d.rglob('*'):
            if f.is_file() and (prefix=='delta_payload' or f.name.endswith('_SNAPSHOT_VERIFIED.json')):files[prefix+'/'+str(f.relative_to(d))]=f
    e=proof(B/'capture01','shuffle')['own_replay_epoch'];m=build(B/'delta02_backup',f'uwb_held6_complete_producers_own{e}_delta02_20261105',files,'held6_complete_producers_own_auditor_delta_depth1_not_full_audit_or_scoring_authority',chunk_size=7340032,limit=268435456)
    m.update(state='held6_complete_producers_own_auditor_checkpoint_delta_streamed',complete_modelset=False,delta_dependency=True,chain_depth=1,base_archive_sha256=BASE_SHA,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False);write(B/'delta02_backup/MANIFEST.json',m)
    return {k:v for k,v in m.items() if k!='chunks'}|dict(parts=len(m['chunks']))
def verify_backup():
    b=B/'delta02_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows);v=verify(m,rows);write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
    r=dict(state='held6_complete_producers_own_auditor_delta_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=True,chain_depth=1,base_archive_sha256=BASE_SHA,base_index_repository_path=read(B/'delta_payload/DELTA.json')['base_index_repository_path'],initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False,diagnostic_labels_read=False,new_accuracy_result=False,full_training_audit_complete=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
    write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));return v
def publication():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==PARENT and not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
    o=observe();write(B/'POST_BACKUP_LIVE_OBSERVATION.json',o);m=read(B/'delta02_backup/MANIFEST.json');v=read(B/'delta02_backup/READBACK_VERIFIED.json');assert m['archive_sha256']==v['archive_sha256']
    for n,h in read(B/'SOURCE_HASHES.json').items():assert sha(B/n)==h,n
    d=read(B/'delta_payload/DELTA.json');table=read(B/'delta_payload/CURRENT_PAYLOAD_HASHES.json');sh=proof(B/'reconstructed01','shuffle');e=sh['own_replay_epoch'];r=G/RP;c=G/CP;r.mkdir(parents=True);c.mkdir(parents=True)
    for n in ['LIVE_OBSERVATION.json','POST_BACKUP_LIVE_OBSERVATION.json','LOCAL_RECONSTRUCTION.json','BASE_VERIFIED.json','SOURCE_HASHES.json']:shutil.copyfile(B/n,r/n)
    for role in ['capture01','reconstructed01']:
        for a in ['timemil','shuffle']:shutil.copyfile(B/role/f'{a}_SNAPSHOT_VERIFIED.json',r/f'{role}_{a}_SNAPSHOT_VERIFIED.json')
    for n in ['DELTA.json','CURRENT_PAYLOAD_HASHES.json','BASE_ARCHIVE_INDEX.json']:
        f=r/'delta_payload'/n;f.parent.mkdir(exist_ok=True);shutil.copyfile(B/'delta_payload'/n,f)
    for n in ['MANIFEST.json','READBACK_VERIFIED.json','DRIVE_BACKUP_RECEIPT.json','CLOUD_DOWNLOAD_INDEX.json']:
        f=r/'delta02_backup'/n;f.parent.mkdir(exist_ok=True);shutil.copyfile(B/'delta02_backup'/n,f)
    for n in CODE:shutil.copyfile(B/n,c/n)
    write(c/'SOURCE_HASHES.json',{n:sha(c/n) for n in CODE})
    cost=o['resource_ledger']['folds']['6'];arc=m['archive_sha256'];audit=o['active_auditor'];current=o['current_own_replay']['epoch']
    note=f'''## 2026-11-05 held6 both producers complete; TimeMIL output audit passed; own Shuffle replay insured

Same original bootad409b48/controller573/startticks34322/attempt002, no restart/duplicate fitting or scientific/config/seed/gate/budget change. Held6 TimeMIL64/4608updates selected63 and Shuffle100/23000updates fixed100 both completed with actual exit0/no stop/no OOM. TimeMIL complete native/selection/output audit passed with actual successful termination, NOT full64trajectory replay. Independent Shuffle full100replay actually runs as PID{audit['pid']}/startticks{audit['startticks']} (live epoch{current}at observation), unfinished. Completed held2/3/4/5 reused unscored; no diagnostic accuracy or originality claim.

New depth1 delta02 preserves both complete producers, actual TimeMIL certificate/termination and auditor OWN{e}/{230*e}updates. All{len(table)}current payloads ({len(d['changed'])}changed) reconstructed in a NEW directory and passed both native model/optimizer/all RNG/selection checks; own-auditor native restore/recapture,{sh['own_epoch_records_exact']}matching epoch records and original-step-log prefix verified. Owner-only raw archive: {len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads, SHA{arc}, whole archive/all members verified. Requires producer insurance01 SHA{BASE_SHA}; BASE_ARCHIVE_INDEX.json/guide included at `{RP.removeprefix('uwb_direction1/')}/RUN_REPORT.md`. Not standalone or final scoring authority; initial source/data/maps/runtime still required. Preserve live/newer state/caps and resume only auditor OWN state. After full audits/fusion, preserve a complete-fold archive to end this dependency.

Held6 cost{cost['wall']:.3f}wall/{cost['cpu']:.3f}CPU seconds within original43200/172800caps; completed-fold charges unchanged. Continue all5newfolds/3arms/audits/frozen outputs and FULL final private recovery before once source-six aggregate. Source1 reused/users7–8 excluded/repeated-user exploratory caveat unchanged; accuracy and originality evidence remain separate and unresolved.

'''
    report=f'''# Held6 completed producers and passed TimeMIL output audit; independent Shuffle replay in progress

{note}
## Verified execution and honest replay scope

Parent `{PARENT}` preserved TimeMIL54/Shuffle100. The SAME original registered controller completed TimeMIL64/4608updates, selectedepoch63 from inner validation, and both actual producer terminations are exit0/no resource stop/no OOM. TimeMIL native model/optimizer/RNG/journal/selection and ALL validation/unscored-diagnostic outputs reproduced exactly under the original successful selected-output audit. This is NOT full64training-trajectory replay. The independent Shuffle auditor starts from a fresh registered seed, checks every native update/order/loss/epoch state/validation output against the producer, and resumes only its OWN checkpoint. Actual current auditor identity/progress is in POST_BACKUP_LIVE_OBSERVATION.json. Full100replay, subsequent Shuffle output audit and fixed three-arm fusion remain unfinished; no held accuracy is read or implied.

## Actual recovery implementation and reconstruction

Generic snapshot/native verifier/content-checked delta code are byte-identical to prior verified helpers; augmentation targets held6's actual TimeMIL certificate/termination. Both complete immutable producer manifests, selected/final/full optimizer/RNG states, actual IDs, complete logs/journals/validation outputs and own-auditor checkpoint are captured. Both original native families are instantiated, restored and recaptured exactly in separate read-only processes; every completed-member payload and selected-model state matches. Own-auditor epoch{e}/{230*e}updates also passes native restore/recapture, registered initial-seed record, all{sh['own_epoch_records_exact']}epoch-record equalities and exact prefix digest of the original23000-row training log. No forward pass, optimizer update, new fitting or diagnostic-label read occurs in preservation checks.

The previous non-delta producer insurance01 was raw/allmember verified, with its payload table cryptographically anchored to that archive. The delta copied{len(d['changed'])}changed paths and reconstructed all{len(table)}current payloads into a NEW directory without modifying the base. Both native verifiers passed on this real reconstruction, with substantive state/checkpoint hashes equal to the source capture; timing metrics from separate processes are not claimed byte-identical. All60 scientific source/config files and original789source snapshot remain unchanged. Completed held2/3/4/5 archives/models/audits/costs are reused without repeating fitting.

## Private archive and dependency

Archive SHA256 `{arc}` has{len(m['chunks'])}parts/{m['archive_bytes']}bytes/{m['tracked_payload_files']}payloads plus member manifest. Every part passed owner-only/nonshared private parent/MIME/size metadata checks before download_raw_file=true/include_base64=false fetch, immediate materialization/size/SHA and whole archive/every member verification, at most3 complete concurrent pipelines. No authentication URL is output or committed. Actual IDs, models, optimizer/RNG tensors, binary unscored predictions and complete logs stay private; raw radar arrays/runtime binaries are not duplicated or put in Git.

Dependency depth1: restore and raw-verify producer insurance01 archive `{BASE_SHA}`, identify overlay/; raw-verify delta02 and apply its delta_payload/ over the exact base overlay into a NEW directory, then run native/own-audit/certificate checks. BASE_ARCHIVE_INDEX.json is included; persistent base index is `uwb_direction1/results/cloud_20261105_held6_producer_insurance01/held6_producer_backup/CLOUD_DOWNLOAD_INDEX.json`. Original initial/source/seed/ID/data/maps/runtime and preregistration still required. Preserve the base and omitted forensic paths, all existing/newer/live state, and all cumulative caps/charges. Only confirmed loss and inactive prior identities permit original --resume with the auditor's OWN state; never shortcut to producer-final state or refit completed models. The preceding held6 producer backup had no transport error; historical held5 upload failures were already recovered and do not imply a scientific negative.

This is intermediate audit insurance, not a completed three-arm fold/model set or final scoring authority. After full replay/output audits/fusion, create a non-delta complete-fold archive to end the dependency. Held6 cost at post-backup observation is{cost['wall']:.6f}wall/{cost['cpu']:.6f}CPU seconds, within unchanged43200wall/172800CPU per-fold and216000wall/864000CPU five-fold caps. No reset/rescue configuration is introduced. All five newfolds/three arms/audits/frozen outputs and FULL final private recoverability precede once-only source-six diagnostics and unchanged gates. Source1 reused without rescoring, users7–8 excluded from new-method selection and repeated source1/eight-user development exploratory. No new accuracy estimate, eight-fold improvement, independent-validation claim or original contribution is established by this milestone.
'''
    (r/'RUN_REPORT.md').write_text(report)
    updates=dict(state='source6_held6_both_producers_complete_TimeMIL_output_audit_pass_Shuffle_replay_running',latest_report=RP.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=True,background_training_pid=573,background_training_observed_unix=o['observed_unix'],background_training_status_note=f'Same controller573/bootad409b48; both held6 producers and TimeMIL selected-output audit complete with actual exit0. Independent Shuffle100 replayPID{audit["pid"]} active, unfinished. Complete producers/own{e} checkpoint reconstructed/native-verified and privately raw-backed by depth1 delta02 over producer insurance01. No new diagnostic accuracy/originality result or scientific/budget change.',compute='2026-11-05 bootad409b48-89ce-4704-beee-3658103d5856; CPU quota4/RAM16GiB; Python3.12.14 NumPy2.3.5 Torch2.15.0+cpu; actual held6 independent Shuffle replay4threads, both producers complete; no GPU')
    key='source6_held6_audit_delta02_20261105';insert=dict(report=updates['latest_report'],observation=RP.removeprefix('uwb_direction1/')+'/POST_BACKUP_LIVE_OBSERVATION.json',private_download_index=RP.removeprefix('uwb_direction1/')+'/delta02_backup/CLOUD_DOWNLOAD_INDEX.json',archive_sha256=arc,base_archive_sha256=BASE_SHA,delta_chain_depth=1,current_payloads=len(table),changed_payloads=len(d['changed']),captured_own_replay_epoch=e,TimeMIL_output_audit_passed=True,TimeMIL_full64_training_replayed=False,full100_Shuffle_replay_complete=False,budget_reset=False,new_accuracy_result=False,original_contribution_validated=False)
    paths=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
    for path in paths:
        f=G/path;body=f.read_text()
        if path.endswith('STATUS.json'):
            assert '"'+key+'"' not in body
            for k,val in updates.items():
                rx=r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)'
                body,n=re.subn(rx,lambda z:z[1]+json.dumps(val,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1,k
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
    return dict(state='held6_complete_producers_own_audit_delta_evidence_prepared',files=len(files),archive_sha256=arc,dependency_depth=1,captured_own_replay_epoch=e,new_accuracy_result=False)
if __name__=='__main__':
    assert len(sys.argv)==2;commands={'prepare':prepare,'archive':archive,'verify':verify_backup,'publication':publication};print(json.dumps(commands[sys.argv[1]]()))
