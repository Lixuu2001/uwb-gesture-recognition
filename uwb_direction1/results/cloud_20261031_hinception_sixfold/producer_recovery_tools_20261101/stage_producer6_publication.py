from pathlib import Path
import hashlib, json

T=Path(__file__).resolve().parent
B=Path('/workspace/uwb-hinception-sixfold-20261031')
P=Path('/workspace/uwb-hinception-producer6-20261101')
BASE=Path('/workspace/uwb-hinception-audited45-first6-publication-20261101/uwb_direction1')
OUT=Path('/workspace/uwb-hinception-producer6-publication-20261101')
S=OUT/'uwb_direction1'
R=Path('results/cloud_20261031_hinception_sixfold')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(rel,value):
 p=S/rel;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes(value if isinstance(value,bytes) else value.encode())
def cp(src,rel): put(rel,src.read_bytes())
def js(rel,obj): put(rel,json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

expected={'STATUS.json':'54f248350edf1b73c10f3e458dfe16f4d056e26d','memory/CLOUD_CONTINUATION.md':'075cf619fa7914e21dc5e6112d41dd03bff9f4b7','memory/PROJECT_MEMORY.md':'a959fc2220080d34596fd7099db16288eeb1af2a','results/cloud_20260930/RUN_REPORT.md':'918f16410d290a98303e32d534051bafde2fd9f0'}
for name,h in expected.items():
 d=(BASE/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()==h
assert not OUT.exists()
obs=read(B/'LIVE_OBSERVATION.json')
assert obs['fully_audited_new_folds']==[2,3,4,5] and obs['completed_producer_folds']==[2,3,4,5,6]
assert obs['background_numeric_audit_running'] and not obs['background_original_training_running']
assert not obs['phase_transitions_or_unverified'] and not obs['source6_summary_file_exists'] and not obs['diagnostic_scoring_performed']
assert obs['folds']['6']['phase']=='numeric_audit' and obs['folds']['6']['active_process']['pid']==63747
m=read(P/'MANIFEST.json');receipt=read(P/'DRIVE_BACKUP_RECEIPT.json');proof=read(P/'READBACK_VERIFIED.json')
assert proof['archive_members']==158 and proof['chunks_verified']==7 and not proof['complete_modelset_verified']
assert sha(P/'DRIVE_BACKUP_RECEIPT.json')=='5657ffd309da4c1449931e088851d708e3b041c8f5061f2eb1367699ce3da3cc'
term=read(B/'jobs/held6/train/TERMINATION.json')
assert term['exit_code']==0 and not term['wall_timeout']
assert not (B/'run01/held6/COMPLETE_AUDIT.json').exists()
note=f'''# 2026-11-01 — 最后用户6固定训练完成并进入完整审计；完整训练产物7私有块/158成员回读通过

先GitHub读取最新098f0e0805fcada7b118cc566fb0d47300ef345e分支四主记录，核查同一可执行云端原controller/jobs/checkpoints。held6原train63069已exit0/no timeout，完成100epoch/7200updates/460000训练出现，strict最低trainingloss选择epoch53；supervisor wall9748.075404300936s/CPU9745.821098s，producer内部wall9743.078748118947s/CPU9744.435731205s。双val+diag2300行预测冻结，不索引诊断标签、未评分。所有5新producer共500epoch/36000updates/2290000appearances均训练完成，但最后一折尚未数值审计完成。

原job63068已自动开始独立fullreplay audit63747。只读观测{obs['observed_utc']}：controller59236/job63068/audit63747实际活跃，completed producer[2,3,4,5,6]，completed fullaudit[2,3,4,5]，pending空/failed空，无phase transition/unverified。当前没有producer训练在运行；held6没有成功audit TERM/COMPLETE_AUDIT，不能把训练/备份通过算成数值审计通过。1465冻结前提重核，未重启训练/重复下载；原100/B64/seed20261029/native模型/Adam/输入/IDs/划分/门槛全部不变。

使用已冻结SHA的通用producer_snapshot.py --held6，只在完成STATE及成功train TERM后核对protocol identity/steps/counts、全部FILES、selected/last/native/checkpoint和双预测SHA，复制后再核。完整6档50295935bytes/7chunks/157payload+MEMBERS=158。先owner-only folder元数据与exact prefix查重，再max3上传/metadata核parent/private/size/MIME，rawfetch(download_raw_file=true,include_base64=false)立即物化校验size/SHA；重组整档及全部成员回读通过，未改分享/未输出认证URL。archiveSHA6d5252fe7374b3c5433105f8f3b92f06276e84d637d90e2781b842009a2a5ec0，memberSHA155aab5aab5a498708a3d98f0bde6a5e7edb61d788bdc5fa746df8a3604d040d，receiptSHA5657ffd309da4c1449931e088851d708e3b041c8f5061f2eb1367699ce3da3cc。

scope=held6_training_complete_producer_recovery_no_numeric_audit_claim，含initial/first/best/last/final model/Adam/BN/metrics/callback/RNG、nativebest/last、准确IDs、全部训练logs、未评分预测及成功train TERM。真实ID/model/predictions始终私有；Git仅代码/hash metadata/恢复索引，原789源码不动，无raw/maps/runtime大数组。private完整训练模型现覆盖1–6，但complete audited source6 modelset仍false，6审计仍待完成；不是整jobresume、source6准确率或原创贡献。

恢复索引producer6_recovery/CLOUD_DOWNLOAD_INDEX.json；producer6_training/保存成功metadata和TERM。更新producer_recovery_tools_20261101/RECOVERY_COVERAGE.json，原4/5审计补充索引继续对应原私有模型，未重传1–5。一个操作脚本序列化函数存储错误发生在首次upload前，已改为保存函数源码后调用，7分块均仅成功上传一次，工程记录ORCHESTRATION_REPAIR_PRODUCER6.json保留；实验无故障/改动。

继续现有audit63747，不手工重启。5新fold全部双预测冻结/fullaudit通过后由固定aggregate统一对比LocalCNN6及GAP6：对每项均需mean至少+1pp、至少4/6严格胜、worst不低于-2pp。随后完整六模型私有回读并发布真实结果；若通过再独立预登记matched消融，若失败关闭确切配置转下一个文献方法。7–8标签不用于方法选择，source反复开发明确探索性；暂无source6/八折提升/原创结论，目标未达，原自动接续保留。

'''
S.mkdir(parents=True)
for rel in ['memory/CLOUD_CONTINUATION.md','memory/PROJECT_MEMORY.md','results/cloud_20260930/RUN_REPORT.md']:
 put(rel,note+(BASE/rel).read_text())
s=read(BASE/'STATUS.json')
s.update(state='hinception_all_source_producers_complete_last6_numeric_audit_pending',background_training_running=False,background_audit_running=True,background_audit_pid=63747,background_audit_supervisor_pid=63068,background_numeric_audit_running=True,background_experiment_queue_running=True)
h=s['hinception_source_sixfold_20261031']
h.update(state='all5_new_producers_complete_last6_full_numeric_audit_active',job_supervisor_pids={'6':63068},worker_pids={},audit_worker_pids={'6':63747},completed_producer_folds=[2,3,4,5,6],live_observed_utc=obs['observed_utc'])
h['observed_epoch_steps']['6']=[100,7200]
h.update(producer6_private_readback_verified=True,producer6_recovery_scope=m['scope'],producer6_recovery_chunks=7,producer6_recovery_members=158,producer6_recovery_index=str(R/'producer6_recovery/CLOUD_DOWNLOAD_INDEX.json'),full_private_training_models_available_for=[1,2,3,4,5,6],private_first_update_only_for=[],latest_producer6_weights_preserved=True,all5_new_producers_training_complete=True,new_producer_epochs_completed=500,new_producer_updates_completed=36000,new_producer_training_appearances=2290000,producer6_selected_epoch=53)
s['next_action_source_confirmation']='Continue existing controller59236/job63068/audit63747 without relaunch. All new producers complete; full audits2-5 passed, audit6 pending. Require full audit6 before fixed aggregate, then complete audited six-model private preservation before final source6 publication; no scores or real matched ablation yet.'
s['latest_report']=str(R/'PRODUCER6_COMPLETION_AND_RECOVERY.md')
js('STATUS.json',s)
put(R/'PRODUCER6_COMPLETION_AND_RECOVERY.md',note)
cp(B/'LIVE_OBSERVATION.json',R/'LIVE_OBSERVATION.json')
for name in ['CLOUD_DOWNLOAD_INDEX.json','DRIVE_BACKUP_RECEIPT.json','MANIFEST.json','READBACK_VERIFIED.json']:
 cp(P/name,R/'producer6_recovery'/name)
for name in ['TRAINING_COMPLETE.json','TRAINING_RESULT.json','ALL_PREDICTIONS_FROZEN.json','FILES.json']:
 cp(B/'run01/held6'/name,R/'producer6_training'/name)
cp(B/'jobs/held6/train/TERMINATION.json',R/'producer6_training/TERMINATION.json')
# Operational code and receipts only. This does not modify the frozen scientific tree.
names=['producer_snapshot.py','observe_current.py','test_guard.py','README.md','BUILD.log','VERIFY.log','GUARD_TESTS.log','BUILD_PRODUCER5.log','VERIFY_PRODUCER5.log','BUILD_PRODUCER6.log','VERIFY_PRODUCER6.log','OBSERVE.log','RECOVERY_COVERAGE.json','REMOTE_PUBLICATION_PRODUCER4.json','REMOTE_PUBLICATION_PRODUCER5.json','ORCHESTRATION_REPAIR_PRODUCER6.json','stage_producer6_publication.py']
hashes={name:sha(T/name) for name in names}
(T/'PREPARATION_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
for name in names+['PREPARATION_HASHES.json']:
 cp(T/name,R/'producer_recovery_tools_20261101'/name)
entries=[{'path':str(p.relative_to(OUT)),'content':p.read_bytes().decode('utf8'),'mode':'100644','type':'blob'} for p in sorted(S.rglob('*')) if p.is_file()]
assert all(e['path'].startswith('uwb_direction1/') and not any(t in e['path'] for t in ['.npz','.pt','.keras','.tar.gz','.part0']) for e in entries)
(OUT/'ENTRIES.json').write_text(json.dumps(entries,ensure_ascii=True))
print(json.dumps({'files':len(entries),'export_chars':(OUT/'ENTRIES.json').stat().st_size,'audit_worker':63747,'scientific_scores':False,'stage':str(OUT)}))
