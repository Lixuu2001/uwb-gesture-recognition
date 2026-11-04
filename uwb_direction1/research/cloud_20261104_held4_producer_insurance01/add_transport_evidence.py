"""Append post-assembly transport forensics; never rebuild/reupload model payloads."""
from pathlib import Path
import sys,json,shutil,hashlib,re,time
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');G=Path('/workspace/uwb-cloud-current');sys.path.insert(0,str(ROOT))
from common import read,write,sha,alive,process_identity
archive=B/'held4_producer_backup';v=read(archive/'READBACK_VERIFIED.json');assert v['state']=='remote_archive_and_all_members_verified'
failure_path=archive/'ERROR_000_1793798343704.json';failure=read(failure_path);row=read(archive/'READBACK_000.json')
assert failure['stage']=='materialize' and failure['error']=='cannot create attachment directory: exec-server transport closed'
assert failure['file_id']==row['file_id']=='1ydECQaXAiu87UF-N3kChuQr1GkZUNtbF'
assert row['recovered_known_uploaded_ID_without_reupload'] and row['download_verified'] and row['raw_fetch'] and row['include_base64'] is False
ident=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(ident) and ident['pid']==573 and ident['startticks']==34322 and ident['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
helper=G/'uwb_direction1/research/cloud_20261103_runtime_recovery/recover_existing_drive_part.js'
evidence=dict(state='post_assembly_transport_failure_recovered_known_uploaded_ID_without_reupload',observed_unix=time.time(),failure=failure,recovery=dict(file_id=row['file_id'],part_name=row['name'],bytes=row['bytes'],sha256=row['sha256'],raw_fetch=True,include_base64=False,metadata_owner_private_verified=True,download_and_SHA_verified=True,upload_retried=False),controller=process_identity(573),native_training_not_restarted=True,boot_unchanged=True,scientific_configuration_changed=False,new_accuracy_result=False,scientific_negative=False,whole_archive_raw_allmember_verified=True,archive_sha256=v['archive_sha256'],post_archive_transport_evidence_not_in_model_payload=True,recovery_helper_repository_path=str(helper.relative_to(G)),recovery_helper_sha256=sha(helper))
write(B/'TRANSPORT_RECOVERY.json',evidence)
m=read(B/'PUBLICATION_MANIFEST.json');rp='uwb_direction1/results/cloud_20261104_held4_producer_insurance01';cp='uwb_direction1/research/cloud_20261104_held4_producer_insurance01'
write(G/rp/'TRANSPORT_RECOVERY.json',evidence);write(G/rp/'TRANSPORT_FAILURE.json',failure)
write(G/rp/'TRANSPORT_READBACK_000.json',{k:val for k,val in row.items() if k!='path'})
shutil.copyfile(B/'add_transport_evidence.py',G/cp/'add_transport_evidence.py')
report=G/rp/'RUN_REPORT.md'
with report.open('a') as f:
    f.write('\n## Actual transport fault and recovery after payload assembly\n\nPart000 was uploaded with its known ID retained, but immediate local materialization failed with `cannot create attachment directory: exec-server transport closed`. The worker pool stopped new assignments and preserved the ERROR receipt; already-active parts001/002 finished verification. Current managed runtime remained running/connected; kernel bootad409b48, controller573/startticks34322 and original native training stayed active. This is an observed materialization-channel failure, not an established provider restart or scientific/model failure.\n\nThe existing published known-ID recovery helper rechecked metadata/private parent/MIME/size, fetched RAW with base64disabled, immediately materialized and verified bytes/SHA, and recorded READBACK000 for the SAME uploaded ID, with NO repeat upload. Remaining unused part indices then proceeded under the same max3complete-pipeline limit. All parts/whole archive/every member subsequently passed. TRANSPORT_FAILURE.json, TRANSPORT_RECOVERY.json and TRANSPORT_READBACK_000.json retain the post-assembly evidence separately from the already assembled model payload. No model/config/seed/gate/budget was changed or training restarted.\n')
extra='One backup part encountered an actual local materialization transport-close error after upload; its known ID was retained and raw metadata/bytes/SHA reverified without reupload. Same boot/controller/native training stayed active. Failure/recovery evidence is retained in this report folder separately from the already assembled model archive; no scientific negative or restart is inferred.\n\n'
oldnote=m['note'];m['note']=oldnote+extra
for p in ['uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']:
    f=G/p;body=f.read_text();assert body.startswith(oldnote);f.write_text(m['note']+body[len(oldnote):])
note=m['status_updates']['background_training_status_note']+' One materialization-channel failure was repaired by existing-ID raw verification without reupload; native training was not interrupted.'
m['status_updates']['background_training_status_note']=note
status=G/'uwb_direction1/STATUS.json';body=status.read_text();rx=r'("background_training_status_note"\s*:\s*)("(?:[^"\\]|\\.)*")'
body,n=re.subn(rx,lambda z:z[1]+json.dumps(note,ensure_ascii=False,separators=(',',':')),body,count=1);assert n==1
oldinsert=m['status_insert_text'];insert=json.loads(oldinsert);insert.update(backup_transport_failure_repaired=True,upload_retried=False,native_training_interrupted=False,transport_recovery_evidence=rp.removeprefix('uwb_direction1/')+'/TRANSPORT_RECOVERY.json')
m['status_insert_text']=json.dumps(insert,ensure_ascii=False,indent=2);assert oldinsert in body;status.write_text(body.replace(oldinsert,m['status_insert_text'],1));json.loads(status.read_text())
new_paths=[rp+'/TRANSPORT_RECOVERY.json',rp+'/TRANSPORT_FAILURE.json',rp+'/TRANSPORT_READBACK_000.json',cp+'/add_transport_evidence.py']
known={f['path'] for f in m['files']};assert not known.intersection(new_paths)
m['files'] += [dict(path=p) for p in new_paths]
for f in m['files']:
    raw=(G/f['path']).read_bytes();f['bytes']=len(raw);f['blob_sha1']=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
write(B/'PUBLICATION_MANIFEST.json',m)
print(json.dumps(dict(state=evidence['state'],publication_files=len(m['files']),whole_archive_verified=True,upload_retried=False,native_training_interrupted=False)))
