"""Add post-assembly transport forensics without altering archived model payloads."""
from pathlib import Path
import sys,json,re,shutil,hashlib
B=Path(__file__).resolve().parent
G=Path('/workspace/uwb-cloud-current')
sys.path.insert(0,'/workspace/uwb-softvote-source6-real-20261103')
from common import read,write,sha
ARCHIVE=B/'held5_complete_backup'
RP='uwb_direction1/results/cloud_20261104_held5_complete'
CP='uwb_direction1/research/cloud_20261104_held5_complete'
def sanitize(x):
    if isinstance(x,dict):return {k:('[redacted]' if re.search(r'url|token|secret|authorization|email',k,re.I) else sanitize(v)) for k,v in x.items()}
    if isinstance(x,list):return [sanitize(v) for v in x]
    if isinstance(x,str):return re.sub(r'Bearer\s+\S+','[redacted bearer]',re.sub(r'https?://\S+','[redacted URL]',x),flags=re.I)
    return x
def main():
    m=read(B/'PUBLICATION_MANIFEST.json');old_note=m['note']
    manifest=read(ARCHIVE/'MANIFEST.json');verified=read(ARCHIVE/'READBACK_VERIFIED.json')
    assert verified['complete_modelset_verified'] and verified['archive_sha256']==manifest['archive_sha256']
    for n,h in read(B/'SOURCE_HASHES.json').items():assert sha(B/n)==h,n
    rows=[]
    for error in sorted(ARCHIVE.glob('ERROR_*.json')):
        failure=read(error);i=failure['i'];suffix=f'{i:03}';chunk=manifest['chunks'][i]
        assert failure['stage']=='upload' and failure['file_id'] is None
        absence=read(ARCHIVE/f'ABSENCE_BEFORE_EXPLICIT_RETRY_{suffix}.json')
        assert absence['folder_filter_results']==absence['global_exact_name_results']==0 and absence['provider_reported_no_next_page']
        uploaded=read(ARCHIVE/f'UPLOAD_{suffix}.json');back=read(ARCHIVE/f'READBACK_{suffix}.json')
        assert uploaded['file_id']==back['file_id'] and back['download_verified'] and back['raw_fetch'] and not back['include_base64']
        assert all(back[k]==chunk[k] for k in ['name','bytes','sha256'])
        assert sha(ARCHIVE/chunk['name'])==chunk['sha256']
        files=[error,ARCHIVE/f'ABSENCE_BEFORE_EXPLICIT_RETRY_{suffix}.json',ARCHIVE/f'EXPLICIT_RETRY_RESPONSE_{suffix}.json',ARCHIVE/f'UPLOAD_{suffix}.json',ARCHIVE/f'READBACK_{suffix}.json']
        files+=sorted(ARCHIVE.glob(f'MISSING_ID_RESPONSE_{suffix}_*.json'))
        files+=sorted(ARCHIVE.glob(f'EXPLICIT_RETRY_STARTED_{suffix}.json'))+sorted(ARCHIVE.glob(f'EXPLICIT_RETRY_RECOVERED_{suffix}.json'))
        assert all(p.exists() for p in files)
        rows.append(dict(index=i,name=chunk['name'],first_upload_response_retained=bool(list(ARCHIVE.glob(f'MISSING_ID_RESPONSE_{suffix}_*.json'))),explicit_upload_retries=1,absence_checked_before_retry=True,known_ID_raw_readback_verified=True,original_source_files={p.name:dict(sha256=sha(p),evidence=sanitize(read(p))) for p in files}))
    assert len({x['index'] for x in rows})==len(rows) and rows
    evidence=dict(state='all_observed_upload_failures_recovered_with_one_absence_checked_retry_per_affected_part',failed_parts=[x['index'] for x in rows],failures=rows,first_original_provider_response_unavailable=True,observed_provider_error='MCP -32603 Internal error in retained responses; first original response was not retained by the old helper',provider_root_cause_determined=False,serial_transfer_did_not_eliminate_upstream_error=True,metadata_and_raw_SHA_recovery_verified=True,archive_sha256=manifest['archive_sha256'],archive_members_verified=verified['archive_members'],all_archive_parts_verified=verified['chunks_verified'],archived_model_payload_changed=False,transport_forensics_added_after_archive_assembly=True,scientific_configuration_changed=False,new_fit_started=False,diagnostic_labels_read=False,users7_8_labels_used=False)
    r=G/RP;c=G/CP
    write(r/'TRANSPORT_RECOVERY.json',evidence)
    for n in ['UPLOAD_RESPONSE_REGRESSION_TESTS.json','MISSING_ID_RECOVERY_GUARD_TESTS.json']:shutil.copyfile(B/n,r/n)
    code=['private_upload_missing_id.js','upload_response_failure_tests.js','recover_missing_id_once.js','missing_id_recovery_guard_tests.js','add_transport_evidence.py']
    for n in code:shutil.copyfile(B/n,c/n)
    write(c/'SOURCE_HASHES.json',{p.name:sha(p) for p in sorted(c.iterdir()) if p.is_file() and p.name!='SOURCE_HASHES.json'})
    note=f'''Backup transport had{len(rows)}actual missing-ID upload failures (parts{evidence['failed_parts']}); retained responses identify MCP -32603 Internal error, including under serial transfer. The first original provider response was not retained by the old helper, so its upstream cause is unknown. Each affected part was retried exactly once only after successful folder-scoped and global exact-name searches returned no file, then persisted by known ID and metadata/private raw SHA-verified. Existing successful parts were reused; no unknown-ID blind retry or training restart occurred. Missing-ID responses are now bounded/redacted/persisted before failure, while known IDs still persist before optional diagnostics. Three response-failure regressions and three recovery refusal guards pass (mock tools only, no model fitting). Complete archive payload/source hashes are unchanged; this transport evidence/code was added after archive assembly. See TRANSPORT_RECOVERY.json.\n\n'''
    m['note']=old_note+note
    key='background_training_status_note';m['status_updates'][key]+=' Three actual upload failures were recovered after exact-name absence checks, one explicit retry each and known-ID private raw verification; provider root cause unknown, no training interruption. Missing-ID evidence retention/refusal guards fixed and tested.'
    insert=json.loads(m['status_insert_text']);old_insert=m['status_insert_text']
    insert.update(transport_upload_failures_recovered=len(rows),transport_failed_parts=evidence['failed_parts'],explicit_upload_retry_limit_per_failed_part=1,provider_root_cause_determined=False,post_assembly_transport_evidence=True)
    m['status_insert_text']=json.dumps(insert,ensure_ascii=False,indent=2)
    for f in [G/'uwb_direction1/memory/CLOUD_CONTINUATION.md',G/'uwb_direction1/memory/PROJECT_MEMORY.md',G/'uwb_direction1/results/cloud_20260930/RUN_REPORT.md']:
        body=f.read_text();assert body.startswith(old_note);f.write_text(m['note']+body[len(old_note):])
    f=G/'uwb_direction1/STATUS.json';body=f.read_text()
    rx=r'("background_training_status_note"\s*:\s*)("(?:[^"\\]|\\.)*")'
    body,count=re.subn(rx,lambda match:match[1]+json.dumps(m['status_updates'][key],ensure_ascii=False,separators=(',',':')),body,count=1);assert count==1
    assert body.count(old_insert)==1;body=body.replace(old_insert,m['status_insert_text'],1);json.loads(body);f.write_text(body)
    f=r/'RUN_REPORT.md';f.write_text(f.read_text()+'\n## Actual transport failure and bounded recovery\n\n'+note+'The folder listing was bounded at1000 items and was not used to establish absence. Provider exact-name searches were the absence evidence. Two retained original failure responses explicitly report the MCP internal error; the first response cannot be reconstructed. No concurrency-specific or scientific cause is inferred. Each recovery used the same private owner-only folder and exact bytes/SHA; the original held6 controller/trainers continued under unchanged caps. Recovery guards refuse uploads if a prior retry is recorded, an exact match exists, or a search fails. Full source-six scoring prerequisites and fixed gates remain unchanged.\n')
    paths={x['path'] for x in m['files']}
    paths.update(str(p.relative_to(G)) for root in [r,c] for p in root.rglob('*') if p.is_file())
    transforms={x['path']:x.get('remote_transform') for x in m['files']}
    m['files']=[]
    for path in sorted(paths):
        raw=(G/path).read_bytes();row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        if transforms.get(path):row['remote_transform']=transforms[path]
        m['files'].append(row)
    write(B/'PUBLICATION_MANIFEST.json',m)
    for n,h in read(B/'SOURCE_HASHES.json').items():assert sha(B/n)==h,n
    print(json.dumps(dict(state='transport_failures_recovery_and_guard_fixes_added_without_changing_archived_models',failed_parts=evidence['failed_parts'],files=len(m['files']),archive_sha256=manifest['archive_sha256'])))
if __name__=='__main__':main()
