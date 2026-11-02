from pathlib import Path
import hashlib,json,shutil,sys
B=Path(__file__).resolve().parent;V=Path('/workspace/uwb-multirocket-20261029/vendor/site-packages');index=json.loads((B/'REMOTE_SOURCE_INDEX.json').read_bytes())
for row in index['files']:
 p=V/row['path'];raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1'];row.update(sha256=hashlib.sha256(raw).hexdigest(),installed_native_source_exact=True);dest=B/'local_native_sources'/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
shutil.copyfile('/workspace/uwb-arsenal-parallel-source-probe-20261102/LICENSE',B/'LICENSE')
(B/'SOURCE_PROVENANCE.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(dict(state='six_pinned_native_source_files_exact',files=len(index['files']),commit=index['commit'],actual_UWB_values_labels_fit=False)))
