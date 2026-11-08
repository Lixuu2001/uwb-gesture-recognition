"""Materialize a complete saved fold under an empty isolated target from two verified archives."""
from pathlib import Path,PurePosixPath
import sys,json,tarfile,gzip,hashlib,shutil,argparse
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,sha
from stream_archive import OrderedReader,safe

def extract(manifest,rows,out):
 out=Path(out);out.mkdir(exist_ok=False);reader=OrderedReader(rows);seen=set();expected=None;payload=0
 with reader,gzip.GzipFile(fileobj=reader,mode='rb') as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   for item in tar:
    assert item.isfile() and item.name not in seen;seen.add(item.name);f=tar.extractfile(item)
    if item.name=='MEMBERS.json':
     assert expected is None and item.size<32*1048576;data=f.read();assert hashlib.sha256(data).hexdigest()==manifest['members_manifest_sha256'];expected={r['path']:r for r in json.loads(data)['members']};assert len(expected)==manifest['tracked_payload_files']
    else:
     safe(item.name);assert expected is not None and item.name in expected;r=expected[item.name];assert item.size==r['bytes'];dest=out/item.name;dest.parent.mkdir(parents=True,exist_ok=True)
     with dest.open('xb') as target:shutil.copyfileobj(f,target,1048576)
     assert dest.stat().st_size==r['bytes'] and sha(dest)==r['sha256'];payload+=1
  while gz.read(1048576):pass
 assert not reader.read(1048576) and reader.total==manifest['archive_bytes'] and reader.global_hash.hexdigest()==manifest['archive_sha256'] and reader.i==len(rows)
 assert seen==set(expected)|{'MEMBERS.json'};return payload

def restore(base_manifest,base_rows,supplement_manifest,supplement_rows,destination):
 dest=Path(destination);assert not dest.exists();assert str(dest.resolve()).startswith('/tmp/') or str(dest.resolve()).startswith('/workspace/uwb-swad-isolated-')
 dest.mkdir();base=dest/'base';supplement=dest/'supplement';n=extract(base_manifest,base_rows,base);m=extract(supplement_manifest,supplement_rows,supplement)
 recipe=read(supplement/'RECIPE.json');assert recipe['base_archive_sha256']==base_manifest['archive_sha256'] and recipe['base_members_manifest_sha256']==base_manifest['members_manifest_sha256'] and supplement_manifest['requires_base_archive_sha256']==base_manifest['archive_sha256']
 target=dest/'restored_phase';target.mkdir();seen=set();total=0
 for r in recipe['rows']:
  name=r['phase_relative_path'];safe(name);assert name.startswith('folds/held1/') and name not in seen;seen.add(name)
  source=(base if r['source']=='base' else supplement)/safe(r['archive_member']);assert r['source'] in ['base','supplement'];assert source.is_file() and not source.is_symlink() and source.stat().st_size==r['bytes'] and sha(source)==r['sha256']
  out=target/name;out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();shutil.copyfile(source,out);assert out.stat().st_size==r['bytes'] and sha(out)==r['sha256'];total+=r['bytes']
 actual={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()};assert actual==seen and len(seen)==recipe['files'] and total==recipe['bytes']
 return dict(state='both_raw_archives_extracted_full_completed_fold_physically_reconstructed_all_files_SHA_exact',base_payload=n,supplement_payload=m,restored_files=len(seen),restored_bytes=total,raw_arrays_or_gesture_labels_read=False,models_loaded_or_training_invoked=False,original_active_phase_touched=False,whole_job_resume_authorized=False,complete_all_six=False)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);a.add_argument('--supplement',required=True);a.add_argument('--destination',required=True);q=a.parse_args();base=Path(q.base);sup=Path(q.supplement)
 print(json.dumps(restore(read(base/'MANIFEST.json'),read(base/'READBACK_INPUT.json'),read(sup/'MANIFEST.json'),read(sup/'READBACK_INPUT.json'),q.destination)))
