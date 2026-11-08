"""Lossless object recovery of one completed fold into a new isolated destination."""
from pathlib import Path,PurePosixPath
import sys,json,gzip,tarfile,hashlib,shutil
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,sha
from stream_archive import OrderedReader,safe

def extract(m,rows,out):
 out=Path(out);out.mkdir(exist_ok=False);source=OrderedReader(rows);seen=set();expected=None
 with source,gzip.GzipFile(fileobj=source,mode='rb') as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   for info in tar:
    assert info.isfile() and info.name not in seen;seen.add(info.name);f=tar.extractfile(info)
    if info.name=='MEMBERS.json':
     assert expected is None and info.size<32*1048576;b=f.read();assert hashlib.sha256(b).hexdigest()==m['members_manifest_sha256'];expected={r['path']:r for r in json.loads(b)['members']};assert len(expected)==m['tracked_payload_files']
    else:
     safe(info.name);assert expected is not None and info.name in expected;r=expected[info.name];assert info.size==r['bytes'];dest=out/info.name;dest.parent.mkdir(parents=True,exist_ok=True)
     with dest.open('xb') as target:shutil.copyfileobj(f,target,1048576)
     assert dest.stat().st_size==r['bytes'] and sha(dest)==r['sha256']
  while gz.read(1048576):pass
 assert not source.read(1048576) and source.i==len(rows) and source.total==m['archive_bytes'] and source.global_hash.hexdigest()==m['archive_sha256'] and seen==set(expected)|{'MEMBERS.json'}
 return len(expected)

def restore(base_manifest,base_rows,new_manifest,new_rows,destination):
 dest=Path(destination);assert not dest.exists();assert str(dest.resolve()).startswith('/tmp/') or str(dest.resolve()).startswith('/workspace/uwb-swad-isolated-');dest.mkdir()
 base=dest/'base';new=dest/'new';base_n=extract(base_manifest,base_rows,base);new_n=extract(new_manifest,new_rows,new)
 recipe=read(new/'RECIPE.json');h=recipe['held'];assert h in range(1,7) and recipe['base_archive_sha256']==base_manifest['archive_sha256'] and recipe['base_members_manifest_sha256']==base_manifest['members_manifest_sha256']
 assert new_manifest['requires_base_archive_sha256']==base_manifest['archive_sha256'] and new_manifest['held']==h
 target=dest/'restored_phase';target.mkdir();seen=set();total=0
 for r in recipe['rows']:
  n=safe(r['phase_relative_path']);assert n.startswith(f'folds/held{h}/') and n not in seen;seen.add(n);assert r['source'] in ['base','new']
  p=(base if r['source']=='base' else new)/safe(r['archive_member']);assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
  out=target/n;out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();shutil.copyfile(p,out);assert out.stat().st_size==r['bytes'] and sha(out)==r['sha256'];total+=r['bytes']
 actual={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()};assert actual==seen and len(seen)==recipe['files'] and total==recipe['bytes']
 return dict(state='base_and_new_raw_archives_physically_extracted_exact_fullfold_copies_verified',held=h,base_payload_members=base_n,new_payload_members=new_n,restored_files=len(seen),restored_bytes=total,models_loaded_or_training_invoked=False,raw_UWB_arrays_or_gesture_labels_opened=False,original_active_phase_touched=False,whole_job_resume_authorized=False,complete_all_six=False)
