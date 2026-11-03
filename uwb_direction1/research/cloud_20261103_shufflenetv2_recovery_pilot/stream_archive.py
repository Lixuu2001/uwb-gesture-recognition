"""Bounded-memory tar/gzip chunk writer and full-member recovery verifier.

No payload copy or monolithic tar is written. Verified Drive raw downloads may
atomically replace identical local transport chunks to bound disk duplication.
"""
from pathlib import Path,PurePosixPath
import io,json,hashlib,tarfile,gzip,math,os
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def safe(name):
 p=PurePosixPath(name);assert str(p)==name and not p.is_absolute() and '..' not in p.parts and name!='MEMBERS.json';return name
class Chunks(io.RawIOBase):
 def __init__(self,out,prefix,chunk_size,limit):
  self.out=Path(out);self.prefix=prefix;self.chunk_size=chunk_size;self.limit=limit;self.n=0;self.total=0;self.part=None;self.rows=[];self.global_hash=hashlib.sha256();self.part_hash=None;self.part_bytes=0
 def writable(self):return True
 def write(self,b):
  b=memoryview(b);length=len(b);assert self.total+length<=self.limit,'archive exceeds preregistered upper bound'
  self.global_hash.update(b);self.total+=length
  while b:
   if self.part is None:
    self.name=f'{self.prefix}.part{self.n:03d}';self.part=(self.out/self.name).open('xb');self.part_hash=hashlib.sha256();self.part_bytes=0
   count=min(len(b),self.chunk_size-self.part_bytes);self.part.write(b[:count]);self.part_hash.update(b[:count]);self.part_bytes+=count;b=b[count:]
   if self.part_bytes==self.chunk_size:self.finish_part()
  return length
 def finish_part(self):
  if self.part is not None:self.part.close();self.rows.append(dict(name=self.name,bytes=self.part_bytes,sha256=self.part_hash.hexdigest()));self.part=None;self.n+=1
 def finish(self):self.finish_part();return dict(archive_bytes=self.total,archive_sha256=self.global_hash.hexdigest(),chunks=self.rows)
class OrderedReader(io.RawIOBase):
 def __init__(self,rows):
  self.rows=rows;self.i=0;self.part=None;self.total=0;self.global_hash=hashlib.sha256();self.part_hash=None;self.part_bytes=0
 def readable(self):return True
 def close(self):
  if self.part is not None:self.part.close();self.part=None
  super().close()
 def read(self,size=-1):
  assert size>=0,'reader requires bounded requests'
  out=bytearray()
  while len(out)<size:
   if self.part is None:
    if self.i==len(self.rows):break
    self.part=Path(self.rows[self.i]['path']).open('rb');self.part_hash=hashlib.sha256();self.part_bytes=0
   b=self.part.read(size-len(out))
   if b:self.part_hash.update(b);self.global_hash.update(b);self.part_bytes+=len(b);self.total+=len(b);out.extend(b)
   else:
    r=self.rows[self.i];assert self.part_bytes==r['bytes'] and self.part_hash.hexdigest()==r['sha256'];self.part.close();self.part=None;self.i+=1
  return bytes(out)
def build(out,prefix,files,scope,chunk_size=7340032,limit=218997749):
 out=Path(out);out.mkdir(exist_ok=False);assert files and len(set(files))==len(files)
 members=[]
 for name,p in sorted(files.items()):
  safe(name);p=Path(p);assert p.is_file() and not p.is_symlink();members.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
 manifest_bytes=(json.dumps(dict(members=members),indent=2)+'\n').encode();sink=Chunks(out,prefix,chunk_size,limit)
 with gzip.GzipFile(fileobj=sink,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w|') as tar:
   info=tarfile.TarInfo('MEMBERS.json');info.size=len(manifest_bytes);info.mtime=0;tar.addfile(info,io.BytesIO(manifest_bytes))
   for row in members:
    p=Path(files[row['path']]);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mtime=0
    with p.open('rb') as f:tar.addfile(info,f)
    assert sha(p)==row['sha256'],'source changed while archiving'
 m=dict(state='complete_audited_modelset_streamed_to_chunks',scope=scope,complete_modelset=True,shared_initial_encoder_included_once=True,tracked_payload_files=len(members),members_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),raw_arrays_included=False,runtime_binaries_included=False,whole_job_resume=False,**sink.finish())
 (out/'MEMBERS.json').write_bytes(manifest_bytes);(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');return m
def verify(m,rows):
 assert len(rows)==len(m['chunks']) and all(all(r[k]==c[k] for k in ['name','bytes','sha256']) for r,c in zip(rows,m['chunks']))
 source=OrderedReader(rows);seen=set();expected=None;count=0
 with source, gzip.GzipFile(fileobj=source,mode='rb') as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   for info in tar:
    assert info.isfile() and info.name not in seen;seen.add(info.name);f=tar.extractfile(info)
    if info.name=='MEMBERS.json':
     assert count==0 and info.size<32*1048576;data=f.read();assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256'];expected={x['path']:x for x in json.loads(data)['members']};assert len(expected)==m['tracked_payload_files']
    else:
     safe(info.name);assert expected is not None and info.name in expected;row=expected[info.name];assert row['bytes']==info.size;h=hashlib.sha256();n=0
     while True:
      b=f.read(1048576)
      if not b:break
      h.update(b);n+=len(b)
     assert n==row['bytes'] and h.hexdigest()==row['sha256']
    count+=1
  while gz.read(1048576):pass
 assert not source.read(1048576);assert source.i==len(rows) and source.total==m['archive_bytes'] and source.global_hash.hexdigest()==m['archive_sha256'];assert seen==set(expected)|{'MEMBERS.json'}
 return dict(state='remote_archive_and_all_members_verified',scope=m['scope'],chunks_verified=len(rows),archive_members=count,tracked_payload_members_verified=len(expected),archive_sha256=m['archive_sha256'],complete_modelset_verified=m['complete_modelset'])
def replace_verified_download(part,download,size,digest):
 part=Path(part);download=Path(download);assert part!=download and not part.is_symlink() and not download.is_symlink()
 assert part.stat().st_size==download.stat().st_size==size and sha(part)==sha(download)==digest
 os.replace(download,part);assert part.stat().st_size==size and sha(part)==digest;return str(part)
