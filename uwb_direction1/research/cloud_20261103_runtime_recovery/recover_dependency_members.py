from pathlib import Path
import io,gzip,tarfile,json,hashlib
base=Path('/workspace/uwb-resnet-dependency-recovery-20261103')
repo=Path('/workspace/uwb-cloud-current/uwb_direction1/results/cloud_20261102_resnet18_finetune/complete_backup')
index=json.loads((repo/'CLOUD_DOWNLOAD_INDEX.json').read_bytes())
members_bytes=(repo/'MEMBERS.json').read_bytes()
assert hashlib.sha256(members_bytes).hexdigest()==index['members_manifest_sha256']
members={r['path']:r for r in json.loads(members_bytes)['members']}
targets={'dependencies/source_ids.npz':Path('/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz'),'reference/predictions_unscored.npz':Path('/workspace/uwb-timemil-20261029/run02/timemil/predictions_unscored.npz'),'reference/SUMMARY.json':Path('/workspace/uwb-timemil-20261029/run02/SUMMARY.json')}
class Prefix(io.RawIOBase):
    def __init__(self):self.i=0;self.part=None
    def readable(self):return True
    def read(self,n=-1):
        assert n>=0
        out=bytearray()
        while len(out)<n and self.i<8:
            if self.part is None:self.part=(base/'parts'/index['chunks'][self.i]['name']).open('rb')
            data=self.part.read(n-len(out))
            if data:out.extend(data)
            else:self.part.close();self.part=None;self.i+=1
        return bytes(out)
found=[]
with gzip.GzipFile(fileobj=Prefix(),mode='rb') as gz:
    with tarfile.open(fileobj=gz,mode='r|') as tar:
        for info in tar:
            if info.name not in targets:continue
            assert info.isfile() and info.size==members[info.name]['bytes']
            data=tar.extractfile(info).read()
            assert hashlib.sha256(data).hexdigest()==members[info.name]['sha256']
            dst=targets[info.name];dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():assert dst.read_bytes()==data
            else:dst.write_bytes(data)
            found.append(dict(member=info.name,bytes=len(data),sha256=members[info.name]['sha256'],path=str(dst)))
            if len(found)==len(targets):break
assert len(found)==3,found
(base/'MEMBER_RECOVERY.json').write_text(json.dumps(dict(state='three_dependency_members_SHA_verified_from_verified_prefix',full_archive_verified=False,labels_values_deserialized=False,members=found),indent=2)+'\n')
print(json.dumps(dict(recovered_members=len(found),full_archive_verified=False,labels_deserialized=False)))
