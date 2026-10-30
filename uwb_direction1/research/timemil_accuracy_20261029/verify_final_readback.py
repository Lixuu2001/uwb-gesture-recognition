"""Read-only completed-pair backup verifier; no partial-epoch schema assumption."""
import hashlib,json,tarfile
from pathlib import Path
from backup_epoch import sha

def verify_readback(output):
    output=Path(output);m=json.loads((output/'MANIFEST.json').read_text())
    rows=json.loads((output/'READBACK_INPUT.json').read_text());assert len(rows)==len(m['chunks'])
    archive=output/'remote_reassembled.tar.gz'
    with archive.open('wb') as stream:
        for c,r in zip(m['chunks'],rows):
            p=Path(r['path']);assert c['name']==r['name']
            assert p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
            stream.write(p.read_bytes())
    assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
    with tarfile.open(archive) as tar:
        data=tar.extractfile('MEMBERS.json').read()
        assert hashlib.sha256(data).hexdigest()==m['members_manifest_sha256']
        manifest=json.loads(data);members=manifest['members'];expected={r['path'] for r in members}|{'MEMBERS.json'}
        assert len(members)==m['tracked_payload_files'] and set(tar.getnames())==expected and len(tar.getmembers())==len(expected)
        for r in members:
            data=tar.extractfile(r['path']).read()
            assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
    result=dict(state='remote_archive_and_all_members_verified',archive_sha256=m['archive_sha256'],
                chunks_verified=len(rows),tracked_payload_members_verified=len(members),archive_members=len(expected),
                epochs_per_arm=m['epochs_per_arm'],arms=m['arms'],verifier_sha256=sha(Path(__file__)))
    (output/'READBACK_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    import sys
    verify_readback(Path(sys.argv[1]))
