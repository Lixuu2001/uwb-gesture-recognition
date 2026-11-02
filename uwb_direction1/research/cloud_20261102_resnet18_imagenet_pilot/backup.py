"""Complete audited private modelset preservation, no scoring."""
from pathlib import Path
import argparse,importlib.util,tarfile,shutil,time,json
from common import B,complete,read,write,sha
CORE=Path('/workspace/uwb-hinception-sixfold-preservation-20261031/complete_modelset.py')
def core():
    assert sha(CORE)=='3c9f9968ced54caf2df9b774b57071ee99b9f7b14b792e64073a483ffef2d229'
    s=importlib.util.spec_from_file_location('ResNet18_member_checker',CORE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def build(out):
    c=complete();assert read(B/'STATE.json')['state']=='complete_audited_unscored_requires_private_backup_then_once_finalize'
    assert not (B/'run01/SUMMARY.json').exists();out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[];seen=set();cc=core()
    def copy(src,name):
        src=Path(src);name=str(name);cc.relative_name(name);assert name not in seen and src.is_file() and not src.is_symlink();seen.add(name)
        d=sha(src);dst=payload/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==sha(dst)==d
        members.append(dict(path=name,bytes=dst.stat().st_size,sha256=d))
    for name in read(B/'FROZEN_HASHES.json'):copy(B/name,Path('study')/name)
    for name in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','COMPLETE_AUDIT.json','STATE.json']:copy(B/name,name)
    for group in ['run01','supervisor01','audit_supervisor01']:
        for p in sorted((B/group).iterdir()):
            if p.is_file() and p.suffix!='.tmp':copy(p,Path(group)/p.name)
    copy(c['source_manifest'],'dependencies/source_ids.npz');copy(CORE,'preservation/complete_modelset.py')
    source=Path(c['source_preparation'])
    for p in sorted(source.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:copy(p,Path('source_preparation')/p.relative_to(source))
    assert any(row['path']=='source_preparation/weights/resnet18-f37072fd.pth' and row['sha256']==c['weights_sha256'] for row in members)
    cost=Path(c['cost_workspace'])
    for name in [*read(cost/'COST_HASHES.json'),'COST_HASHES.json','VERIFIED_COST_RESULT.json','VERIFY_RESULT.log','REMOTE_PREREGISTRATION.json','profile01/DECLARATION_BEFORE_TIMING.json','profile01/COST_RESULT.json','supervisor01/TERMINATION.json']:
        copy(cost/name,Path('dependencies/cost_evidence')/name)
    for name,key in [('reference_predictions.npz','reference_predictions'),('reference_summary.json','reference_summary')]:copy(c[key],Path('dependencies/reference')/name)
    meta=dict(state='complete_audited_full_ImageNet_ResNet18_frozen_encoder_LinearSVC_C1_snapshot',scope='complete_audited_ResNet18_ImageNetV1_frozen512_LinearSVC_C1_source_held1',created_unix=time.time(),members=members,shared_frozen_encoder_included_once=True,encoder_weights_sha256=c['weights_sha256'],complete_modelset=True,full_numerical_audit_included=True,diagnostic_labels_scored=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,whole_job_resume=False,limitations='Complete shared official frozen encoder weights once, initial/fitted/final heads, exact IDs/features/scores/logs/audit/source retained. Raw arrays/maps and pinned runtime restored separately; not whole OS or historical runtime parity.')
    write(payload/'MEMBERS.json',meta);archive=out/'resnet18_imagenet_complete_20261102.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(payload.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
    chunks=[]
    with archive.open('rb') as f:
        i=0
        while data:=f.read(7*1024*1024):
            p=out/f'resnet18_imagenet_complete_20261102.part{i:03d}';p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
    manifest={k:v for k,v in meta.items() if k!='members'};manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks)
    write(out/'MANIFEST.json',manifest);cc.verify_members(archive,manifest['members_manifest_sha256'],len(members));print(json.dumps({k:v for k,v in manifest.items() if k!='chunks'}))
def verify(out):
    m=read(out/'MANIFEST.json');rows=read(out/'READBACK_INPUT.json');assert len(rows)==len(m['chunks']);archive=out/'remote_reassembled.tar.gz'
    with archive.open('wb') as f:
        for c,r in zip(m['chunks'],rows):
            p=Path(r['path']);assert r['name']==c['name'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
            with p.open('rb') as g:shutil.copyfileobj(g,f)
    assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256'];v=core().verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files'])
    assert v['metadata']['scope']==m['scope'] and v['metadata']['complete_modelset']
    proof=dict(state='remote_archive_and_all_members_verified',scope=m['scope'],chunks_verified=len(rows),archive_members=v['archive_members'],tracked_payload_members_verified=v['payload_members'],archive_sha256=m['archive_sha256'],complete_modelset_verified=True)
    write(out/'READBACK_VERIFIED.json',proof);print(json.dumps(proof))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['complete','verify']);p.add_argument('--out',required=True,type=Path);a=p.parse_args();build(a.out) if a.action=='complete' else verify(a.out)
