from pathlib import Path
import argparse,json,hashlib,os
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--i',type=int,required=True);a.add_argument('--raw',required=True);a.add_argument('--metadata',required=True);v=a.parse_args()
    base=Path('/workspace/uwb-transfer-20261103');base.mkdir(exist_ok=True)
    index=json.loads(Path('/workspace/uwb-cloud-current/uwb_direction1/migration/20260930/CLOUD_DOWNLOAD_INDEX.json').read_bytes());row=index['entries'][v.i]
    m=json.loads(v.metadata)
    assert m['id']==row['id'] and m['shared'] is False
    assert m['parent_ids']==[index['folder_id']]
    assert m['permissions']==[dict(type='user',role='owner')]
    assert m['size_bytes']==row['bytes']
    src=Path(v.raw);assert src.stat().st_size==row['bytes'] and sha(src)==row['sha256']
    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
    dst=base/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256']
    else:os.replace(src,dst)
    assert sha(dst)==row['sha256']
    receipt=dict(index=v.i,file_id=row['id'],path=str(dst),bytes=row['bytes'],sha256=row['sha256'],metadata=m,raw_fetch=True,include_base64=False,immediately_materialized_SHA_verified=True)
    receipt_dir=base/'receipts';receipt_dir.mkdir(exist_ok=True)
    path=receipt_dir/('%03d.json'%v.i)
    if path.exists():assert json.loads(path.read_bytes())==receipt
    else:path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(index=v.i,bytes=row['bytes'],verified=True)))
if __name__=='__main__':main()
