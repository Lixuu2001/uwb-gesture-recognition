"""Content-checked delta overlays; always reconstruct into a new directory."""
from pathlib import Path,PurePosixPath
import json,hashlib,shutil
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def safe(name):
    p=PurePosixPath(name);assert name and str(p)==name and not p.is_absolute() and '..' not in p.parts and name!='PAYLOAD_HASHES.json';return name
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def create(base,current,dest,base_archive_sha256):
    base=Path(base);current=Path(current);dest=Path(dest);dest.mkdir(exist_ok=False)
    old=read(base/'PAYLOAD_HASHES.json');new=read(current/'PAYLOAD_HASHES.json');changed={}
    for name,h in new.items():
        safe(name);src=current/name;assert src.is_file() and not src.is_symlink() and sha(src)==h,name
        if old.get(name)!=h:
            p=dest/'changed'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p);assert sha(p)==h;changed[name]=h
        else:assert sha(base/name)==h,name
    write(dest/'CURRENT_PAYLOAD_HASHES.json',new)
    plan=dict(state='delta_over_one_verified_base_overlay_not_final_scoring_authority',base_archive_sha256=base_archive_sha256,base_payload_table_sha256=sha(base/'PAYLOAD_HASHES.json'),current_payload_table_sha256=sha(dest/'CURRENT_PAYLOAD_HASHES.json'),changed=changed,omitted_prior_paths=sorted(set(old)-set(new)),base_input_modified=False,creates_new_reconstruction_directory=True,diagnostic_labels_read=False,new_model_fit=False)
    write(dest/'DELTA.json',plan);return plan
def apply(base,delta,out):
    base=Path(base);delta=Path(delta);out=Path(out);p=read(delta/'DELTA.json')
    assert sha(base/'PAYLOAD_HASHES.json')==p['base_payload_table_sha256']
    assert sha(delta/'CURRENT_PAYLOAD_HASHES.json')==p['current_payload_table_sha256']
    new=read(delta/'CURRENT_PAYLOAD_HASHES.json');old=read(base/'PAYLOAD_HASHES.json')
    for name in new:safe(name)
    assert set(p['changed'])<=set(new)
    # Validate every input before creating any reconstruction; preserve base.
    for name,h in new.items():
        src=delta/'changed'/name if name in p['changed'] else base/name
        assert (p['changed'].get(name) if name in p['changed'] else old.get(name))==h
        assert src.is_file() and not src.is_symlink() and sha(src)==h,name
    out.mkdir(exist_ok=False)
    for name,h in new.items():
        src=delta/'changed'/name if name in p['changed'] else base/name;dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(dest)==h
    shutil.copyfile(delta/'CURRENT_PAYLOAD_HASHES.json',out/'PAYLOAD_HASHES.json')
    return dict(state='all_current_payloads_reconstructed_exactly_without_mutating_base',files=len(new),changed_files=len(p['changed']),omitted_prior_paths_preserved_in_base=p['omitted_prior_paths'],diagnostic_labels_read=False,new_model_fit=False)
if __name__=='__main__':
    import sys
    assert len(sys.argv)==5 and sys.argv[1]=='apply';print(json.dumps(apply(*sys.argv[2:])))
