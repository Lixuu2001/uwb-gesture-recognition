"""Exact parent-based metadata/code publication, preserving historical JSON integers."""
from pathlib import Path
import json,hashlib,sys,re,subprocess,shutil,time
B=Path('/workspace/uwb-regmixup-gap-final-meta-20261105');G=Path('/workspace/uwb-cloud-current');M=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def main():
    a=read(sys.argv[1]);parent=a['parent'];stage=a['stage'];note=a['note']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==parent
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
    rp='uwb_direction1/results/cloud_20261105_regmixup_gap64_'+stage;cp=a.get('code_path','uwb_direction1/research/cloud_20261105_regmixup_gap64_preparation')
    r=G/rp;r.mkdir(parents=True)
    for n in a['metadata_files']:
        dest=r/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/n,dest)
    extra=[]
    if a.get('include_source'):
        for n in [*read(B/'FROZEN_HASHES.json'),'FROZEN_HASHES.json']:
            d=G/cp/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/n,d);extra.append(str(d.relative_to(G)))
    (r/'RUN_REPORT.md').write_text('# Fixed RegMixup transfer: '+stage+'\n\n'+note+a.get('details',''))
    updates=dict(state=a['state'],latest_report=rp.removeprefix('uwb_direction1/')+'/RUN_REPORT.md',background_training_running=a.get('running',False),background_training_pid=a.get('pid'),background_training_observed_unix=time.time(),background_training_status_note=a['status_note'])
    updates.update(a.get('scalar_updates',{}))
    key='regmixup_gap64_'+stage+'_20261105';insert=dict(report=updates['latest_report'],method='RegMixup_alpha20_clean_plus_mixed_full_GAP64_seed20261029',original_contribution_validated=False,independent_validation=False,users7_8_labels_used=False,**a['status_insert'])
    paths=['uwb_direction1/STATUS.json','uwb_direction1/memory/CLOUD_CONTINUATION.md','uwb_direction1/memory/PROJECT_MEMORY.md','uwb_direction1/results/cloud_20260930/RUN_REPORT.md']
    for path in paths:
        f=G/path;s=f.read_text()
        if path.endswith('STATUS.json'):
            assert '"'+key+'"' not in s
            for k,v in updates.items():
                s,n=re.subn(r'("'+k+r'"\s*:\s*)("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?)',lambda z:z[1]+json.dumps(v,ensure_ascii=False,separators=(',',':')),s,count=1);assert n==1,k
            s=s.rstrip()[:-1]+',\n  "'+key+'": '+json.dumps(insert,ensure_ascii=False,indent=2)+'\n}\n';json.loads(s)
        else:s=note+s
        f.write_text(s)
    assert not subprocess.check_output(['git','diff','--name-only',parent,'--','uwb_direction1/original'],cwd=G,text=True).strip()
    files=[]
    for path in paths+extra+[str(f.relative_to(G)) for f in sorted(r.rglob('*')) if f.is_file()]:
        raw=(G/path).read_bytes();assert not path.endswith(('.pt','.pth','.npz','.npy','.bin','.jsonl'))
        row=dict(path=path,bytes=len(raw),blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
        if path in paths:row['remote_transform']='status' if path.endswith('STATUS.json') else 'prepend'
        files.append(row)
    manifest=dict(root=str(G),parent=parent,files=files,note=note,status_updates=updates,status_insert_key=key,status_insert_text=json.dumps(insert,ensure_ascii=False,indent=2))
    p=M/(stage+'_PUBLICATION.json');p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(manifest=str(p),files=len(files),stage=stage)))
if __name__=='__main__':main()
