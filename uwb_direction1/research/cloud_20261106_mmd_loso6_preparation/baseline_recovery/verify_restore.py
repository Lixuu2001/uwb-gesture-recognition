"""Verify known private GAP archives completely, then safely materialize isolated payloads."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,shutil
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cfg=json.loads((B/'DOWNLOAD_MANIFEST.json').read_text());rows=[];models={}
    for held in range(2,7):
        m=cfg['manifests'][str(held)];archive=B/m['archive_name'];parts=[x for x in cfg['chunks'] if x['held']==held]
        assert not archive.exists(),'Inspect existing verified archive before repeat extraction'
        with archive.open('xb') as dst:
            for c in parts:
                r=json.loads((B/f'READBACK_{c["index"]:03}.json').read_text());p=Path(r['path'])
                assert r['raw_fetch'] and not r['include_base64'] and r['download_verified']
                assert p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
                with p.open('rb') as f:shutil.copyfileobj(f,dst)
        assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
        with tarfile.open(archive,'r:gz') as tf:
            raw=tf.extractfile('MEMBERS.json').read();assert hashlib.sha256(raw).hexdigest()==m['members_manifest_sha256']
            members=json.loads(raw)['members'];assert len(members)==m['tracked_payload_files']
            expected={q['path'] for q in members}|{'MEMBERS.json'};actual=tf.getmembers()
            assert len(actual)==len(expected) and set(x.name for x in actual)==expected and all(x.isfile() for x in actual)
            for x in actual:
                p=PurePosixPath(x.name);assert not p.is_absolute() and '..' not in p.parts
            assert sum(q['bytes'] for q in members)<536870912
            for q in members:
                raw=tf.extractfile(q['path']).read();assert len(raw)==q['bytes'] and hashlib.sha256(raw).hexdigest()==q['sha256'],q['path']
            out=B/f'held{held}_payload';out.mkdir(exist_ok=False)
            for x in actual:
                dst=out/x.name;dst.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(x) as src,dst.open('xb') as f:shutil.copyfileobj(src,f)
            for q in members:assert sha(out/q['path'])==q['sha256']
        best=list(out.rglob(f'held{held}/best.pt'));assert len(best)==1
        fold=best[0].parent
        for n in ['predictions_unscored.npz','initial.pt','first_update.pt','last.pt','split.json','epochs.jsonl','FOLD_COMPLETE.json']:assert (fold/n).exists(),n
        models[str(held)]=str(fold)
        rows.append(dict(held=held,archive_sha256=m['archive_sha256'],chunks_verified=len(parts),payload_members_verified=len(members),archive_members=len(expected),safe_isolated_extract=True))
        print(json.dumps(rows[-1]),flush=True)
    (B/'BASELINE_ARTIFACTS.json').write_text(json.dumps(dict(held1='/workspace/uwb-timemil-20261029/run02/gap',restored_folds=models),indent=2)+'\n')
    (B/'RESTORE_VERIFIED.json').write_text(json.dumps(dict(state='all_five_existing_GAP_native_model_archives_raw_SHA_and_all_members_verified',folds=rows,new_upload=False,new_training=False,no_held_truth_or_predictions_decoded=True),indent=2)+'\n')
if __name__=='__main__':main()
