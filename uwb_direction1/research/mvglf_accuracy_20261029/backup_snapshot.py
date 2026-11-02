"""Preserve an immutable epoch snapshot without touching the live worker."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib, io, json, tarfile, time
from pathlib import Path
import torch

B = Path(__file__).resolve().parent
R = B / 'run01'
def sha(data): return hashlib.sha256(data).hexdigest()
def main():
    torch.set_num_threads(1)
    out = B / 'backup_20261029_1210'
    out.mkdir(exist_ok=False)
    state = json.loads((R / 'STATE.json').read_text())
    copies = {}
    last_raw = (R / 'full/last.pt').read_bytes()
    best_raw = (R / 'full/best.pt').read_bytes()
    last = torch.load(io.BytesIO(last_raw), map_location='cpu', weights_only=False)
    best = torch.load(io.BytesIO(best_raw), map_location='cpu', weights_only=False)
    records = [json.loads(s) for s in (R / 'full/epochs.jsonl').read_text().splitlines() if s.strip()]
    records = [r for r in records if r['epoch'] <= last['epoch']]
    assert [r['epoch'] for r in records] == list(range(1, last['epoch'] + 1))
    chosen = max(records, key=lambda r: (r['validation_macro'], -r['validation_loss'], -r['epoch']))
    assert best['epoch'] == chosen['epoch']
    assert tuple(last['best']) == (chosen['validation_macro'], -chosen['validation_loss'], -chosen['epoch'])
    assert last['identity'] == best['identity'] == json.loads((R / 'IDENTITY.json').read_text())
    copies['run01/full/last.pt'] = last_raw
    copies['run01/full/best.pt'] = best_raw
    copies['run01/full/epochs.jsonl'] = ''.join(json.dumps(r) + '\n' for r in records).encode()
    for p in B.iterdir():
        if p.is_file() and p.suffix in {'.json', '.py', '.md', '.log'}:
            copies[p.name] = p.read_bytes()
    for n in ['IDENTITY.json', 'INPUT_HASHES.json', 'SOURCE_CACHE.json', 'split.json', 'STATE.json', 'full/initial.pt', 'full/first_update.pt']:
        copies['run01/' + n] = (R / n).read_bytes()
    cfg = json.loads((B / 'PROTOCOL.json').read_text())
    copies['source_manifest/source_ids.npz'] = Path(cfg['source_manifest']).read_bytes()
    manifest = dict(observed_unix=time.time(), scope='partial_full_arm_epoch_checkpoint',
        full_last_epoch=last['epoch'], selected_epoch=best['epoch'],
        identity=last['identity'], live_state_observed=state,
        diagnostic_labels_read=False, users7_8_labels_used=False,
        raw_arrays_included=False, prepared_maps_included=False,
        resume_note='Use last.pt, not first_update.pt. Restore original source-manifest path or explicitly verify an engineering path amendment. Preserve remaining cumulative budget; never resume while original worker is alive.',
        members={k: dict(bytes=len(v), sha256=sha(v)) for k,v in copies.items()})
    archive = out / 'snapshot.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for k,v in copies.items():
            info=tarfile.TarInfo(k);info.size=len(v);info.mode=0o600;tar.addfile(info,io.BytesIO(v))
        data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('SNAPSHOT.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    raw=archive.read_bytes();chunks=[]
    for i, start in enumerate(range(0,len(raw),7*1024*1024)):
        v=raw[start:start+7*1024*1024];p=out/f'snapshot.part{i:03d}';p.write_bytes(v)
        chunks.append(dict(name=p.name,bytes=len(v),sha256=sha(v)))
    with tarfile.open(archive,'r:gz') as tar:
        for k,v in manifest['members'].items(): assert sha(tar.extractfile(k).read())==v['sha256']
    receipt=dict(snapshot=manifest,archive_bytes=len(raw),archive_sha256=sha(raw),chunks=chunks,all_local_members_verified=True)
    (out/'MANIFEST.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(directory=str(out),epoch=last['epoch'],selected_epoch=best['epoch'],members=len(copies),archive_bytes=len(raw),chunks=len(chunks))))
if __name__=='__main__': main()
