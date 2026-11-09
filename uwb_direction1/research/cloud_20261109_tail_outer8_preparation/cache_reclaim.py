"""Explicit reclaim of verified redundant transport chunks; never scientific payload."""
from pathlib import Path
import json,hashlib,shutil,datetime
Q=Path(__file__).resolve().parent;P=Path('/workspace/uwb-swad-loso6-20261108');B=P/'full_backup'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()
def main():
    import sys
    assert sys.argv[1]=='--verified-transport-cache-only'
    manifest=json.loads((B/'MANIFEST.json').read_text());rows=json.loads((B/'READBACK_INPUT.json').read_text());private=json.loads((Q/'CACHE_REMOTE_METADATA_VERIFIED.json').read_text())
    assert private['passed'] and private['verified_parts']==247 and private['readback_input_sha256']==sha(B/'READBACK_INPUT.json')
    assert len(rows)==len(manifest['chunks'])==247 and len({r['file_id'] for r in rows})==247
    assert manifest['archive_sha256']=='99453112a56ece88f5d7dca37e6a110978c5202c97d6113012f9aa0526187408'
    # Re-run all-member verification over remote raw bytes previously materialized locally,
    # before any deletion; preserve manifests, all receipt IDs and all scientific originals.
    from stream_archive import verify
    assert verify(manifest,rows)==json.loads((B/'READBACK_VERIFIED.json').read_text())
    frozen=json.loads((P/'FROZEN_HASHES.json').read_text())
    for n,h in frozen.items():assert sha(P/n)==h
    assert sha(P/'SUMMARY.json')=='09b3b0f3bac0280e4b4e6e295b336febf311d06e399303ddcac3793587b950f3'
    payload_refs=json.loads((P/'ALL_SIX_PREDICTIONS_FROZEN.json').read_text())['files']
    for r in payload_refs.values():assert sha(r['path'])==r['sha256']
    paths=[]
    for c,r in zip(manifest['chunks'],rows):
        p=B/c['name'];assert p.parent==B and not p.is_symlink() and p.suffix.startswith('.part')
        assert r['name']==c['name'] and r['sha256']==c['sha256'] and r['bytes']==c['bytes']
        assert p.stat().st_size==c['bytes'] and sha(p)==c['sha256'];paths.append(p)
    assert sum(p.stat().st_size for p in paths)==manifest['archive_bytes']==1811485836
    receipt=dict(state='predelete_full247_raw_allmember_and_remote_owner_only_verified',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=manifest['archive_sha256'],archive_bytes=manifest['archive_bytes'],cache_parts=247,scientific_payload_deleted=False,old_scientific_files_and_result_unchanged=True,permanent_Drive_objects_not_deleted=True,raw_arrays_and_models_not_deleted=True,private_complete_recovery_indexes_retained=True,restoration='Use retained canonical247 Drive IDs and SHA manifests, metadata first/rawfetch/materialize; do not rerun old scorers or training',free_before=shutil.disk_usage(Q).free)
    pre=Q/'CACHE_RECLAIM_PREDELETE.json';assert not pre.exists();pre.write_text(json.dumps(receipt,indent=2)+'\n')
    for p in paths:p.unlink()
    assert all(not p.exists() for p in paths) and all((B/n).exists() for n in ['MANIFEST.json','MEMBERS.json','READBACK_INPUT.json','READBACK_VERIFIED.json'])
    for n,h in frozen.items():assert sha(P/n)==h
    assert sha(P/'SUMMARY.json')=='09b3b0f3bac0280e4b4e6e295b336febf311d06e399303ddcac3793587b950f3'
    for r in payload_refs.values():assert sha(r['path'])==r['sha256']
    receipt.update(scientific_model_output_refs_unchanged=len(payload_refs),state='verified_redundant_transport_chunks_reclaimed_scientific_originals_and_private_objects_retained',free_after=shutil.disk_usage(Q).free,deleted_transport_bytes=1811485836)
    (Q/'CACHE_RECLAIM_RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
