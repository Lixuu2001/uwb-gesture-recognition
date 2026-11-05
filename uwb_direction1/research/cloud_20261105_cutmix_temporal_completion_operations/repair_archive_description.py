"""Correct an inherited generic description only; payload/science remain immutable."""
from pathlib import Path
import hashlib,json,sys
B=Path('/workspace/uwb-cutmix-temporal-gap64-20261105');D=B/'full_backup';M=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((D/'MANIFEST.json').read_text());members=json.loads((D/'MEMBERS.json').read_text())['members']
    names=['study/initial.pt','run/train/initial.pt','run/replay/initial.pt'];by={r['path']:r for r in members};assert all(n in by for n in names)
    assert len({by[n]['sha256'] for n in names})==1 and len({by[n]['bytes'] for n in names})==1
    assert all((D/c['name']).stat().st_size==c['bytes'] and sha(D/c['name'])==c['sha256'] for c in m['chunks'])
    original=D/'MANIFEST_AS_BUILT.json';proof=M/'MODELSET_INVENTORY.json'
    if not original.exists():
        assert m['shared_initial_encoder_included_once'] is True
        original.write_bytes((D/'MANIFEST.json').read_bytes());m['shared_initial_encoder_included_once']=False;m['initial_state_lineage_copies']=names;m['description_repair']='inapplicable_generic_shared_initial_encoder_flag_only_no_payload_change';(D/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    else:
        assert m['shared_initial_encoder_included_once'] is False and m['initial_state_lineage_copies']==names
    q=dict(state='archive_description_corrected_payload_unchanged',reason='single full pilot retains three exact initial-state lineage copies; generic included-once flag inapplicable',as_built_manifest_sha256=sha(original),corrected_manifest_sha256=sha(D/'MANIFEST.json'),archive_sha256=m['archive_sha256'],members_manifest_sha256=m['members_manifest_sha256'],initial_state_lineage_paths=names,initial_state_file_sha256=by[names[0]]['sha256'],initial_state_file_bytes=by[names[0]]['bytes'],complete_native_initial_model_included=True,original_initial_guard_raw_chunks_external=True,initial_guard_persistent_index='results/cloud_20261105_cutmix_temporal_gap64_release/initial_backup/CLOUD_DOWNLOAD_INDEX.json',chunk_or_member_bytes_changed=False,scientific_source_or_gate_changed=False,diagnostic_labels_read=False)
    proof.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
