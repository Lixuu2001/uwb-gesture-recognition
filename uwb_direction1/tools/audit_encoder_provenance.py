"""Check recorded fold-specific encoder ancestry, not absent weight replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_restored import digest, dump, expected_parts

def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    y=np.load(args.data/'cache/y_clutterremoved.npy'); u=np.load(args.data/'cache/subj_clutterremoved.npy')
    records=[]
    for h in range(1,9):
        base=args.data/f'experiments/p1_20260928/baseline_s2026/fold{h}/warmup'
        multi=args.data/f'experiments/p1_20260928/multiview_s2026/fold{h}'
        b=json.loads((base/'result.json').read_text()); m=json.loads((multi/'result.json').read_text())
        meta=json.loads((base/'features.json').read_text()); identity=m['identity']; init=identity['initial_checkpoint']
        hashes={k:array_hash(v) for k,v in expected_parts(y,u,h).items()}
        assert identity['fold']==h and b['identity']['fold']==h
        assert identity['partitions']==b['identity']['partitions']==hashes
        assert init['model_identity']==b['identity']
        assert init['right_sha256']==b['files']['right.pt']
        assert init['result_sha256']==digest(base/'result.json')
        assert meta['identity']['model_identity']==b['identity']
        assert meta['identity']['checkpoint']==b['files']['right.pt']
        assert meta['sha256']==digest(base/'features.npz')
        assert m['files']['predictions.npz']==digest(multi/'predictions.npz')
        assert set(m['training_subjects'])==set(range(1,9))-{h,h%8+1}
        assert m['validation_subject']==h%8+1 and m['test_subject']==h
        history=m['history']; best=min(history,key=lambda e:e['val_ce']) if history and 'val_ce' in history[0] else None
        records.append(dict(held_user=h,validation_user=h%8+1,source_users=m['training_subjects'],
                            baseline_weight_sha256=b['files']['right.pt'],selected_state_sha256=m['selected_state_sha256'],
                            cache_sha256=m['files']['predictions.npz'],partitions=hashes,
                            provenance_records_match=True,weights_present=(multi/'best.pt').exists(),
                            selected_epoch=m['best_epoch'],history_fields=list(history[0]) if history else []))
    result=dict(state='recorded_ancestry_verified',folds=records,
                limitations=['Weights not migrated: cannot independently rerun encoder or prove original training beyond recorded provenance and code.',
                             'Outer test users have repeatedly informed historical route decisions; all eight users are development data.',
                             '6 train / 1 validation / 1 outer test; not 7-user-training LOSO.'])
    dump(args.out,result); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
