"""Label-free source-only geometry diagnosis on the recorded cost-probe IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
from circular_signal import analytic, increment_channels
from audit_restored import dump, expected_parts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--benchmark',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    b=json.loads(args.benchmark.read_text())
    y=np.load(args.data/'cache/y_clutterremoved.npy')
    users=np.load(args.data/'cache/subj_clutterremoved.npy')
    assert np.isin(b['source_ids'],expected_parts(y,users,1)['train']).all()
    x=np.load(args.data/'cache/X_clutterremoved.npy',mmap_mode='r')[b['source_ids']]
    z=analytic(x);z-=z.mean(axis=-2,keepdims=True)
    d=z[...,1:,:]*z[...,:-1,:].conj();w=np.sqrt(np.abs(d));theta=np.angle(d)
    c=increment_channels(d)
    wp=w.reshape(*w.shape[:-1],63,3).sum(-1)
    linear=c[:,:,1].reshape(len(x),3,89,63,3).sum(-1)/np.maximum(wp,1e-12)
    phasor=(c[:,:,3]+1j*c[:,:,4]).reshape(len(x),3,89,63,3).sum(-1)
    error=np.abs(np.angle(np.exp(1j*(linear-np.angle(phasor)))))
    record={'scope':'fixed64 fold1 source examples; labels only establish approved source split',
        'source_ids':b['source_ids'],
        'weighted_near_branch_cut_fraction_by_view':[(w[:,v]*(np.abs(theta[:,v])>.9*np.pi)).sum().item()/w[:,v].sum().item() for v in range(3)],
        'weight_fraction_range_triplets_with_linear_circular_error_over_pi_over2':[(wp[:,v]*(error[:,v]>np.pi/2)).sum().item()/wp[:,v].sum().item() for v in range(3)],
        'interpretation':'input geometry diagnostic only; not classification or physical velocity evidence'}
    if args.out.exists():raise ValueError('Use a new output file')
    dump(args.out,record)


if __name__=='__main__':main()
