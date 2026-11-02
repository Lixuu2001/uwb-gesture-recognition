"""Summarize a fully completed, independently audited frozen signal experiment.

No selection or training occurs here. Bootstrap intervals are descriptive
resampling of eight development-user effects, not independent confirmation.
"""
import argparse
import csv
import json
from pathlib import Path
import tarfile

import numpy as np
from audit_restored import digest,dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','audit','historical','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--archive',type=Path)
    args=ap.parse_args()
    state=json.loads((args.runs/'queue_status.json').read_text())
    if state.get('state')!='complete' or state.get('runs')!=24:
        raise ValueError('Refuse interim performance summary: need all24 completed runs')
    audit=json.loads(args.audit.read_text())
    if audit.get('state')!='verified' or not audit.get('full_experiment_complete') or audit['completed_runs_audited']!=24:
        raise ValueError('Need completed independent24-model audit')
    if args.out.exists():raise ValueError('Use a new summary directory')
    if args.archive and args.archive.exists():raise ValueError('Preserve existing archive')
    raw=json.loads((args.runs/'RESULTS.json').read_text())
    historical=json.loads(args.historical.read_text())
    arms=['legacy','angle_control','circular']
    rows=[];artifact_manifest={}
    for held in range(1,9):
        for arm in arms:
            folder=args.runs/f'fold{held}'/arm
            r=json.loads((folder/'result.json').read_text())
            assert r['held_user']==held and r['arm']==arm
            for file,sha in r['files'].items():
                assert digest(folder/file)==sha
                artifact_manifest[str((folder/file).relative_to(args.runs))]=sha
            rows.append({'held_user':held,'arm':arm,'macro_accuracy':r['metrics']['macro_accuracy'],
                         'selected_epoch':r['selected_epoch'],'epochs_run':r['history'][-1]['epoch'],
                         'selected_validation_ce':r['selected_val_ce'],
                         'epoch_seconds':sum(x.get('seconds',0) for x in r['history'])})
    scores={arm:np.array([r['macro_accuracy'] for r in rows if r['arm']==arm]) for arm in arms}
    scores['historical_LTR']=np.array([r['subsets']['LTR']['macro_accuracy'] for r in historical['folds']])
    for arm in arms:np.testing.assert_array_equal(scores[arm],raw['summary'][arm]['per_user'])
    bootstrap=np.random.default_rng(20260930).integers(0,8,size=(100000,8))
    effects={}
    for lhs,rhs in [('circular','legacy'),('circular','angle_control'),('circular','historical_LTR'),
                    ('legacy','historical_LTR'),('angle_control','historical_LTR')]:
        d=(scores[lhs]-scores[rhs])*100
        effects[lhs+'__minus__'+rhs]={'mean_pp':float(d.mean()),'improved_users':int((d>0).sum()),
            'tied_users':int((d==0).sum()),'per_user_pp':d.tolist(),
            'descriptive_user_bootstrap_95pct_pp':np.quantile(d[bootstrap].mean(1),[.025,.975]).tolist()}
    gate=((scores['circular']-scores['legacy']).mean()>=.01
          and (scores['circular']-scores['angle_control']).mean()>=.005
          and (scores['circular']>scores['angle_control']).sum()>=6
          and scores['circular'].mean()>=.9344166666666667)
    assert bool(gate)==raw['continuation_gate_passed']
    report={'state':'complete_and_audited','protocol_sha256':raw['identity']['protocol_sha256'],
        'summary':{a:{'mean_user_macro_accuracy':float(v.mean()),'worst_user_macro_accuracy':float(v.min()),'per_user':v.tolist()} for a,v in scores.items()},
        'paired':effects,'frozen_continuation_gate_passed':bool(gate),'parameters':265836,
        'runs':rows,'runs_count':24,'epochs_total':sum(r['epochs_run'] for r in rows),
        'epoch_seconds_total':sum(r['epoch_seconds'] for r in rows),
        'timing_scope':'epoch timers include train and validation, exclude checkpoint writes, startup, final test and independent audit',
        'all_encoder_parameters_trained_from_scratch':True,'training_seeds':[2026],
        'inference_latency_measured':False,'novel_contribution_confirmed':False,
        'limitations':['eight repeatedly used development subjects; no independent confirmation cohort',
                      'one model seed; changes versus historicalLTR include architecture and training',
                      'circular representation and compactCNN are established building blocks',
                      'bootstrap describes observed user differences; does not resolve shared-training-fold dependence or research selection'],
        'audit_sha256':digest(args.audit),'historical_audit_sha256':digest(args.historical),
        'artifacts_sha256':artifact_manifest}
    args.out.mkdir(parents=True)
    dump(args.out/'SUMMARY.json',report)
    with (args.out/'PER_USER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if args.archive:
        # Explicit whitelist: no original RF/prepared signal arrays; final
        # weights/predictions/IDs/configuration only, plus the source needed.
        paths=[args.runs/name for name in ('protocol.json','identity.json','RESULTS.json','queue_status.json')]
        for held in range(1,9):
            for arm in arms:
                paths.extend(args.runs/f'fold{held}'/arm/name for name in ('best.pt','predictions.npz','split.json','result.json'))
        archive_hashes={str(p.relative_to(args.runs)):digest(p) for p in paths}
        dump(args.out/'ARCHIVE_CONTENTS.json',archive_hashes)
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        temp=args.archive.with_suffix('.pending')
        with tarfile.open(temp,'w:gz') as tar:
            for p in paths:tar.add(p,arcname='runs/'+str(p.relative_to(args.runs)))
            tar.add(args.audit,arcname='AUDIT.json')
            for name in ('SUMMARY.json','PER_USER.csv','ARCHIVE_CONTENTS.json'):
                tar.add(args.out/name,arcname=name)
            source=Path(__file__).parent
            for name in ('circular_signal.py','audit_circular_signal.py','summarize_circular_signal.py','audit_restored.py','full_view_heads.py'):
                tar.add(source/name,arcname='tools/'+name)
            tar.add(source.parent/'research/CIRCULAR_SIGNAL_PROTOCOL.json',arcname='research/CIRCULAR_SIGNAL_PROTOCOL.json')
        temp.replace(args.archive)
        # Verify every expected training artifact from archive bytes as well.
        import hashlib
        with tarfile.open(args.archive,'r:gz') as tar:
            for path,sha in archive_hashes.items():
                f=tar.extractfile('runs/'+path)
                assert hashlib.sha256(f.read()).hexdigest()==sha
        dump(args.out/'ARCHIVE_RECEIPT.json',{'path':str(args.archive),'bytes':args.archive.stat().st_size,
             'sha256':digest(args.archive),'training_artifact_files':len(paths),'all_archived_training_artifact_hashes_verified':True,
             'contains_original_signal_arrays':False,'uploaded':False})
    print(json.dumps({'summary':report['summary'],'continuation_gate_passed':bool(gate)},ensure_ascii=False))


if __name__=='__main__':main()
