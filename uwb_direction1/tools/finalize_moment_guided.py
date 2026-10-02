"""Supervise the existing frozen queue; audit/summary/archive only, never train."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np
from audit_restored import digest,dump


def summarize(runs,audit_path,baseline,out,archive):
    q=json.loads((runs/'queue_status.json').read_text())
    if q.get('state')!='complete_pending_independent_audit' or q.get('runs')!=16:
        raise ValueError('Require complete sixteen-run queue; no interim summary')
    audit=json.loads(audit_path.read_text())
    assert audit['state']=='verified' and audit['runs']==16 and audit['functional_forward_bitwise_equal']
    assert audit['identity_sha256']==digest(runs/'identity.json')
    assert audit['results_sha256']==digest(runs/'RESULTS.json')
    if out.exists() or archive.exists():raise ValueError('Preserve existing summary/archive')
    raw=json.loads((runs/'RESULTS.json').read_text());identity=json.loads((runs/'identity.json').read_text())
    assert raw['identity']==identity
    rows=[];paths=[runs/n for n in ['identity.json','RESULTS.json','queue_status.json']]
    scores={'static':[],'ordered':[],'legacy':[]}
    for h in range(1,9):
        for arm in ['static','ordered']:
            d=runs/f'fold{h}'/arm;r=json.loads((d/'result.json').read_text())
            assert r['identity']==dict(identity,fold=h,arm=arm)
            for name,sha in r['files'].items():assert digest(d/name)==sha
            paths.extend(d/n for n in ['best.pt','predictions.npz','split.json','result.json'])
            rows.append({'held_user':h,'arm':arm,'macro_accuracy':r['metrics']['macro_accuracy'],
                'selected_epoch':r['selected_epoch'],'epochs_run':r['history'][-1]['epoch'],
                'epoch_seconds':sum(x.get('seconds',0) for x in r['history'])})
            scores[arm].append(r['metrics']['macro_accuracy'])
        bp=baseline/f'fold{h}'/'legacy'/'result.json'
        assert digest(bp)==identity['baseline_result_sha256'][str(h)]
        scores['legacy'].append(json.loads(bp.read_text())['metrics']['macro_accuracy'])
    for arm,v in scores.items():np.testing.assert_array_equal(v,raw['scores'][arm])
    scores={k:np.array(v) for k,v in scores.items()}
    delta=scores['ordered']-scores['static'];base=scores['ordered']-scores['legacy']
    gate=bool(base.mean()>=.01 and delta.mean()>=.005 and (delta>0).sum()>=6
              and scores['ordered'].min()>=scores['legacy'].min())
    assert gate==raw['continue_gate_passed']
    bootstrap=np.random.default_rng(20261002).integers(0,8,size=(100000,8))
    effects={}
    for lhs,rhs in [('ordered','static'),('ordered','legacy'),('static','legacy')]:
        d=(scores[lhs]-scores[rhs])*100
        effects[lhs+'__minus__'+rhs]={'mean_pp':float(d.mean()),'improved_users':int((d>0).sum()),
            'per_user_pp':d.tolist(),'descriptive_user_bootstrap_95pct_pp':np.quantile(d[bootstrap].mean(1),[.025,.975]).tolist()}
    summary={'state':'complete_and_audited','protocol_sha256':identity['protocol_sha256'],
        'summary':{a:{'mean':float(v.mean()),'worst_user':float(v.min()),'per_user':v.tolist()} for a,v in scores.items()},
        'paired':effects,'frozen_continuation_gate_passed':gate,'new_models':16,'training_seeds':[2026],
        'parameters':{'static':273900,'ordered':273900,'legacy':265836},'runs':rows,
        'epochs_total':sum(r['epochs_run'] for r in rows),'epoch_seconds_total':sum(r['epoch_seconds'] for r in rows),
        'audit_sha256':digest(audit_path),'identity_sha256':digest(runs/'identity.json'),
        'novel_contribution_confirmed':False,
        'limitations':['eight repeatedly used development users, not independent confirmation',
            'one model seed; only static/ordered branches parameter matched',
            'legacy reused with audit/hash provenance, not newly retrained',
            'bootstrap is descriptive; overlapping train folds and research selection remain',
            'RF moments are not calibrated physical hand trajectories',
            'known modules concatenation alone establishes no novelty']}
    out.mkdir(parents=True);dump(out/'SUMMARY.json',summary)
    with (out/'PER_USER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hashes={str(p.relative_to(runs)):digest(p) for p in paths};dump(out/'ARCHIVE_CONTENTS.json',hashes)
    archive.parent.mkdir(parents=True,exist_ok=True);pending=archive.with_suffix('.pending')
    if pending.exists():raise ValueError('Preserve pending archive')
    tools=Path(__file__).parent
    with tarfile.open(pending,'w:gz') as tar:
        for p in paths:tar.add(p,arcname='runs/'+str(p.relative_to(runs)))
        tar.add(audit_path,arcname='AUDIT.json')
        for n in ['SUMMARY.json','PER_USER.csv','ARCHIVE_CONTENTS.json']:tar.add(out/n,arcname=n)
        for h in range(1,9):tar.add(baseline/f'fold{h}/legacy/result.json',arcname=f'baseline/fold{h}/result.json')
        for n in ['moment_guided_signal.py','probe_range_moments.py','circular_signal.py','full_view_heads.py',
                  'audit_restored.py','audit_moment_guided.py','finalize_moment_guided.py']:
            tar.add(tools/n,arcname='tools/'+n)
        tar.add(tools.parent/'research/MOMENT_GUIDED_PROTOCOL.json',arcname='research/MOMENT_GUIDED_PROTOCOL.json')
    pending.replace(archive)
    with tarfile.open(archive,'r:gz') as tar:
        for name,sha in hashes.items():assert hashlib.sha256(tar.extractfile('runs/'+name).read()).hexdigest()==sha
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),
        'training_artifact_files':len(paths),'all_archived_training_artifact_hashes_verified':True,
        'contains_original_signal_arrays':False,'contains_optimizer_progress':False,'uploaded':False}
    dump(out/'ARCHIVE_RECEIPT.json',receipt)
    return summary,receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['runs','data','prepared','moments','baseline','audit','summary','archive','state']:
        ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();lock=a.state.with_suffix('.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    tools=Path(__file__).parent;identity=json.loads((a.runs/'identity.json').read_text())
    def check_identity():
        assert digest(tools/'moment_guided_signal.py')==identity['script_sha256']
        assert digest(tools.parent/'research/MOMENT_GUIDED_PROTOCOL.json')==identity['protocol_sha256']
        for n,h in identity['helper_sha256'].items():assert digest(tools/n)==h
    def status(s,**kw):dump(a.state,dict(state=s,pid=os.getpid(),observed_unix=time.time(),**kw))
    try:
        check_identity();started=time.monotonic();status('waiting_for_existing_training')
        while True:
            try:q=json.loads((a.runs/'queue_status.json').read_text())
            except json.JSONDecodeError:time.sleep(1);continue
            if q['state']=='complete_pending_independent_audit':break
            if q['state']=='failed':raise RuntimeError('Training queue failed: '+str(q))
            proc=Path('/proc')/str(q['pid'])
            if not (proc/'stat').exists():raise RuntimeError('Training process disappeared')
            if (proc/'stat').read_text().split(') ',1)[1].split()[0] in ['Z','X']:
                raise RuntimeError('Training exited before complete queue')
            if b'moment_guided_signal.py' not in (proc/'cmdline').read_bytes():raise RuntimeError('Training PID identity mismatch')
            if time.monotonic()-started>8*3600:raise TimeoutError('Eight-hour finalizer wait exceeded')
            time.sleep(15)
        assert q['runs']==16;check_identity();status('auditing_completed_training')
        if not a.audit.exists():
            subprocess.run([sys.executable,str(tools/'audit_moment_guided.py'),'--data',str(a.data),
                '--prepared',str(a.prepared),'--moments',str(a.moments),'--runs',str(a.runs),'--out',str(a.audit)],check=True)
        status('summarizing_and_archiving')
        summary,receipt=summarize(a.runs,a.audit,a.baseline,a.summary,a.archive)
        status('complete',audit=str(a.audit),summary=str(a.summary),archive=str(a.archive),
            archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],
            continuation_gate_passed=summary['frozen_continuation_gate_passed'],uploaded=False)
    except BaseException as e:
        status('failed',error_type=type(e).__name__,error=str(e));raise


if __name__=='__main__':main()
