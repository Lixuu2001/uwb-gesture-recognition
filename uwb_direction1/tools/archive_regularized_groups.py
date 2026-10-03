"""Immutable postprocessing for the registered regularized group study.
Never edits training, selects models, or uploads data. Requires complete audit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np
from audit_restored import digest, dump
from archive_view_sharing import write_bundle, LIMIT


def pack_run(runs, folder, out):
    c=json.loads((runs/'protocol.json').read_text())
    identity=json.loads((runs/'identity.json').read_text())
    r=json.loads((folder/'result.json').read_text())
    arm=r['arm'];h=r['held_source_user']
    assert arm in c['arms'] and h in c['users']
    assert r['identity']==dict(identity,arm=arm,held=h)
    assert [x['epoch'] for x in r['history']]==list(range(c['epochs']+1))
    for n,sha in r['files'].items():assert digest(folder/n)==sha
    names=['result.json','split.json','selection.json','predictions.npz','selected.pt','progress.pt']
    for row in r['history'][1:]:
        for prefix,key,suffix in [('segment','segment_sha256','pt'),('updates','updates_sha256','npz')]:
            name=f"{prefix}_{row['epoch']:02d}.{suffix}"
            assert digest(folder/name)==row[key];names.append(name)
    assert len(names)==6+2*c['epochs']
    out.mkdir(parents=True,exist_ok=True);rows=[];members={};size=0;part=1
    for name in names:
        p=folder/name;n=p.stat().st_size
        if n>LIMIT:raise ValueError('Single artifact exceeds bundle limit')
        if members and size+n>LIMIT:
            rows.append(write_bundle(out/f'{arm}-held{h}-part{part}.tar.gz',members))
            part+=1;members={};size=0
        members[str(p.relative_to(runs))]=p;size+=n
    if members:rows.append(write_bundle(out/f'{arm}-held{h}-part{part}.tar.gz',members))
    return rows


def finish(runs,out,log):
    assert json.loads((runs/'queue_status.json').read_text())['state']=='complete'
    c=json.loads((runs/'protocol.json').read_text());identity=json.loads((runs/'identity.json').read_text())
    audit=json.loads((runs/'AUDIT.json').read_text());summary=json.loads((runs/'SUMMARY.json').read_text())
    wanted={(arm,h) for arm in c['arms'] for h in c['users']}
    assert audit['passed'] and len(audit['runs'])==len(wanted)
    assert {(a['arm'],a['held']) for a in audit['runs']}==wanted
    assert all(a['functional_logits_bitwise_equal'] and a['source_sampler_and_all_q_updates_rebuilt'] for a in audit['runs'])
    source=Path(__file__).resolve().parent
    assert digest(source/'regularized_group_source.py')==identity['script']
    assert digest(source.parent/'research/REGULARIZED_GROUP_PROTOCOL.json')==identity['protocol']
    for n,sha in identity['helpers'].items():assert digest(source/n)==sha
    values={};archives=[];seconds=0
    for arm in c['arms']:
        values[arm]={k:[] for k in ['erm','tail']}
        for h in c['users']:
            folder=runs/arm/f'held{h}';r=json.loads((folder/'result.json').read_text())
            with np.load(folder/'predictions.npz') as pred:
                for k in values[arm]:
                    y=pred['y'];p=pred[k].argmax(1)
                    score=float(np.mean([np.mean(p[y==v]==v) for v in np.unique(y)]))
                    assert abs(score-r['metrics'][k]['macro_accuracy'])<1e-14
                    values[arm][k].append(score)
            seconds+=sum(x['seconds'] for x in r['history'][1:])
            archives.extend(pack_run(runs,folder,out))
        for k in values[arm]:np.testing.assert_allclose(values[arm][k],summary['scores'][arm][k],atol=1e-14,rtol=0)
    old=[]
    for h in c['users']:
        folder=Path(c['baseline'])/f'held{h}'
        assert digest(folder/'result.json')==identity['baseline_result_hashes'][str(h)]
        r=json.loads((folder/'result.json').read_text())
        assert digest(folder/'predictions.npz')==r['files']['predictions.npz']
        with np.load(folder/'predictions.npz') as p:
            old.append(float(np.mean([np.mean(p['tail'].argmax(1)[p['y']==v]==v) for v in np.unique(p['y'])])))
    np.testing.assert_allclose(old,summary['old_tail'],atol=1e-14,rtol=0)
    effects={};g=c['gate'];primary=np.array(values['dro_reg']['erm'])
    for name,ref in [('uniform_base',values['uniform_base']['erm']),('uniform_reg',values['uniform_reg']['erm']),('old_tail',old)]:
        delta=primary-np.array(ref)
        ok=bool(delta.mean()>=g['mean_gain'] and (delta>0).sum()>=g['wins_each'] and delta.min()>=-g['max_user_decline'])
        assert ok==summary['comparisons'][name]['passed']
        effects[name]=dict(per_user_pp=(100*delta).tolist(),mean_pp=float(100*delta.mean()),wins=int((delta>0).sum()),min_user_pp=float(100*delta.min()),passed=ok)
    passed=all(x['passed'] for x in effects.values());assert passed==summary['passed']
    analysis=dict(state='complete_audited',effects=effects,passed=passed,epoch_seconds=seconds,
                  new_trajectories=len(wanted),new_epochs=len(wanted)*c['epochs'],novel_contribution_validated=False,
                  limits=['Six repeatedly used source-development users; not original outer8','One seed; no independent final test','Established group-risk and regularization methods'])
    dump(out/'ANALYSIS.json',analysis)
    common={'runs/'+n:runs/n for n in ['protocol.json','identity.json','queue_status.json','AUDIT.json','SUMMARY.json']}
    common['ANALYSIS.json']=out/'ANALYSIS.json';common['training.log']=log
    for n in list(identity['helpers'])+['regularized_group_source.py','archive_regularized_groups.py','archive_view_sharing.py']:
        common['tools/'+n]=source/n
    for n in ['REGULARIZED_GROUP_PROTOCOL.json','REGULARIZED_GROUP_RATIONALE.md']:
        common['research/'+n]=source.parent/'research'/n
    for n in ['COST.json','TESTS.json','TRAINING_TESTS.json','RESTORE.md','START_DRIVE_RECEIPT.json','LAUNCH_VERIFIED.json']:
        common['provenance/'+n]=source.parent/'results/cloud_20261003_regularized_groups'/n
    archives.append(write_bundle(out/'metadata.tar.gz',common))
    receipt=dict(state='local_archives_verified',archives=archives,training_artifact_files=len(wanted)*(6+2*c['epochs']),
                 all_epoch_segments_and_update_ids=True,all_optimizer_rng_states=True,raw_data_included=False,uploaded=False)
    dump(out/'ARCHIVE_RECEIPT.json',receipt)
    return dict(state='complete',archives=len(archives),bytes=sum(x['bytes'] for x in archives),passed=passed,uploaded=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['runs','out','log','state']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    lock=a.state.with_suffix('.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();dump(a.state,dict(state='waiting',pid=os.getpid(),started_unix=start))
    try:
        while True:
            # Trainer writes small JSON directly; tolerate observing its write window.
            for attempt in range(4):
                try:
                    q=json.loads((a.runs/'queue_status.json').read_text());break
                except json.JSONDecodeError:
                    if attempt==3:raise
                    time.sleep(0.1)
            if q['state']=='complete':break
            if q['state']=='failed':raise RuntimeError('Training queue failed; preserve and inspect')
            if not a.watch:raise RuntimeError('Study not complete')
            stat=Path('/proc')/str(q['pid'])/'stat'
            if not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z':raise RuntimeError('Training process no longer live')
            if time.time()-start>8*3600:raise RuntimeError('Exceeded 8h bound')
            time.sleep(45)
        dump(a.state,dict(state='archiving',pid=os.getpid()))
        result=finish(a.runs,a.out,a.log);dump(a.state,dict(result,pid=os.getpid(),finished_unix=time.time()))
    except BaseException as e:
        dump(a.state,dict(state='failed',pid=os.getpid(),error=str(e),type=type(e).__name__));raise

if __name__=='__main__':main()
