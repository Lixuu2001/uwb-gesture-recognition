"""Verify and archive frozen sharing study, including exact resume state.

This is postprocessing only. It never constructs/trains a model or changes the
registered decision. Large artifacts stay in verified local archives for later
private Drive upload by the authorized connector.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import os
import tarfile
import time
import numpy as np
from audit_restored import digest, dump

LIMIT=28*1024*1024


def write_bundle(destination, members):
    """Immutable, bounded bundles. Existing output must verify before reuse."""
    manifest={name:{'bytes':p.stat().st_size,'sha256':digest(p)} for name,p in members.items()}
    for name in manifest:
        if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe member path')
    def verify():
        with tarfile.open(destination,'r:gz') as t:
            saved=json.load(t.extractfile('CONTENTS_SHA256.json'))
            assert saved==manifest
            assert set(t.getnames())==set(manifest)|{'CONTENTS_SHA256.json'}
            for name,info in manifest.items():
                f=t.extractfile(name);h=hashlib.sha256();size=0
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
                assert size==info['bytes'] and h.hexdigest()==info['sha256']
        assert destination.stat().st_size<32*1024*1024
    if not destination.exists():
        temp=destination.with_suffix(destination.suffix+'.pending')
        if temp.exists():raise ValueError('Preserve incomplete archive; inspect pending file')
        with tarfile.open(temp,'w:gz') as t:
            for name,p in members.items():t.add(p,arcname=name,recursive=False)
            b=(json.dumps(manifest,indent=2)+'\n').encode();i=tarfile.TarInfo('CONTENTS_SHA256.json');i.size=len(b);t.addfile(i,io.BytesIO(b))
        temp.replace(destination)
    verify()
    return {'name':destination.name,'path':str(destination),'bytes':destination.stat().st_size,
            'sha256':digest(destination),'contents':manifest}


def pack_run(runs, folder, out):
    """Only completed immutable trajectories, never copy a live progress file."""
    r=json.loads((folder/'result.json').read_text());h=r['held_source_user'];arm=r['arm']
    for name,sha in r['files'].items():assert digest(folder/name)==sha
    c=json.loads((runs/'protocol.json').read_text());assert len(r['history'])==c['epochs']+1
    for row in r['history'][1:]:assert digest(folder/f"segment_{row['epoch']:02d}.pt")==row['segment_sha256']
    paths=[folder/n for n in ['result.json','split.json','selection.json','predictions.npz','selected.pt','progress.pt']]
    paths += [folder/f'segment_{e:02d}.pt' for e in range(1,c['epochs']+1)]
    assert len(paths)==30
    out.mkdir(parents=True,exist_ok=True);parts=[];members={};size=0;part=1
    for p in paths:
        n=p.stat().st_size
        if n>LIMIT:raise ValueError('Single artifact exceeds bundling limit')
        if members and size+n>LIMIT:
            parts.append(write_bundle(out/f'{arm}-held{h}-part{part}.tar.gz',members));members={};size=0;part+=1
        members[str(p.relative_to(runs))]=p;size+=n
    if members:parts.append(write_bundle(out/f'{arm}-held{h}-part{part}.tar.gz',members))
    return parts


def finish(runs, baseline, out, log):
    q=json.loads((runs/'queue_status.json').read_text());assert q['state']=='complete' and q['new_trajectories']==12
    identity=json.loads((runs/'identity.json').read_text());c=json.loads((runs/'protocol.json').read_text())
    summary=json.loads((runs/'SUMMARY.json').read_text());audit=json.loads((runs/'AUDIT.json').read_text())
    assert summary['identity']==audit['identity']==identity and audit['state']=='passed'
    wanted={(arm,h) for arm in ['shared128']+c['arms'] for h in c['users']}
    assert {(r['arm'],r['held']) for r in audit['runs']}==wanted and len(audit['runs'])==18
    assert all(r['functional_logits_bitwise_equal'] and r['source_roles_order_selection_verified'] for r in audit['runs'])
    source=Path(__file__).resolve().parent
    assert digest(source/'view_sharing_source.py')==identity['script']
    assert digest(source.parent/'research/VIEW_SHARING_PROTOCOL.json')==identity['protocol']
    for name,sha in identity['helpers'].items():assert digest(source/name)==sha
    out.mkdir(parents=True,exist_ok=True)
    values={};rows=[];seconds=0;archive_rows=[]
    for arm in ['shared128']+c['arms']:
        values[arm]={k:[] for k in ['erm','tail']}
        for h in c['users']:
            folder=baseline/f'held{h}' if arm=='shared128' else runs/arm/f'held{h}'
            r=json.loads((folder/'result.json').read_text())
            if arm=='shared128':assert digest(folder/'result.json')==identity['baseline_result_hashes'][str(h)]
            else:assert r['identity']==dict(identity,arm=arm,held=h)
            for name,sha in r['files'].items():assert digest(folder/name)==sha
            with np.load(folder/'predictions.npz') as pred:
                for k in ['erm','tail']:
                    # Independent arithmetic of per-user existing-class macro accuracy.
                    y=pred['y'];yhat=pred[k].argmax(1)
                    score=float(np.mean([float(np.mean(yhat[y==label]==label)) for label in np.unique(y)]))
                    assert abs(score-r['metrics'][k]['macro_accuracy'])<1e-14
                    values[arm][k].append(score)
                    rows.append({'source_diagnostic_user':h,'architecture':arm,'selection':k,'macro_accuracy':score})
            if arm!='shared128':
                seconds+=sum(row['seconds'] for row in r['history'][1:])
                archive_rows.extend(pack_run(runs,folder,out))
        for k in ['erm','tail']:np.testing.assert_allclose(values[arm][k],summary['summary'][arm][k]['per_user'],atol=1e-14,rtol=0)
    g=c['gate'];effects={};passed=True
    bootstrap=np.random.default_rng(20261003).integers(0,6,size=(100000,6))
    for control,margin in [('shared128',g['untied_minus_shared128']),('shared177',g['untied_minus_shared177'])]:
        u=np.array(values['untied128']['tail']);b=np.array(values[control]['tail']);d=u-b
        ok=d.mean()>=margin and int((d>0).sum())>=g['wins_each'] and u.min()>=b.min()+g['worst_margin_each']
        passed=passed and bool(ok)
        effects[control]={'mean_pp':float(d.mean()*100),'per_user_pp':(d*100).tolist(),'wins':int((d>0).sum()),
            'descriptive_user_bootstrap95_pp':np.quantile(d[bootstrap].mean(1)*100,[.025,.975]).tolist(),'gate_passed':bool(ok)}
    assert passed==summary['gate_passed']
    dump(out/'ANALYSIS.json',{'state':'complete_audited','identity':identity,'gate_passed':passed,'primary_tail_effects':effects,
        'new_trajectories':12,'reused_trajectories':6,'new_epochs':288,'new_epoch_seconds':seconds,
        'outer_eight_fold_result':False,'novel_contribution_validated':False,
        'limitations':['six repeatedly used development users','overlapping training sets; intervals descriptive only','one seed','parameter placement differs between capacity controls']})
    with (out/'PER_USER.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    common={}
    for name in ['SUMMARY.json','AUDIT.json','identity.json','protocol.json','queue_status.json','BASELINE_INDEX.json','LAUNCH.json']:
        common['runs/'+name]=runs/name
    for name in ['ANALYSIS.json','PER_USER.csv']:common[name]=out/name
    for name in list(identity['helpers'])+['view_sharing_source.py','archive_view_sharing.py']:common['tools/'+name]=source/name
    for name in ['VIEW_SHARING_PROTOCOL.json','VIEW_GRADIENT_REVIEW.md','SWAD_SOURCE_PROTOCOL.json','sources/swad_official/LICENSE']:
        common['research/'+name]=source.parent/'research'/name
    for name in ['COST.json','TESTS.json','START_DRIVE_RECEIPT.json','BASELINE_STATE_DRIVE_RECEIPT.json','RESTORE.md']:
        common['provenance/'+name]=source.parent/'results/cloud_20261003_view_sharing'/name
    common['training.log']=log
    archive_rows.append(write_bundle(out/'metadata.tar.gz',common))
    dump(out/'ARCHIVE_RECEIPT.json',{'state':'local_archives_verified','archives':archive_rows,
        'new_training_artifact_files':360,'all_288_epoch_segments':True,'all_12_optimizer_rng_states':True,
        'original_data_arrays_included':False,'uploaded':False,
        'reused_baseline_backups':'provenance/BASELINE_STATE_DRIVE_RECEIPT.json plus original SWAD7 packages'})
    return {'state':'complete','archives':len(archive_rows),'bytes':sum(r['bytes'] for r in archive_rows),'gate_passed':passed,'uploaded':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runs','baseline','out','log','state']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    lock=a.state.with_suffix('.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time()
    dump(a.state,{'state':'waiting','pid':os.getpid(),'started_unix':start})
    try:
        while True:
            q=json.loads((a.runs/'queue_status.json').read_text())
            if q['state']=='complete':break
            if q['state']=='failed':raise RuntimeError('Training queue failed; inspect original error')
            if not a.watch:raise RuntimeError('Study not complete')
            stat=Path('/proc')/str(q['pid'])/'stat'
            if not stat.exists() or stat.read_text().split(') ',1)[1][0]=='Z':raise RuntimeError('Training process no longer live')
            if time.time()-start>8*3600:raise RuntimeError('Exceeded8h waiting bound; inspect without restarting training')
            time.sleep(45)
        dump(a.state,{'state':'archiving','pid':os.getpid()})
        result=finish(a.runs,a.baseline,a.out,a.log)
        dump(a.state,dict(result,pid=os.getpid(),finished_unix=time.time()))
    except BaseException as e:
        dump(a.state,{'state':'failed','pid':os.getpid(),'type':type(e).__name__,'error':str(e)});raise


if __name__=='__main__':main()
