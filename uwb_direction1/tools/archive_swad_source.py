"""Archive completed audited source pilot; never train or change selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from audit_restored import digest, dump


def archive(runs, out):
    if out.exists(): raise ValueError('Preserve existing output; use new directory')
    q=json.loads((runs/'queue_status.json').read_text())
    assert q['state']=='complete' and q['trajectories']==6 and q['selected_models']==18
    raw=json.loads((runs/'SUMMARY.json').read_text());audit=json.loads((runs/'AUDIT.json').read_text())
    identity=json.loads((runs/'identity.json').read_text());c=json.loads((runs/'protocol.json').read_text())
    assert raw['identity']==audit['identity']==identity and audit['state']=='passed'
    assert [x['held'] for x in audit['runs']]==c['users']
    assert all(x['functional_logits_bitwise_equal'] and x['official_valley_replay_bitwise_equal'] for x in audit['runs'])
    scores={a:[] for a in ['erm','tail','swad']};rows=[];members={}
    for h in c['users']:
        folder=runs/f'held{h}';r=json.loads((folder/'result.json').read_text())
        assert r['identity']==dict(identity,held=h)
        for n,s in r['files'].items(): assert digest(folder/n)==s
        for row in r['history'][1:]:assert digest(folder/f"segment_{row['epoch']:02d}.pt")==row['segment_sha256']
        assert len(r['history'])==c['epochs']+1
        paths=[folder/n for n in ['selected.pt','selection.json','predictions.npz','split.json','result.json']]
        paths+=sorted(folder.glob('segment_*.pt'));assert len(paths)==29
        members[f'held{h}.tar.gz']={str(p.relative_to(runs)):digest(p) for p in paths}
        for arm in scores:
            v=r['metrics'][arm]['macro_accuracy'];scores[arm].append(v)
            rows.append({'source_diagnostic_user':h,'arm':arm,'macro_accuracy':v})
    for a,v in scores.items():np.testing.assert_array_equal(v,raw['summary'][a]['per_user'])
    scores={a:np.array(v) for a,v in scores.items()};g=c['gate']
    diffs={a:scores['swad']-scores[a] for a in ['erm','tail']}
    gate=bool(diffs['erm'].mean()>=g['swad_minus_erm'] and diffs['tail'].mean()>=g['swad_minus_tail'] and all((d>0).sum()>=g['wins_each'] for d in diffs.values()) and scores['swad'].min()>=scores['erm'].min()+g['worst_vs_erm_margin'])
    assert gate==raw['gate_passed']
    boot=np.random.default_rng(20261003).integers(0,6,size=(100000,6));effects={}
    for a,b in [('swad','erm'),('swad','tail'),('tail','erm')]:
        d=(scores[a]-scores[b])*100
        effects[a+'_minus_'+b]={'mean_pp':float(d.mean()),'wins':int((d>0).sum()),'per_user_pp':d.tolist(),
            'descriptive_user_bootstrap_95pct_pp':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
    seconds=sum(row['seconds'] for h in c['users'] for row in json.loads((runs/f'held{h}/result.json').read_text())['history'][1:])
    out.mkdir(parents=True)
    dump(out/'ANALYSIS.json',{'state':'source_pilot_complete_and_audited','gate_passed':gate,'effects':effects,
        'epochs':6*c['epochs'],'epoch_seconds':seconds,'audit_sha256':digest(runs/'AUDIT.json'),
        'summary_sha256':digest(runs/'SUMMARY.json'),'protocol_sha256':identity['protocol'],
        'outer_eight_fold_result':False,'novel_contribution_validated':False,
        'limitations':['six repeatedly used source development users, not untouched confirmation',
                      'overlapping training splits; bootstrap is descriptive','one training seed',
                      'four source training users, unlike original six-source outer protocol',
                      'fixed tail improvement is an existing-method control, not novelty']})
    with (out/'PER_USER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(out/'ARCHIVE_CONTENTS.json',members)
    archives=[]
    for name,manifest in members.items():
        path=out/name
        with tarfile.open(path,'w:gz') as t:
            for rel in manifest:t.add(runs/rel,arcname=rel)
        with tarfile.open(path,'r:gz') as t:
            for rel,sha in manifest.items():assert hashlib.sha256(t.extractfile(rel).read()).hexdigest()==sha
        assert path.stat().st_size<=32*1024*1024
        archives.append({'name':name,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'training_artifact_files':len(manifest)})
    root=Path(__file__).resolve().parents[1]
    common={}
    for n in ['SUMMARY.json','AUDIT.json','identity.json','protocol.json','queue_status.json']:common['runs/'+n]=runs/n
    for n in ['ANALYSIS.json','PER_USER.csv','ARCHIVE_CONTENTS.json']:common[n]=out/n
    for n in ['swad_source_pilot.py','swad_core.py','test_swad_source.py','circular_signal.py','full_view_heads.py','audit_restored.py','audit_circular_signal.py','archive_swad_source.py']:common['tools/'+n]=root/'tools'/n
    for f in (root/'research/sources/swad_official').glob('*'):common[str(f.relative_to(root))]=f
    for n in ['SWAD_SOURCE_PROTOCOL.json','SWAD_REVIEW.md','sources/SWAD_IMPLEMENTATION_INDEX.json']:common['research/'+n]=root/'research'/n
    common['training.log']=Path('/workspace/uwb-swad-source-train-20261003.log')
    path=out/'metadata.tar.gz'
    with tarfile.open(path,'w:gz') as t:
        for rel,f in common.items():t.add(f,arcname=rel)
    with tarfile.open(path,'r:gz') as t:
        for rel,f in common.items():assert hashlib.sha256(t.extractfile(rel).read()).hexdigest()==digest(f)
    archives.append({'name':path.name,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'metadata_files':len(common)})
    receipt={'state':'local_archives_verified','archives':archives,'training_artifact_files':174,
             'includes_all_144_epoch_segments':True,'contains_original_arrays':False,'contains_optimizer_progress':False,
             'optimizer_progress_remains_in_cloud':True,'uploaded':False}
    dump(out/'ARCHIVE_RECEIPT.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();archive(a.runs,a.out)
