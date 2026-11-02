"""Audit all source sample roles, saved Gram summaries and first-batch vectors."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_restored import digest, dump, quality_mask


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--models',type=Path,help='Optional restored SWAD runs root, preserving saved MODEL_INDEX bytes')
    a=ap.parse_args();p=a.out;s=json.loads((p/'SUMMARY.json').read_text());c=json.loads((p/'protocol.json').read_text())
    source=Path(__file__).resolve().parents[1]
    assert digest(source/'tools/diagnose_view_gradients.py')==s['script_sha256']
    assert digest(source/'research/VIEW_GRADIENT_PROTOCOL.json')==s['protocol_sha256']
    assert json.loads((source/'research/VIEW_GRADIENT_PROTOCOL.json').read_text())==c
    for n,h in s['files_sha256'].items():assert digest(p/n)==h
    rows=json.loads((p/'BATCHES.json').read_text());ids=json.loads((p/'SAMPLE_IDS.json').read_text())
    models=json.loads((p/'MODEL_INDEX.json').read_text());arrays=np.load(p/'first_batch_gradients.npz')
    y=np.load(a.data/'cache/y_clutterremoved.npy');users=np.load(a.data/'cache/subj_clutterremoved.npy');good=quality_mask(y,users)
    for key,m in models.items():
        model_path=(a.models/f"held{key.split('_')[0]}"/Path(m['path']).name) if a.models else Path(m['path'])
        assert digest(model_path)==m['sha256']
    assert len(rows)==216 and len(models)==18
    cosines=[];shares=[];cancel=[];max_gram_error=0.;vectors=0
    for held in c['source_users']:
        val=c['source_users'][(c['source_users'].index(held)+1)%6]
        train_users=[u for u in c['source_users'] if u not in [held,val]]
        chosen=np.array(ids[str(held)]);assert len(chosen)==len(set(chosen.tolist()))==384
        assert good[chosen].all() and np.isin(users[chosen],train_users).all()
        rng=np.random.default_rng(c['sample_seed']+held)
        expected=np.concatenate([rng.choice(np.flatnonzero(good&(users==u)),96,replace=False) for u in sorted(train_users)])
        np.testing.assert_array_equal(chosen,expected)
        for epoch in c['epochs']:
            group=[r for r in rows if r['held']==held and r['epoch']==epoch]
            assert len(group)==12 and [r['batch_index'] for r in group]==list(range(12))
            np.testing.assert_array_equal(np.concatenate([r['ids'] for r in group]),chosen)
            cs=[];ns=[];cr=[];valid=[]
            for r in group:
                ii=np.array(r['ids']);assert (users[ii]==r['source_user']).all()
                gram=np.array(r['gram']);np.testing.assert_allclose(gram,gram.T,atol=1e-15)
                norms=np.sqrt(np.diag(gram));vv=bool((norms>1e-20).all());valid.append(vv)
                cs.append([gram[i,j]/(norms[i]*norms[j]) if vv else 0 for i,j in [(0,1),(0,2),(1,2)]])
                ns.append(norms/norms.sum());cr.append(np.sqrt(max(gram.sum(),0))/norms.sum())
                assert r['sum_relative_error']<1e-5
                if r['batch_index']==0:
                    g=arrays[f'g_{held}_{epoch}'].astype(np.float64);total=arrays[f'total_{held}_{epoch}'].astype(np.float64)
                    ref=np.einsum('ik,jk->ij',g,g);np.testing.assert_allclose(ref,gram,rtol=1e-12,atol=1e-12)
                    max_gram_error=max(max_gram_error,float(np.max(np.abs(ref-gram))))
                    assert np.linalg.norm(g.sum(0)-total)/(np.linalg.norm(total)+1e-30)<1e-5
                    vectors+=1
            cs=np.array(cs);ns=np.array(ns);valid=np.array(valid);ref=s['trajectories'][str(held)]['epochs'][str(epoch)]
            conflict=float(np.mean(valid&(np.min(cs,axis=1)<=c['screen']['cosine_max'])))
            dominance=float(np.mean(valid&(np.max(ns,axis=1)>=c['screen']['dominant_share_min'])))
            assert ref['conflict_fraction']==conflict and ref['dominance_fraction']==dominance and ref['valid']==int(valid.sum())
            np.testing.assert_allclose(np.median(cs,axis=0),ref['median_cosines_LT_LR_TR'],atol=1e-14)
            np.testing.assert_allclose(np.median(ns,axis=0),ref['median_norm_shares_L_T_R'],atol=1e-14)
            assert abs(np.median(cr)-ref['median_cancellation_ratio'])<1e-14
            cosines.extend(cs);shares.extend(ns);cancel.extend(cr)
    for kind,field in [('conflict','conflict_fraction'),('dominance','dominance_fraction')]:
        passed=[]
        for h in c['source_users']:
            epochs=[e for e in c['epochs'] if s['trajectories'][str(h)]['epochs'][str(e)][field]>=c['screen']['batch_fraction_min']]
            assert epochs==s['trajectories'][str(h)][kind+'_epochs']
            if len(epochs)>=c['screen']['min_epochs']:passed.append(h)
        assert passed==s[kind+'_trajectories']
    expected=len(s['conflict_trajectories'])>=c['screen']['min_trajectories'] or len(s['dominance_trajectories'])>=c['screen']['min_trajectories']
    assert expected==s['screen_passed']
    report={'state':'passed','batches':len(rows),'models_sha256_checked':len(models),'first_batch_vector_gram_replays':vectors,'max_gram_absolute_error':max_gram_error,
        'source_id_uses_per_epoch':sum(len(v) for v in ids.values()),'unique_ids_across_trajectories':len(set(sum(ids.values(),[]))),
        'no_validation_or_diagnostic_ids_used':True,'all_saved_statistics_and_screens_recomputed':True,
        'diagnostic_only_no_classifier_updates':True,'audit_script_sha256':digest(__file__),
        'limits':'18 independently replicated first-batch gradients were checked by producer; post audit replays saved vectors for those18, not all216 backward passes.'}
    dump(p/'POST_AUDIT.json',report);print(json.dumps(report))


if __name__=='__main__':main()
