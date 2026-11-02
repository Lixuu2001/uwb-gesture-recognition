"""Independent saved-score metric and selection audit (no research metric helper)."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def macro(y,s):
    pred=np.argmax(s,axis=1)
    return float(np.mean([sum(pred[y==c]==c)/sum(y==c) for c in sorted(set(y.tolist()))]))
def main():
    p=argparse.ArgumentParser();p.add_argument('--test',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.test/'RESULTS.json').read_text());checks=0
    for r,run in zip(report['runs'],a.runs):
        assert sha(run/'RESULTS.json')==r['selection_sha256'];checks+=1
        for fold in r['folds']:
            h=fold['held_user'];pred=a.test/f'seed{r["seed"]}/fold{h}_test.npz'
            assert sha(pred)==fold['prediction_sha256'];checks+=1
            with np.load(pred) as data:
                assert len(data['ids'])==len(set(data['ids'].tolist()));checks+=1
                for method in ['original_LTR','additive','joint']:
                    assert np.isclose(macro(data['y'],data[method]),fold['metrics'][method]['macro_accuracy'],atol=1e-14,rtol=0);checks+=1
            selected=json.loads((run/f'fold{h}/validation.json').read_text())
            for method in ['additive','joint']:
                with np.load(run/f'fold{h}/{method}_validation.npz') as data:
                    saved=selected['methods'][method]
                    assert np.isclose(macro(data['y'],data['scores']),saved['metrics']['macro_accuracy'],atol=1e-14,rtol=0)
                    best=max(saved['grid'],key=lambda g:g['macro_accuracy'])
                    assert best['alpha']==saved['alpha'];checks+=2
    artifact_paths=list(a.test.rglob('*.npz'))+[p for run in a.runs for p in run.rglob('*.npz')]
    output=dict(state='verified',checks=checks,artifacts={str(p):sha(p) for p in artifact_paths},artifact_count=len(artifact_paths),bytes=sum(p.stat().st_size for p in artifact_paths))
    a.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='artifacts'}))
if __name__=='__main__':main()
