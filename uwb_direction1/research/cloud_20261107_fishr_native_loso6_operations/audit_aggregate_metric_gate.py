"""Audit already scored aggregate counts with rational arithmetic; never reopen predictions or truth."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib, json
P = Path('/workspace/uwb-fishr-native-loso6-20261107')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert (P/'FINALIZE_ONCE.lock').exists()
    assert not (P/'EXACT_METRIC_AND_GATE_AUDIT.json').exists()
    terminal=read(P/'FINAL_RESULT_STATE.json');assert terminal['finalizer_consumed'] and terminal['summary_sha256']==sha(P/'SUMMARY.json')
    s=read(P/'SUMMARY.json');assert s['finalizer_invocations']==1 and not s['users7_8_labels_used']
    per=[];arms={a:[] for a in ['baseline','method']}
    for row in s['rows']:
        h=row['held'];record={'held':h}
        for a in arms:
            m=row[a];support=m['class_support'];recall=m['class_recall'];assert len(support)==len(recall)==12
            correct=[];fractions=[]
            for n,r in zip(support,recall):
                assert isinstance(n,int) and n>=0
                if n==0:assert r is None;correct.append(None);continue
                assert r is not None and 0<=r<=1
                k=round(n*r);assert abs(n*r-k)<1e-10 and 0<=k<=n
                correct.append(k);fractions.append(Q(k,n))
            assert sum(support)==m['n']
            accuracy=Q(sum(k for k in correct if k is not None),m['n']);macro=sum(fractions,Q(0))/len(fractions)
            assert abs(float(accuracy)-m['accuracy'])<=1e-12 and abs(float(macro)-m['present_class_macro_recall'])<=1e-12
            missing=[i for i,n in enumerate(support) if n==0];assert missing==m['missing_classes']
            assert missing==({2:[10,11],6:[3]}.get(h,[]))
            record[a]={'integer_correct_by_class':correct,'support':support,'accuracy_fraction':str(accuracy),'macro_fraction':str(macro)}
            arms[a].append((accuracy,macro,m['n']))
        assert row['baseline']['class_support']==row['method']['class_support']
        per.append(record)
    assert sorted(r['held'] for r in per)==list(range(1,7))
    deltas=[m[1]-b[1] for b,m in zip(arms['baseline'],arms['method'])]
    for a,rows in arms.items():
        expected={'equal_user_mean_accuracy':sum((r[0] for r in rows),Q(0))/6,'equal_user_mean_macro_recall':sum((r[1] for r in rows),Q(0))/6,'worst_user_macro_recall':min(r[1] for r in rows),'pooled_sample_accuracy':sum((r[0]*r[2] for r in rows),Q(0))/sum(r[2] for r in rows)}
        for k,v in expected.items():assert abs(float(v)-s[a][k])<=1e-12
    mean=sum(deltas,Q(0))/6;worst=min(deltas);wins=sum(d>0 for d in deltas)
    assert abs(float(mean)-s['mean_paired_macro_recall_delta'])<=1e-12 and abs(float(worst)-s['worst_paired_macro_recall_delta'])<=1e-12 and wins==s['strict_user_wins']
    g=read(P/'DEVELOPMENT_PROTOCOL.json')['bounded_LOSO_continuation_gate'];tol=Q(str(g['tolerance']))
    checks={'mean':mean+tol>=Q(str(g['mean_paired_macro_recall_gain_min'])),'wins':wins>=g['strict_wins_min'],'worst_paired':worst+tol>=Q(str(g['worst_paired_macro_recall_delta_min']))}
    assert all(checks.values())==s['cohort_spending_gate_passed']
    result={'state':'already_aggregated_metric_counts_and_gate_independently_reconstructed_with_rational_arithmetic','summary_sha256':sha(P/'SUMMARY.json'),'rows':per,'mean_delta_fraction':str(mean),'worst_delta_fraction':str(worst),'strict_wins':wins,'gate_components':checks,'gate_passed':all(checks.values()),'raw_truth_or_predictions_opened':False,'new_inference_or_fits':False,'scorer_invocations_added':0,'source_sha256':sha(__file__)}
    (P/'EXACT_METRIC_AND_GATE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
