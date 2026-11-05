"""Prepare a fixed next study ONLY after the original Manifold once gate closes it.

No preregistration proof is forged, no model is created, and no fit is launched.
Actual production-source tests/private initial release still precede real fitting.
"""
from pathlib import Path
import ast,hashlib,json,shutil,sys
M=Path('/workspace/uwb-manifold-gap64-20261105');D=Path('/workspace/uwb-regmixup-gap64-draft-20261105');B=Path('/workspace/uwb-regmixup-gap64-20261105')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=json.loads((M/'SUMMARY.json').read_text());state=json.loads((M/'STATE.json').read_text())
    assert summary['finalizer_invocations']==1 and not summary['predeclared_gate_passed'] and summary['route']=='close_exact_Manifold_Mixup_alpha2_random012_GAP64_B64_seed20261029_no_rescue'
    assert (M/'FINALIZE_ONCE.lock').exists() and state['finalizer_consumed'] and state['summary_sha256']==sha(M/'SUMMARY.json')
    for n,h in json.loads((D/'DRAFT_SOURCE_HASHES.json').read_text()).items():assert sha(D/n)==h,n
    assert not B.exists(),'existing next study requires inspection, never overwrite or restart'
    B.mkdir();files=['common.py','worker.py','journal.py','controller.py','data.py','stream_archive.py','DEPENDENCIES.json','PROTOCOL.json','engine.py','synthetic.py','capacity_profile.py','regmixup_math.py']
    for n in files:shutil.copyfile(D/n,B/n)
    shutil.copytree(D/'author',B/'author')
    p=B/'PROTOCOL.json';q=json.loads(p.read_text());q.update(state='fixed_new_published_method_before_any_UWB_fit',prior_Manifold_exact_config_failed_and_closed=True,prior_Manifold_result_sha256=sha(M/'SUMMARY.json'),prospective_native_math_and_alpha_config_commit='3f2086896239c6cbc4556602d7e9b9d6056d9e1f');p.write_text(json.dumps(q,indent=2)+'\n')
    (B/'operator').mkdir()
    for n in ['initial_archive.py','full_archive.py','finalize_once.py']:
        s=(M/'operator'/n).read_text().replace(str(M),str(B)).replace('Manifold_Mixup','RegMixup').replace('Manifold Mixup','RegMixup').replace('manifold_gap64','regmixup_gap64').replace('alpha2_random012','alpha20_clean_plus_mixed')
        s=s.replace("    files['author_review/utils.py']=Path(c['author_utils_reference_path'])", "    for ref in c['author_reference_files']:files['author_review/'+Path(ref).name]=Path(ref)")
        (B/'operator'/n).write_text(s)
    for p in B.rglob('*.py'):ast.parse(p.read_text())
    science=[*files,*['author/'+n for n in ['mixup.py','regmixup.py','LICENSE']]]
    (B/'FROZEN_HASHES.json').write_text(json.dumps({n:sha(B/n) for n in science},indent=2)+'\n')
    (B/'operator/SOURCE_HASHES.json').write_text(json.dumps({n:sha(B/'operator'/n) for n in ['initial_archive.py','full_archive.py','finalize_once.py']},indent=2)+'\n')
    q=dict(state='fixed_next_RegMixup_study_prepared_not_registered_or_released_or_fitted',previous_exact_failure_sha256=sha(M/'SUMMARY.json'),seed=20261029,alpha_beta=[20,20],prospective_fixed_config_commit='3f2086896239c6cbc4556602d7e9b9d6056d9e1f',frozen_sha256=sha(B/'FROZEN_HASHES.json'),new_UWB_fit_started=False,remote_preregistration_exists=False,initial_private_release_exists=False,required_next='Exact latest-parent remote preregistration, tests on THIS production source, fixed resource gate, privately verified initial model/roles before real release')
    (B/'PREPARATION.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
