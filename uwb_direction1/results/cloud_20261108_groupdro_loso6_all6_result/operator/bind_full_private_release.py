"""Bind already exactly published final private recovery metadata to the frozen scorer."""
from pathlib import Path
import sys,subprocess
P=Path('/workspace/uwb-groupdro-loso6-20261108');G=Path('/workspace/uwb-cloud-current');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from stream_archive import verify
ready(True);assert not(P/'FULL_PRIVATE_RELEASE.json').exists() and not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
v=read(P/'all6_private_verified_PUBLICATION_VERIFIED.json');assert v['branch_head_verified'] and v['force'] is False
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==v['commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=G,text=True).strip()
m=read(P/'full_backup/MANIFEST.json');a=read(P/'full_backup/READBACK_VERIFIED.json');assert verify(m,read(P/'full_backup/READBACK_INPUT.json'))==a and a['complete_modelset_verified']
for n,h in read(P/'PRE_SCORE_FREEZE.json')['all_fold_artifact_sha256'].items():assert sha(P/n)==h,n
for n,h in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==h,n
scorer=sha(P/'operator/finalize_cohort_once.py');assert scorer==read(P/'FROZEN_HASHES.json')['operator/finalize_cohort_once.py']
write(P/'FULL_PRIVATE_RELEASE.json',dict(**v,frozen_sha256=sha(P/'FROZEN_HASHES.json'),complete_six_archive_sha256=m['archive_sha256'],once_scorer_sha256=scorer,full_composite_recovery_index_sha256=sha(P/'FULL_COMPOSITE_RECOVERY_INDEX.json'),pre_score_freeze_sha256=sha(P/'PRE_SCORE_FREEZE.json'),complete_all6_private_raw_allmember_verified=True,held_scores_not_computed_at_binding=True,private_release_published_before_any_held_truth_pairing=True))
print(__import__('json').dumps(dict(state='exact_full_private_commit_bound_before_any_scoring',commit=v['commit'],archive_sha256=m['archive_sha256'])))
