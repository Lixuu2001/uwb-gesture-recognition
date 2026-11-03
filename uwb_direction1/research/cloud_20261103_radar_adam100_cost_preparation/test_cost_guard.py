"""UNEXECUTED stdlib fixtures for refusing incomplete/stale promotion evidence.

All fixture records and hashes are virtual. No UWB/Drive/process/training access,
no POLICY.json creation, no actual promotion or cost screen. They test structure
and hash-binding decisions; they do not certify a real previous experiment.
"""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cost_guard as g
DRAFT=json.loads(Path(__file__).with_name('COST_POLICY_DRAFT.json').read_text())
CERT_KEYS=['full40_epoch12560_update_replay_exact',
    'all40_validation_scores_selection_exact',
    'best_full_model_and_final_model_optimizer_momentum_rng_exact',
    'all2200_selected_scores_predictions_exact','all6900_raw_RMS_cache_FP64_verified']
class Evidence:
    def __init__(self):
        self.base=Path('/synthetic_not_a_real_study/cost')
        self.root=Path(DRAFT['current_trial_root'])
        self.data={};self.hashes={}
        self.p=copy.deepcopy(DRAFT)
        self.p['draft_state']='selected_frozen_cost_configuration'
        self.put(self.base/'POLICY.json',self.p)
        for name in ['cost_math.py','cost_guard.py','profile_cost.py',
                     'bounded_screen.py','verify_result.py','test_cost_math.py',
                     'test_cost_guard.py','NATIVE_SOURCE_TESTS.log']:
            self.digest(self.base/name)
        native=(self.base/self.p['native_module_directory']).resolve()
        required=[native/'native_adam100.py',native/'test_adam100_native.py',
            Path('/workspace/uwb-shufflenetv2-x1-source-20261103/native_shufflenet.py'),
            Path('/workspace/uwb-shufflenetv2-x1-source-20261103/local_author_sources/shufflenetv2.py'),
            Path('/workspace/uwb-shufflenetv2-x1-source-20261103/weights/shufflenetv2_x1-5666bf0f80.pth'),
            Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py'),
            Path('/workspace/uwb-resnet18-imagenet-source-20261102/native_resnet18.py'),
            Path(sys.executable).resolve()]
        self.put(self.base/'DEPENDENCIES.json',
                 {'files':{str(x):self.digest(x) for x in required}})
        names=['initial12_head.pt','best_model.pt','final_full.pt','split.npz',
            'steps.jsonl','TRAIN_SELECTION.json','FILES.json',
            'validation_unscored.npz','diagnostic_unscored.npz']
        names += ['validation_epoch%02d.npz'%i for i in range(1,41)]
        artifacts={str(self.root/'run01'/n):self.digest(self.root/'run01'/n) for n in names}
        self.cert=dict(state='pass',diagnostic_labels_read=False,
            users7_8_labels_used=False,pixel_count=1038643200,
            artifact_hashes=artifacts,**{k:True for k in CERT_KEYS})
        self.put(self.root/'COMPLETE_AUDIT.json',self.cert)
        self.put(self.root/'FROZEN_HASHES.json',
            {'fixture%d.py'%i:self.digest(self.root/('fixture%d.py'%i)) for i in range(26)})
        self.hashes[str(self.root/'FROZEN_HASHES.json')]=g.CURRENT_FROZEN
        self.registration=dict(state='real_pilot_remote_exact_readback_verified',
            branch_head_verified=True,commit=self.p['previous_real_registration_commit'],
            frozen_sha256=g.CURRENT_FROZEN,all45_changed_files_exact_before_Popen=True)
        self.put(self.root/'REMOTE_PREREGISTRATION.json',self.registration)
        self.initial=dict(state='owner_only_raw_allmember_initial_insurance_verified',
            frozen_sha256=g.CURRENT_FROZEN,shared=False,diagnostic_labels_scored=False)
        self.put(self.root/'INITIAL_PRIVATE_INSURANCE.json',self.initial)
        self.summary=dict(predeclared_gate_passed=False,
            complete_private_modelset_verified=True,users7_8_labels_used=False,
            route='close_exact_full_ShuffleNetV2_x1_SGDM40_B15_seed20261029_no_rescue')
        self.put(self.root/'run01/SUMMARY.json',self.summary)
        self.terms={}
        for name in ['supervisor01','audit_supervisor01']:
            t=dict(exit_code=0,wall_timeout=False,resource_stop_reason=None,
                   cgroup_oom_kill_delta=0)
            self.terms[name]=t
            self.put(self.root/name/'TERMINATION.json',t)
        bindings={'run01/'+n:self.root/'run01'/n for n in [
            'best_model.pt','final_full.pt','validation_unscored.npz',
            'diagnostic_unscored.npz']}
        bindings['study/COMPLETE_AUDIT.json']=self.root/'COMPLETE_AUDIT.json'
        bindings['dependencies/source_ids.npz']=Path(
            '/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz')
        self.members=[dict(path=n,bytes=1,sha256=self.digest(p)) for n,p in bindings.items()]
        self.members.append(dict(path='official/weights/shufflenetv2_x1-5666bf0f80.pth',
            bytes=9218294,sha256='5666bf0f80d1e7f8fb61f6a65742b9c27c37f3208c1f087cdcd7c1ec1aa8b848'))
        backup=self.root/'complete_backup'
        self.put(backup/'MEMBERS.json',{'members':self.members})
        archive='a'*64
        self.manifest=dict(scope=g.CURRENT_SCOPE,archive_sha256=archive,
            members_manifest_sha256=self.digest(backup/'MEMBERS.json'),
            tracked_payload_files=len(self.members))
        self.receipt=dict(state='private_readback_verified',scope=g.CURRENT_SCOPE,
            complete_modelset_verified=True,numeric_audit_certified=True,
            archive_sha256=archive,diagnostic_labels_scored=False,
            members_manifest_sha256=self.manifest['members_manifest_sha256'])
        self.verified=dict(state='remote_archive_and_all_members_verified',
            scope=g.CURRENT_SCOPE,complete_modelset_verified=True,archive_sha256=archive)
        for n,value in [('MANIFEST.json',self.manifest),
                        ('DRIVE_BACKUP_RECEIPT.json',self.receipt),
                        ('READBACK_VERIFIED.json',self.verified)]:
            self.put(backup/n,value)
        activation=dict(state=self.p['required_activation_state'],
            current_trial_closed_without_rescore=True,
            source_summary_sha256=self.digest(self.root/'run01/SUMMARY.json'),
            source_private_receipt_sha256=self.digest(backup/'DRIVE_BACKUP_RECEIPT.json'))
        self.put(self.base/'CURRENT_TRIAL_ACTIVATION.json',activation)
        frozen={n:self.digest(self.base/n) for n in [
            'POLICY.json','DEPENDENCIES.json','CURRENT_TRIAL_ACTIVATION.json',
            'NATIVE_SOURCE_TESTS.log','cost_math.py','cost_guard.py','profile_cost.py',
            'bounded_screen.py','verify_result.py','test_cost_math.py','test_cost_guard.py']}
        self.put(self.base/'COST_HASHES.json',frozen)
        marker=dict(state='remote_cost_preregistration_exact_readback_verified',
            branch_head_verified=True,source_synthetic_tests_executed=True,
            cost_hashes_sha256=self.digest(self.base/'COST_HASHES.json'),
            source_tests_log_sha256=self.digest(self.base/'NATIVE_SOURCE_TESTS.log'),
            activation_sha256=self.digest(self.base/'CURRENT_TRIAL_ACTIVATION.json'))
        self.put(self.base/'REMOTE_COST_PREREGISTRATION.json',marker)
    def digest(self,path):
        key=str(path)
        if key not in self.hashes:
            self.hashes[key]=hashlib.sha256(key.encode()).hexdigest()
        return self.hashes[key]
    def put(self,path,value):
        self.data[str(path)]=value
        self.digest(path)
    def ready(self):
        with patch.object(g,'B',self.base),patch.object(g,'read',
            side_effect=lambda p:copy.deepcopy(self.data[str(p)])),patch.object(
            g,'sha',side_effect=lambda p:self.hashes[str(p)]):
            return g.ready()

class PromotionRefusal(unittest.TestCase):
    def reject(self,change):
        f=Evidence();change(f)
        with self.assertRaises((AssertionError,KeyError,FileNotFoundError,ValueError)):
            f.ready()
    def test_coherent_virtual_fixture_reaches_end_only(self):
        self.assertEqual(Evidence().ready()['batch_size'],20)
    def test_unknown_active_or_passed_previous_trial_cannot_activate(self):
        for value in [None,True]:
            with self.subTest(gate=value):
                self.reject(lambda f:f.summary.update(predeclared_gate_passed=value))
        for name in ['supervisor01','audit_supervisor01']:
            for change in [dict(exit_code=None),dict(exit_code=-15),
                dict(cgroup_oom_kill_delta=1),dict(resource_stop_reason='RSS_resource_limit')]:
                with self.subTest(term=name,change=change):
                    self.reject(lambda f,n=name,c=change:f.terms[n].update(c))
    def test_each_complete_replay_claim_is_required(self):
        for key in CERT_KEYS:
            with self.subTest(key=key):
                self.reject(lambda f,k=key:f.cert.update({k:False}))
    def test_empty_or_missing_models_cannot_pass_an_empty_hash_loop(self):
        self.reject(lambda f:f.cert.update(artifact_hashes={}))
        self.reject(lambda f:f.cert['artifact_hashes'].pop(str(f.root/'run01/best_model.pt')))
    def test_stale_receipt_with_same_scope_cannot_certify_other_model(self):
        self.reject(lambda f:f.members[0].update(sha256='b'*64))
        self.reject(lambda f:f.members[4].update(sha256='b'*64))
        self.reject(lambda f:f.members[5].update(sha256='b'*64))
    def test_archive_membership_must_be_unique_and_hash_bound(self):
        self.reject(lambda f:f.members.append(copy.deepcopy(f.members[0])))
        self.reject(lambda f:f.manifest.update(members_manifest_sha256='b'*64))
        self.reject(lambda f:f.receipt.update(members_manifest_sha256='b'*64))
        self.reject(lambda f:f.manifest.update(archive_sha256='not_sha256'))
    def test_draft_missing_registration_source_mutation_and_label_use_refused(self):
        self.reject(lambda f:f.p.update(draft_state='unselected_not_preregistered_not_executed'))
        self.reject(lambda f:f.data.pop(str(f.base/'REMOTE_COST_PREREGISTRATION.json')))
        self.reject(lambda f:f.data.pop(str(f.base/'CURRENT_TRIAL_ACTIVATION.json')))
        self.reject(lambda f:f.hashes.update({str(f.base/'cost_guard.py'):'b'*64}))
        self.reject(lambda f:f.summary.update(users7_8_labels_used=True))
        self.reject(lambda f:f.cert.update(diagnostic_labels_read=True))
    def test_private_partial_or_other_scope_refused(self):
        self.reject(lambda f:f.receipt.update(complete_modelset_verified=False))
        self.reject(lambda f:f.verified.update(scope='other_trial'))
        self.reject(lambda f:f.receipt.update(diagnostic_labels_scored=True))
    def test_truncated_old_freeze_and_unsafe_archive_names_refused(self):
        self.reject(lambda f:f.data[str(f.root/'FROZEN_HASHES.json')].pop('fixture0.py'))
        self.reject(lambda f:f.members[0].update(path='../best_model.pt'))
    def test_wrong_primary_registration_or_initial_insurance_cannot_activate(self):
        self.reject(lambda f:f.registration.update(commit='7fd8d88d82b1570f90358dca43fc964b9b7eba88'))
        self.reject(lambda f:f.registration.update(all45_changed_files_exact_before_Popen=False))
        self.reject(lambda f:f.initial.update(shared=True))
        self.reject(lambda f:f.initial.update(frozen_sha256='b'*64))
    def test_real_stdlib_sha_reads_all_bytes(self):
        data=b'guard fixture\x00'*500000
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'synthetic.bin';p.write_bytes(data)
            self.assertEqual(g.sha(p),hashlib.sha256(data).hexdigest())
if __name__=='__main__':unittest.main(verbosity=2)
