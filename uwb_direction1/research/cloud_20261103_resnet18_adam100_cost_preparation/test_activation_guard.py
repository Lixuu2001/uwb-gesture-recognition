"""Virtual metadata fixtures only; no real scores, models, Drive or fitting."""
from pathlib import Path
import copy,tempfile,unittest
from unittest.mock import patch
import activation_guard as guard

class Promotion(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.out=self.root/'run01';self.out.mkdir();self.audit=self.root/'audit01';self.audit.mkdir()
        self.backup=self.root/'complete_backup';self.backup.mkdir()
        self.summary=dict(predeclared_gate_passed=False,complete_private_modelset_verified=True,
            users7_8_labels_used=False,route=guard.FAIL_ROUTE)
        self.cert=dict(state='pass',diagnostic_labels_read=False,
            all100_replay_epoch_journals_committed=True,
            checkpoint45_operational_continuation_disclosed=True,pixel_count=1038643200)
        self.targets={'run01/'+n:self.out/n for n in [
            'initial12_head.pt','final_model.pt','final_full.pt','split.npz','steps.jsonl',
            'TRAIN_ENDPOINT.json','validation_unscored.npz','diagnostic_unscored.npz']}
        self.targets['study/COMPLETE_AUDIT.json']=self.root/'COMPLETE_AUDIT.json'
        self.targets['audit01/PROGRESS.json']=self.audit/'PROGRESS.json'
        for i in range(1,101):self.targets[f'audit01/epoch{i:03d}.json']=self.audit/f'epoch{i:03d}.json'
        self.ids=self.root/'virtual_ids';self.targets['dependencies/source_ids.npz']=self.ids
        for p in self.targets.values():p.write_bytes(('virtual '+p.name).encode())
        self.c=dict(source_manifest=str(self.ids),weights_sha256='a'*64)
        self.save(self.out/'SUMMARY.json',self.summary);self.save(self.root/'COMPLETE_AUDIT.json',self.cert)
        entries=[dict(path=k,sha256=guard.sha(p),bytes=p.stat().st_size) for k,p in self.targets.items()]
        entries.append(dict(path='official/weights/shufflenetv2_x1-5666bf0f80.pth',sha256='a'*64,bytes=1))
        self.save(self.backup/'MEMBERS.json',dict(members=entries))
        self.manifest=dict(scope=guard.CURRENT_SCOPE,complete_modelset=True,
            archive_sha256='b'*64,members_manifest_sha256=guard.sha(self.backup/'MEMBERS.json'),tracked_payload_files=len(entries))
        self.verified=dict(scope=guard.CURRENT_SCOPE,state='remote_archive_and_all_members_verified',
            complete_modelset_verified=True,archive_sha256='b'*64)
        self.receipt=dict(scope=guard.CURRENT_SCOPE,state='private_readback_verified',
            complete_modelset_verified=True,numeric_audit_certified=True,diagnostic_labels_scored=False,
            archive_sha256='b'*64,members_manifest_sha256=self.manifest['members_manifest_sha256'])
        self.save(self.backup/'MANIFEST.json',self.manifest)
        self.save(self.backup/'READBACK_VERIFIED.json',self.verified)
        self.save(self.backup/'DRIVE_BACKUP_RECEIPT.json',self.receipt)
    def tearDown(self):self.temp.cleanup()
    def save(self,p,x):p.write_text(__import__('json').dumps(x,indent=2)+'\n')
    def test_coherent_virtual_closed_artifacts(self):
        self.assertIs(guard.validate_closed_artifacts(self.root,self.c)['predeclared_gate_passed'],False)
    def test_pass_unknown_old_SGDM_route_and_forbidden_labels_refused(self):
        for k,v in [('predeclared_gate_passed',True),('route','close_old_SGDM40'),
                    ('users7_8_labels_used',True),('complete_private_modelset_verified',False)]:
            self.save(self.out/'SUMMARY.json',dict(self.summary,**{k:v}))
            with self.assertRaises(AssertionError):guard.validate_closed_artifacts(self.root,self.c)
    def test_partial_replay_or_absent_journal_claim_refused(self):
        for k,v in [('all100_replay_epoch_journals_committed',False),
                    ('checkpoint45_operational_continuation_disclosed',False),('pixel_count',100)]:
            self.save(self.root/'COMPLETE_AUDIT.json',dict(self.cert,**{k:v}))
            with self.assertRaises(AssertionError):guard.validate_closed_artifacts(self.root,self.c)
    def test_producer_only_or_unverified_private_receipt_refused(self):
        for k,v in [('scope','producer_complete_without_numeric_audit'),('numeric_audit_certified',False),
                    ('complete_modelset_verified',False),('diagnostic_labels_scored',True)]:
            self.save(self.backup/'DRIVE_BACKUP_RECEIPT.json',dict(self.receipt,**{k:v}))
            with self.assertRaises(AssertionError):guard.validate_closed_artifacts(self.root,self.c)
    def test_stale_model_or_audit_journal_binding_refused(self):
        for target in [self.out/'final_model.pt',self.audit/'epoch100.json']:
            original=target.read_bytes();target.write_bytes(b'changed virtual bytes')
            with self.assertRaises(AssertionError):guard.validate_closed_artifacts(self.root,self.c)
            target.write_bytes(original)
    def test_changed_members_without_manifest_rehash_refused(self):
        p=self.backup/'MEMBERS.json';p.write_text('{}\n')
        with self.assertRaises(AssertionError):guard.validate_closed_artifacts(self.root,self.c)
    def test_preparation_cannot_activate_without_real_policy(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(guard,'B',Path(directory)):
            with self.assertRaises(FileNotFoundError):guard.ready()

if __name__=='__main__':unittest.main(verbosity=2)
