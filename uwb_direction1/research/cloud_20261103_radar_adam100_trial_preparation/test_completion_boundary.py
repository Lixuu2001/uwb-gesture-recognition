"""Virtual evidence fixtures only. They never activate, score or fit real UWB."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import trial_common as common
import finalize_once as finalizer
from stream_archive import build,verify
from backup import SCOPE,FOLDER,privacy

FLAGS=['full100_epoch23500_update_replay_exact','all100_validation_outputs_exact',
       'fixed_final100_endpoint_exact',
       'full_model_BN_Adam_two_moments_steps_order_and_Torch_RNG_exact',
       'all2200_final_scores_predictions_exact','all6900_raw_RMS_cache_FP64_verified']

class Boundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.b=Path(self.temp.name)
        self.patch=patch.object(common,'B',self.b);self.patch.start()
        self.ready=patch.object(common,'ready',return_value={});self.ready.start()
        for name in ['supervisor01','audit_supervisor01']:
            (self.b/name).mkdir()
            common.write(self.b/name/'TERMINATION.json',dict(exit_code=0,
                wall_timeout=False,resource_stop_reason=None,cgroup_oom_kill_delta=0,cgroup_events_available=True))
        names=['initial12_head.pt','final_model.pt','final_full.pt','split.npz',
            'steps.jsonl','TRAIN_ENDPOINT.json','FILES.json','IDENTITY.json',
            'STATE.json','validation_unscored.npz','diagnostic_unscored.npz']
        names+=['validation_epoch%03d.npz'%i for i in range(1,101)]
        names+=['epoch%03d.json'%i for i in range(1,101)]
        (self.b/'run01').mkdir()
        hashes={}
        for n in names:
            p=self.b/'run01'/n;p.write_bytes(('virtual fixture '+n).encode())
            hashes[str(p)]=common.sha(p)
        self.cert=dict(state='pass',diagnostic_labels_read=False,
            users7_8_labels_used=False,pixel_count=1038643200,
            artifact_hashes=hashes,**{k:True for k in FLAGS})
        self.save()
    def tearDown(self):
        self.ready.stop();self.patch.stop();self.temp.cleanup()
    def save(self):common.write(self.b/'COMPLETE_AUDIT.json',self.cert)
    def test_coherent_virtual_full100_completion(self):
        self.assertEqual(common.complete(),{})
    def test_each_full_replay_claim_required(self):
        for k in FLAGS:
            with self.subTest(key=k):
                self.cert[k]=False;self.save()
                with self.assertRaises(AssertionError):common.complete()
                self.cert[k]=True
    def test_truncated_artifacts_or_absent_epoch100_cannot_unlock(self):
        original=copy.deepcopy(self.cert)
        for key in ['all','validation_epoch100.npz','epoch100.json']:
            self.cert=copy.deepcopy(original)
            if key=='all':self.cert['artifact_hashes']={}
            else:self.cert['artifact_hashes'].pop(str(self.b/'run01'/key))
            self.save()
            with self.assertRaises(AssertionError):common.complete()
    def test_mutated_file_and_escape_path_refused(self):
        (self.b/'run01/final_full.pt').write_bytes(b'changed')
        with self.assertRaises(AssertionError):common.complete()
        self.cert['artifact_hashes'][str(self.b/'run01/final_full.pt')]=common.sha(self.b/'run01/final_full.pt')
        self.cert['artifact_hashes']['/etc/hosts']='0'*64;self.save()
        with self.assertRaises(AssertionError):common.complete()
    def test_partial_pixels_or_label_use_refused(self):
        for key,value in [('pixel_count',100),('diagnostic_labels_read',True),
                          ('users7_8_labels_used',True)]:
            old=self.cert[key];self.cert[key]=value;self.save()
            with self.assertRaises(AssertionError):common.complete()
            self.cert[key]=old
    def test_each_resource_or_exit_failure_refused(self):
        p=self.b/'audit_supervisor01/TERMINATION.json';old=common.read(p)
        for key,value in [('exit_code',1),('exit_code',False),('wall_timeout',True),
                          ('resource_stop_reason','RSS_resource_limit'),
                          ('cgroup_oom_kill_delta',1)]:
            common.write(p,dict(old,**{key:value}))
            with self.assertRaises(AssertionError):common.complete()

    def private_fixture(self):
        files={'run01/'+n:self.b/'run01'/n for n in ['initial12_head.pt',
            'final_model.pt','final_full.pt','split.npz','steps.jsonl',
            'validation_unscored.npz','diagnostic_unscored.npz']}
        files['study/COMPLETE_AUDIT.json']=self.b/'COMPLETE_AUDIT.json'
        ids=self.b/'virtual_ids';ids.write_bytes(b'virtual ids')
        weights=self.b/'virtual_weights';weights.write_bytes(b'virtual initial encoder')
        files['dependencies/source_ids.npz']=ids
        files['official/weights/shufflenetv2_x1-5666bf0f80.pth']=weights
        directory=self.b/'complete_backup';m=build(directory,'test',files,SCOPE,chunk_size=1024)
        rows=[dict(c,path=str(directory/c['name']),file_id='virtual_'+str(i),
                   raw_fetch=True,include_base64=False,download_verified=True,
                   metadata=dict(shared=False,parent_ids=[FOLDER],
                   permissions=[dict(role='owner',type='user')],size_bytes=c['bytes']))
              for i,c in enumerate(m['chunks'])]
        common.write(directory/'READBACK_INPUT.json',rows)
        common.write(directory/'READBACK_VERIFIED.json',verify(m,rows))
        common.write(directory/'DRIVE_BACKUP_RECEIPT.json',dict(
            state='private_readback_verified',scope=SCOPE,complete_modelset_verified=True,
            numeric_audit_certified=True,diagnostic_labels_scored=False,
            archive_sha256=m['archive_sha256'],members_manifest_sha256=m['members_manifest_sha256']))
        return dict(source_manifest=str(ids),weights_sha256=common.sha(weights)),rows
    def test_virtual_private_allmember_stream_passes(self):
        c,rows=self.private_fixture()
        with patch.object(finalizer,'B',self.b):finalizer.require_private(c)
    def test_producer_only_receipt_and_wrong_scope_refused(self):
        c,rows=self.private_fixture();p=self.b/'complete_backup/DRIVE_BACKUP_RECEIPT.json';old=common.read(p)
        for key,val in [('scope','producer_only'),('complete_modelset_verified',False),
                        ('numeric_audit_certified',False),('diagnostic_labels_scored',True)]:
            common.write(p,dict(old,**{key:val}))
            with patch.object(finalizer,'B',self.b),self.assertRaises(AssertionError):
                finalizer.require_private(c)
    def test_private_acl_and_raw_flags_refused(self):
        c,rows=self.private_fixture()
        for key,val in [('raw_fetch',False),('include_base64',True),('download_verified',False)]:
            altered=copy.deepcopy(rows);altered[0][key]=val
            with self.assertRaises(AssertionError):privacy(altered)
        altered=copy.deepcopy(rows);altered[0]['metadata']['shared']=True
        with self.assertRaises(AssertionError):privacy(altered)
    def test_stale_model_cannot_match_other_valid_archive(self):
        c,rows=self.private_fixture();(self.b/'run01/final_model.pt').write_bytes(b'other')
        with patch.object(finalizer,'B',self.b),self.assertRaises(AssertionError):
            finalizer.require_private(c)
    def test_once_lock_blocks_before_any_label_loader(self):
        (self.b/'FINALIZE_ONCE.lock').touch()
        with patch.object(finalizer,'B',self.b),patch.object(finalizer,'complete',return_value={}),\
             patch.object(finalizer,'require_private'),patch.object(finalizer,'load_inputs') as inputs:
            with self.assertRaises(FileExistsError):finalizer.main()
            inputs.assert_not_called()
    def test_failed_completion_never_reaches_private_or_labels(self):
        with patch.object(finalizer,'complete',side_effect=AssertionError('virtual partial')),\
             patch.object(finalizer,'require_private') as private,\
             patch.object(finalizer,'load_inputs') as inputs:
            with self.assertRaises(AssertionError):finalizer.main()
            private.assert_not_called();inputs.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
