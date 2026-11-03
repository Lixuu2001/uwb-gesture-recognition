"""Controller orchestration with virtual ready()/children and real tiny archive."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller,trial_common as common
from stream_archive import build
from backup import FOLDER

class ControllerBoundary(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.b=Path(self.temp.name)
        (self.b/'FROZEN_HASHES.json').write_text('{}\n')
        source=self.b/'virtual_seed';source.write_bytes(b'synthetic')
        m=build(self.b/'initial_insurance','initial',{'seed/test':source},
                'initial_seed_only_ShuffleNetV2_x1_Adam100_B20_20261103')
        m['complete_modelset']=False;common.write(self.b/'initial_insurance/MANIFEST.json',m)
        common.write(self.b/'initial_insurance/READBACK_INPUT.json',[
            dict(c,path=str(self.b/'initial_insurance'/c['name']),file_id='virtual_'+str(i),
            raw_fetch=True,include_base64=False,download_verified=True,
            metadata=dict(shared=False,parent_ids=[FOLDER],size_bytes=c['bytes'],
            permissions=[dict(role='owner',type='user')])) for i,c in enumerate(m['chunks'])])
        common.write(self.b/'INITIAL_PRIVATE_INSURANCE.json',dict(
            state='owner_only_raw_allmember_initial_insurance_verified',
            frozen_sha256=common.sha(self.b/'FROZEN_HASHES.json'),
            archive_sha256=m['archive_sha256'],diagnostic_labels_scored=False,shared=False))
        self.policy=dict(total_wall_seconds=30,total_CPU_seconds=60,RSS_stop_KiB=14680064,
            producer_limits=dict(wall=10,soft_cpu=10,hard_cpu=20),
            audit_limits=dict(wall=10,soft_cpu=10,hard_cpu=20))
        self.patches=[patch.object(controller,'B',self.b),
                      patch.object(controller,'ready',return_value=self.policy)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def term(self,**changes):
        return dict(exit_code=0,wall_timeout=False,resource_stop_reason=None,
                    cgroup_oom_kill_delta=0,**changes)
    def test_coherent_virtual_children_finish_unscored(self):
        with patch.object(controller,'bounded_run',return_value=self.term()) as run:
            controller.main();self.assertEqual(run.call_count,2)
            self.assertEqual([Path(x.args[0][-1]).name for x in run.call_args_list],
                             ['produce.py','audit_full.py'])
        self.assertEqual(common.read(self.b/'STATE.json')['state'],
            'complete_audited_unscored_requires_private_streaming_backup_then_once_finalize')
        self.assertFalse((self.b/'FINALIZE_ONCE.lock').exists())
    def test_existing_producer_refused_before_child(self):
        (self.b/'run01').mkdir()
        with patch.object(controller,'bounded_run') as run:
            with self.assertRaises(AssertionError):controller.main()
            run.assert_not_called()
    def test_failed_producer_never_launches_auditor(self):
        term=self.term();term['resource_stop_reason']='RSS_resource_limit'
        with patch.object(controller,'bounded_run',return_value=term) as run:
            controller.main();self.assertEqual(run.call_count,1)
        state=common.read(self.b/'STATE.json')
        self.assertEqual(state['state'],'execution_incomplete_producer')
        self.assertFalse(state['scientific_negative']);self.assertFalse(state['diagnostic_scored'])
    def test_launch_exception_is_terminal_evidence(self):
        def launch(*a):
            out=a[1];out.mkdir()
            common.write(out/'TERMINATION.json',dict(pid=None,exit_code=None,
                wall_timeout=False,resource_stop_reason='child_launch_failed'))
            raise OSError('virtual launch')
        with patch.object(controller,'bounded_run',side_effect=launch) as run:
            controller.main();self.assertEqual(run.call_count,1)
        self.assertEqual(common.read(self.b/'STATE.json')['state'],'execution_incomplete_producer')

if __name__=='__main__':unittest.main(verbosity=2)
