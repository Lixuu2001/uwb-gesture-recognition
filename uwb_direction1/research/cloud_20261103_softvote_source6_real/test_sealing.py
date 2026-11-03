"""Guard regressions: incomplete/private/fusion failures cannot reach labels."""
import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import common,controller,finalize_once
from stream_archive import build,verify
class Sealing(unittest.TestCase):
    def test_all_five_and_private_recovery_before_any_diagnostic_label(self):
        with patch.object(finalize_once,'all_frozen',side_effect=FileNotFoundError('five folds not frozen')),patch.object(np,'load',side_effect=AssertionError('label access forbidden')) as loader:
            with self.assertRaises(FileNotFoundError):finalize_once.main()
            loader.assert_not_called()
    def test_public_or_unmaterialized_chunks_rejected(self):
        row=dict(file_id='synthetic',bytes=3,raw_fetch=True,include_base64=False,download_verified=True,metadata=dict(shared=False,parent_ids=['12AGMwT99yXCTYBQocQaX901Br-IelmOf'],permissions=[dict(role='owner',type='user')],size_bytes=3))
        common.privacy([row]);row['metadata']['shared']=True
        with self.assertRaises(AssertionError):common.privacy([row])
        row['metadata']['shared']=False;row['download_verified']=False
        with self.assertRaises(AssertionError):common.privacy([row])
    def test_frozen_fusion_requires_successful_actual_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as d,patch.object(controller,'B',Path(d)):
            root=Path(d)/'folds/held2';f=root/'fusion';f.mkdir(parents=True);(f/'predictions').write_bytes(b'synthetic')
            common.write(f/'FOLD_FROZEN.json',dict(state='three_arms_frozen_before_any_new_held_truth',diagnostic_labels_read=False,files={'predictions':common.sha(f/'predictions')}))
            self.assertFalse(controller.valid_fold(2))
            t=dict(exit_code=0,resource_stop_reason=None,wall_timeout=False,cgroup_OOM_kill_delta=0)
            common.write(root/'FUSION_TERMINATION.json',t);self.assertTrue(controller.valid_fold(2))
            t['exit_code']=-9;common.write(root/'FUSION_TERMINATION.json',t);self.assertFalse(controller.valid_fold(2))
    def test_chunk_and_member_stream_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);(b/'fixture').write_bytes(b'synthetic model bytes')
            m=build(b/'archive','fixture',{'model.pt':b/'fixture'},'synthetic',chunk_size=100,limit=10000)
            rows=[dict(**r,path=str(b/'archive'/r['name'])) for r in m['chunks']]
            self.assertTrue(verify(m,rows)['complete_modelset_verified'])
            p=Path(rows[0]['path']);raw=p.read_bytes();p.write_bytes(bytes([raw[0]^1])+raw[1:])
            with self.assertRaises((AssertionError,OSError)):verify(m,rows)
if __name__=='__main__':unittest.main(verbosity=2)
