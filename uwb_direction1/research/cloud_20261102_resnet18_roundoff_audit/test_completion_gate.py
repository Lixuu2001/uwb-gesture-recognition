from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import completion_common as c
class Test(unittest.TestCase):
 def test_missing_correction_remote_blocks(self):
  with tempfile.TemporaryDirectory() as name,patch.object(c,'O',Path(name)),patch.object(c,'original_ready',return_value={}):
   with self.assertRaises(FileNotFoundError):c.ready()
 def test_original_failed_and_corrected_audits_and_shared_encoder_required(self):
  m=dict(scope=c.SCOPE,complete_modelset=True,shared_frozen_encoder_included_once=True,original_failed_audit_included=True,corrected_complete_audit_included=True,archive_sha256='same');v=dict(scope=c.SCOPE,complete_modelset_verified=True,archive_sha256='same');r=dict(scope=c.SCOPE,complete_modelset_verified=True,numeric_audit_certified=True,state='private_readback_verified',archive_sha256='same');b={'source_preparation/weights/resnet18-f37072fd.pth':'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'};c.private_scope(m,v,r,b)
  with self.assertRaises(AssertionError):c.private_scope(dict(m,original_failed_audit_included=False),v,r,b)
  with self.assertRaises((KeyError,AssertionError)):c.private_scope(m,v,r,{})
if __name__=='__main__':unittest.main(verbosity=2)
