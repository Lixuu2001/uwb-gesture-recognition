import unittest
import numpy as np
from finalize_supported_validation import supported_macro,main
class ReportBoundary(unittest.TestCase):
    def test_missing_two_validation_classes_finite(self):
        y=np.repeat(np.arange(10),7);p=y.copy();p[:7]=9
        self.assertAlmostEqual(supported_macro(np,y,p,range(10)),.9)
    def test_full_diagnostic_support_unchanged(self):
        y=np.repeat(np.arange(12),10);p=y.copy();p[:5]=11
        expected=np.mean([np.mean(p[y==k]==k) for k in range(12)])
        self.assertEqual(supported_macro(np,y,p,range(12)),float(expected))
    def test_unexpected_support_fails(self):
        with self.assertRaises(AssertionError):supported_macro(np,np.arange(10),np.arange(10),range(12))
    def test_missing_registration_blocks_score(self):
        with self.assertRaises(FileNotFoundError):main()
if __name__=='__main__':unittest.main()
