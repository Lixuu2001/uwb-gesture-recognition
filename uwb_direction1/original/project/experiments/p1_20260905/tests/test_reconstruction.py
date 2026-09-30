import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reconstruction import Completion, fit_scale, reconstruction_loss

class ReconstructionTests(unittest.TestCase):
    def test_visible_slot_and_single_input(self):
        for v in range(3):
            m=Completion(8,v,'joint').eval(); x=torch.randn(5,8)
            z,parts=m(x)
            self.assertEqual(z.shape,(5,12))
            torch.testing.assert_close(parts[:,v],x)
            self.assertEqual(parts.shape,(5,3,8))
            with self.assertRaises(ValueError): m(torch.randn(5,3,8))

    def test_targets_detached(self):
        p=torch.randn(5,3,8,requires_grad=True); t=torch.randn(5,3,8,requires_grad=True)
        reconstruction_loss(p,t,1).backward()
        self.assertIsNone(t.grad)
        self.assertEqual(p.grad[:,1].abs().sum().item(),0)

    def test_scaler_train_only(self):
        x=torch.randn(10,3,8); a,b=fit_scale(x,[0,1,2]); x[3:]+=1000
        c,d=fit_scale(x,[0,1,2]); torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)

    def test_capacity_joint_identical(self):
        torch.manual_seed(2026);a=Completion(8,2,'capacity')
        torch.manual_seed(2026);b=Completion(8,2,'joint')
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k])

if __name__=='__main__':unittest.main()
