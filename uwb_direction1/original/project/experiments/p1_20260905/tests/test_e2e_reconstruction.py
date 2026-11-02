import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e2e_reconstruction import EndToEnd, compact_grad
from reconstruction import reconstruction_loss

class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.scale=nn.Parameter(torch.ones(32))
    def forward(self,x):
        t=x.mean(-1).unsqueeze(-1)*self.scale
        s=x.mean(-2).unsqueeze(-1)*self.scale
        return None,t,s

class E2ETests(unittest.TestCase):
    def model(self):return EndToEnd(DummyEncoder(),0,'joint',torch.zeros(1,3,992),torch.ones(1,3,992))
    def test_reconstruction_updates_encoder(self):
        m=self.model();x=torch.randn(2,1,90,189)
        z,parts=m(x);target=torch.randn(2,3,992,requires_grad=True)
        reconstruction_loss(parts,target,0).backward()
        self.assertGreater(m.encoder.scale.grad.abs().sum().item(),0)
        self.assertIsNone(target.grad)
    def test_deployment_rejects_missing_real_views(self):
        with self.assertRaises(ValueError):self.model()(torch.randn(2,3,90,189))
    def test_grad_pool_matches_definition(self):
        enc=DummyEncoder();x=torch.randn(2,1,90,189)
        z=compact_grad(enc,x);_,t,s=enc(x)
        ref=torch.cat([t.reshape(2,1,10,9,32).mean(3).flatten(2),s.reshape(2,1,21,9,32).mean(3).flatten(2)],-1)
        torch.testing.assert_close(z,ref);self.assertTrue(z.requires_grad)

if __name__=='__main__':unittest.main()
