import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gated_reconstruction import GatedCompletion

class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.ones(32))
    def forward(self,x):
        t=x.mean(-1).unsqueeze(-1)*self.scale
        s=x.mean(-2).unsqueeze(-1)*self.scale
        return None,t,s

class GatedTests(unittest.TestCase):
    def test_visible_only_and_gate_range(self):
        m=GatedCompletion(DummyEncoder(),0,'joint',torch.zeros(1,3,992),torch.ones(1,3,992))
        z,p,g=m(torch.randn(4,1,90,189))
        self.assertEqual(z.shape,(4,12));self.assertEqual(p.shape,(4,3,992));self.assertEqual(g.shape,(4,2,1))
        self.assertTrue(torch.all((g>=0)&(g<=1)))
        with self.assertRaises(ValueError):m(torch.randn(4,3,90,189))

    def test_gate_is_differentiable(self):
        m=GatedCompletion(DummyEncoder(),0,'joint',torch.zeros(1,3,992),torch.ones(1,3,992))
        _,p,g=m(torch.randn(3,1,90,189));(p[:,1:].square().mean()+g.square().mean()).backward()
        self.assertGreater(m.encoder.scale.grad.abs().sum().item(),0)
        self.assertGreater(m.gate[0].weight.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
