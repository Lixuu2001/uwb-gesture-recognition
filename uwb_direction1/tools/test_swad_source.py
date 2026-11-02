"""Differential official-code checks and interrupted-training replay for SWAD.

Official snapshots are local and licensed; only CUDA returns/logger dependency
are adapted in the independent reference loader. No external code fetched here.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from swad_core import AveragedModel, LossValley, pack_average, unpack_average, pack_valley, unpack_valley


def official_classes():
    root=Path(__file__).resolve().parents[1]/'research/sources/swad_official'
    a=types.ModuleType('reference_average')
    exec(compile((root/'swa_utils.py').read_text(),str(root/'swa_utils.py'),'exec'),a.__dict__)
    source=(root/'swad.py').read_text().replace('from domainbed.lib import swa_utils','')
    source=source.replace('.cuda()', '.cpu()').replace('self.evaluator.logger.error(', 'print(')
    v=types.ModuleType('reference_valley');v.swa_utils=a
    exec(compile(source,str(root/'swad.py'),'exec'),v.__dict__)
    return a.AveragedModel,v.LossValley


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Sequential(nn.Flatten(),nn.Linear(12,12),nn.Dropout(.2))
    def forward(self,x): return self.head(x.mean((-2,-1)))


class SwadChecks(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_official_windows_and_serialized_queues(self):
        OA,OV=official_classes();rng=np.random.default_rng(2026)
        curves=[list(np.linspace(3,.1,18)),[1.,1.,1.]+[2.]*15,[1.,.5,.3,.4,.5,.6,.9,1.,1.1,1.2,1.3,1.4],
                [1.]*18]+[rng.uniform(.1,2.,18).tolist() for _ in range(8)]
        cases=0
        with contextlib.redirect_stdout(io.StringIO()):
            for nc,nt in [(3,6),(6,3),(3,3)]:
                for curve in curves:
                    m=nn.Linear(3,2);v=LossValley(None,nc,nt,.3);ref=OV(None,nc,nt,.3)
                    for i,loss in enumerate(curve):
                        a=AveragedModel(m);b=OA(m)
                        for j in range(1+(i%4)):
                            with torch.no_grad():
                                for p in m.parameters():p.copy_(torch.from_numpy(rng.normal(size=p.shape)).to(p))
                            a.update_parameters(m,step=i*4+j);b.update_parameters(m,step=i*4+j)
                        v.update_and_evaluate(a,None,loss,None);ref.update_and_evaluate(b,None,loss,None)
                        # Exercise serialize/restore after every checkpoint including dead valleys.
                        buf=io.BytesIO();torch.save(pack_valley(v),buf);buf.seek(0)
                        v=unpack_valley(torch.load(buf,weights_only=False),m)
                        self.assertEqual(v.dead_valley,ref.dead_valley);self.assertEqual(v.threshold,ref.threshold)
                    got=v.get_final_model();want=ref.get_final_model()
                    for k,p in want.state_dict().items():self.assertTrue(torch.equal(got.state_dict()[k],p))
                    self.assertEqual((got.start_step,got.end_step),(want.start_step,want.end_step));cases+=1
        self.assertEqual(cases,36)

    def test_optimizer_dropout_average_resume(self):
        import swad_source_pilot as pilot
        c=json.loads((Path(__file__).resolve().parents[1]/'research/SWAD_SOURCE_PROTOCOL.json').read_text())
        c.update(epochs=4,tail_start_epoch=3,batch=8)
        users=np.repeat(np.arange(3,9),4);y=np.tile(np.arange(4),6)
        cache=np.random.default_rng(8).normal(size=(24,3,6,2,2)).astype(np.float16)
        identity={'test':'synthetic resume, not quality evidence'}
        with tempfile.TemporaryDirectory() as d, patch.object(pilot,'SignalNet',Tiny),contextlib.redirect_stdout(io.StringIO()):
            full=Path(d)/'full';resumed=Path(d)/'resumed'
            pilot.fit(full,c,identity,cache,y,users,3)
            pilot.fit(resumed,c,identity,cache,y,users,3,stop_after=2)
            pilot.fit(resumed,c,identity,cache,y,users,3)
            x=torch.load(full/'selected.pt',weights_only=False);z=torch.load(resumed/'selected.pt',weights_only=False)
            self.assertEqual(x['selection'],z['selection'])
            for arm in x['states']:
                for k,v in x['states'][arm].items():self.assertTrue(torch.equal(v,z['states'][arm][k]))
            a=np.load(full/'predictions.npz');b=np.load(resumed/'predictions.npz')
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])

    def test_diagnostic_labels_do_not_select_models(self):
        import swad_source_pilot as pilot
        c=json.loads((Path(__file__).resolve().parents[1]/'research/SWAD_SOURCE_PROTOCOL.json').read_text());c.update(epochs=2,tail_start_epoch=1,batch=8)
        users=np.repeat(np.arange(1,9),4);y=np.tile(np.arange(4),8);changed=y.copy()
        changed[users==3]=(changed[users==3]+7)%12
        changed[users<=2]=11
        cache=np.random.default_rng(9).normal(size=(32,3,6,2,2)).astype(np.float16)
        # Refuse any read of excluded users' signals.
        class Guard:
            def __getitem__(self,ids):
                assert np.all(users[ids]>=3)
                return cache[ids]
        with tempfile.TemporaryDirectory() as d,patch.object(pilot,'SignalNet',Tiny),contextlib.redirect_stdout(io.StringIO()):
            for name,labels in [('original',y),('changed',changed)]:pilot.fit(Path(d)/name,c,{},Guard(),labels,users,3)
            a=torch.load(Path(d)/'original/selected.pt',weights_only=False);b=torch.load(Path(d)/'changed/selected.pt',weights_only=False)
            self.assertEqual(a['selection'],b['selection'])
            for arm in a['states']:
                for k,v in a['states'][arm].items():self.assertTrue(torch.equal(v,b['states'][arm][k]))

    def test_real_architecture_complete_audit_path(self):
        import swad_source_pilot as pilot
        c=json.loads((Path(__file__).resolve().parents[1]/'research/SWAD_SOURCE_PROTOCOL.json').read_text())
        c.update(epochs=2,tail_start_epoch=1,batch=8)
        users=np.repeat(np.arange(3,9),4);y=np.tile(np.arange(4),6)
        cache=np.random.default_rng(10).normal(size=(24,3,6,90,63)).astype(np.float16)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(d)
            pilot.fit(folder,c,{'test':'synthetic architecture, not accuracy evidence'},cache,y,users,3)
            evidence=pilot.audit_run(folder,cache,y,users,c)
            self.assertTrue(evidence['functional_logits_bitwise_equal'])
            self.assertTrue(evidence['official_valley_replay_bitwise_equal'])


if __name__=='__main__':unittest.main(verbosity=2)
