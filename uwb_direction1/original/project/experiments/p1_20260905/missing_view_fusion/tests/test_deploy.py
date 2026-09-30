import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def module():
    assert (HERE/'deploy.py').exists(), 'missing raw Right deployment wrapper'
    spec = importlib.util.spec_from_file_location('fusion_deploy', HERE/'deploy.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rebuild_all_adapters_from_exported_state():
    m = module()
    from models import MissingFusion, SharedSpecific, PrototypeCompletion
    examples = [('JOINT_TOKEN', MissingFusion(8, True)), ('SS_ALIGN', SharedSpecific(8)),
                ('PROTO_MULTI', PrototypeCompletion(8, torch.randn(3, 12, 8)))]
    for arm, source in examples:
        source.head.weight.data.normal_(std=.01)
        rebuilt = m.restore_adapter({'config': {'arm': arm}, 'feature_dim': 8,
                                     'state': source.state_dict()})
        x = torch.randn(4, 8)
        torch.testing.assert_close(rebuilt.deploy(x), source.deploy(x))


def test_raw_deployment_rejects_multiple_views_before_encoder_access():
    m = module()
    wrapper = m.RadarDeployment(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(),
                                torch.zeros(1, 1664), torch.ones(1, 1664))
    with pytest.raises(ValueError, match='Right'):
        wrapper(torch.zeros(2, 3, 90, 378))
