"""Deployment requires only its exported weights and a single Right input."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[1] / 'deploy.py'
    assert path.exists(), 'The standalone Right deployment loader is missing'
    spec = importlib.util.spec_from_file_location('e2e_deployment_contract', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def standalone_export(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'smoke128/E2E_MV_s2026_h1/deployment.pt'
    original = torch.load(source, map_location='cpu', weights_only=False)
    path = tmp_path / 'standalone.pt'
    # Historical training paths and metadata are deliberately absent.
    torch.save({'state': original['state']}, path)
    return path


def test_loading_reads_only_export_not_training_checkpoint_or_cache(standalone_export, monkeypatch):
    deploy = module()
    original_load = torch.load
    calls = []
    def guarded_load(path, *args, **kwargs):
        calls.append(Path(path).resolve())
        assert Path(path).resolve() == standalone_export.resolve()
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, 'load', guarded_load)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('Deployment cannot read training caches'))
    model = deploy.load_deployment(standalone_export, device='cpu')
    assert calls == [standalone_export.resolve()]
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert model(torch.randn(1, 1, 90, 378)).shape == (1, 12)


@pytest.mark.parametrize('shape', [(2, 3, 90, 378), (2, 90, 378), (2, 1, 90, 189), (0, 1, 90, 378)])
def test_deployment_rejects_hidden_views_and_invalid_input_shapes(standalone_export, shape):
    model = module().load_deployment(standalone_export)
    with pytest.raises(ValueError):
        model(torch.zeros(shape))


def test_deployment_stays_eval_and_never_constructs_gradient_graphs(standalone_export):
    model = module().load_deployment(standalone_export)
    x = torch.randn(1, 1, 90, 378, requires_grad=True)
    with pytest.raises(ValueError):
        model.train()
    model.encoder.train()
    first = model(x)
    second = model(x)
    assert all(not child.training for child in model.modules())
    assert not first.requires_grad
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(TypeError):
        model(x, torch.tensor([1]))


def test_deployment_load_is_strict_and_rejects_invalid_saved_normalization(standalone_export, tmp_path):
    deploy = module()
    saved = torch.load(standalone_export, weights_only=False)
    saved['state']['sd'].zero_()
    invalid = tmp_path / 'invalid.pt'
    torch.save(saved, invalid)
    with pytest.raises(ValueError):
        deploy.load_deployment(invalid)
    saved['state']['sd'].fill_(1)
    del saved['state']['encoder.temporal.pos']
    torch.save(saved, invalid)
    with pytest.raises(RuntimeError):
        deploy.load_deployment(invalid)


def test_standalone_forward_matches_training_architecture_fp32(standalone_export):
    deploy = module()
    deployment = deploy.load_deployment(standalone_export)
    path = Path(__file__).resolve().parents[1] / 'model.py'
    spec = importlib.util.spec_from_file_location('e2e_training_architecture_parity', path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    saved = torch.load(standalone_export, weights_only=False)['state']
    # Share architecture instances, without calling training build_model.
    reference = training.EndToEndFusion(deployment.encoder, deployment.fuser, saved['mu'], saved['sd']).eval()
    x = torch.randn(2, 1, 90, 378)
    with torch.inference_mode():
        expected = reference(x)
    actual = deployment(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
