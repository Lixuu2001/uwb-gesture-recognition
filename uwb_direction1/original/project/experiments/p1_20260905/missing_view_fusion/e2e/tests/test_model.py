"""End-to-end gradient, isolation, initialization, and deployment contracts."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


def module():
    path = Path(__file__).resolve().parents[1] / 'model.py'
    assert path.exists(), 'The trainable fusion model has not been implemented'
    spec = importlib.util.spec_from_file_location('e2e_model_contract', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class TinyEncoder(nn.Module):
    """Real differentiable view-independent tokens with the production shape."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(.3, 1.3, 32))
        self.calls = 0
        self.last_shape = None

    def forward(self, x):
        self.calls += 1
        self.last_shape = tuple(x.shape)
        temporal = x.mean(-1).unsqueeze(-1) * self.scale
        spatial = x.mean(-2).unsqueeze(-1) * self.scale
        return None, temporal, spatial


def make_model(m, seed=2026):
    torch.manual_seed(seed)
    encoder = TinyEncoder().requires_grad_(False)
    fuser = m.MissingFusion(1664, learned_tokens=True)
    mu = torch.stack([torch.full((1664,), v / 5) for v in range(3)])[None]
    sd = torch.stack([torch.full((1664,), 1 + v / 4) for v in range(3)])[None]
    return m.EndToEndFusion(encoder, fuser, mu, sd)


def test_single_encoder_call_and_matched_r_mv_compute_shapes():
    m = module()
    model = make_model(m)
    right = torch.randn(2, 1, 90, 378)
    outputs = model.training_outputs(right, 'R')
    assert len(outputs) == 4
    assert model.encoder.calls == 1
    assert model.encoder.last_shape == (2, 3, 90, 378)
    for logits, code in outputs:
        assert logits.shape == (2, 12) and code.shape == (2, 128)
        assert torch.count_nonzero(logits) == 0
        torch.testing.assert_close(code, outputs[0][1])
    model.training_outputs(torch.randn(2, 3, 90, 378), 'MV')
    assert model.encoder.calls == 2
    assert model.encoder.last_shape == (2, 3, 90, 378)


def test_full_multiview_classification_backpropagates_into_hidden_inputs_and_encoder():
    m = module()
    model = make_model(m)
    assert all(p.requires_grad for p in model.encoder.parameters())
    assert 'mu' in dict(model.named_buffers()) and 'sd' in dict(model.named_buffers())
    assert not model.mu.requires_grad and not model.sd.requires_grad
    nn.init.normal_(model.fuser.head.weight, std=.05)
    x = torch.randn(2, 3, 90, 378, requires_grad=True)
    before = model.encoder.scale.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    logits, _ = model.training_outputs(x, 'MV')[-1]
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.encoder.scale.grad.abs().sum() > 0
    assert x.grad[:, 0].abs().sum() > 0 and x.grad[:, 1].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.encoder.scale)


def test_eval_right_subset_matches_deployment_and_ignores_hidden_values():
    m = module()
    model = make_model(m).eval()
    nn.init.normal_(model.fuser.head.weight, std=.05)
    x = torch.randn(2, 3, 90, 378, requires_grad=True)
    right_only = model(x[:, 2:3])
    r_training = model.training_outputs(x[:, 2:3], 'R')[0][0]
    mv_r_subset = model.training_outputs(x, 'MV')[0][0]
    torch.testing.assert_close(right_only, r_training)
    torch.testing.assert_close(right_only, mv_r_subset)
    grad = torch.autograd.grad(mv_r_subset.sum(), x)[0]
    assert torch.count_nonzero(grad[:, :2]) == 0
    changed = x.detach().clone()
    changed[:, :2] = float('nan')
    torch.testing.assert_close(model.training_outputs(changed, 'MV')[0][0], right_only)


@pytest.mark.parametrize('operation', ['forward', 'R', 'MV', 'unknown'])
def test_input_contract_rejects_hidden_deployment_inputs_and_wrong_modes(operation):
    m = module()
    model = make_model(m)
    with pytest.raises(ValueError):
        if operation == 'forward':
            model(torch.randn(2, 3, 90, 378))
        elif operation == 'R':
            model.training_outputs(torch.randn(2, 3, 90, 378), 'R')
        elif operation == 'MV':
            model.training_outputs(torch.randn(2, 1, 90, 378), 'MV')
        else:
            model.training_outputs(torch.randn(2, 1, 90, 378), 'unknown')


def test_stats_are_copied_and_invalid_normalization_is_rejected():
    m = module()
    mu, sd = torch.zeros(1, 3, 1664), torch.ones(1, 3, 1664)
    model = m.EndToEndFusion(TinyEncoder(), m.MissingFusion(1664), mu, sd)
    mu.fill_(10)
    assert torch.count_nonzero(model.mu) == 0
    with pytest.raises(ValueError):
        m.EndToEndFusion(TinyEncoder(), m.MissingFusion(1664), mu[:, :1], sd)
    sd.zero_()
    with pytest.raises(ValueError):
        m.EndToEndFusion(TinyEncoder(), m.MissingFusion(1664), mu, sd)


def test_real_checkpoint_build_reproducibly_initializes_trainable_encoder_and_new_fuser():
    m = module()
    mu, sd = torch.zeros(1, 3, 1664), torch.ones(1, 3, 1664)
    first, a = m.build_model(1, 2026, mu, sd, device='cpu')
    second, b = m.build_model(1, 2026, mu, sd, device='cpu')
    assert a == b
    assert a['initial_encoder_sha256'] == m.state_hash(first.encoder)
    assert a['initial_fuser_sha256'] == m.state_hash(first.fuser)
    assert a['checkpoint_sha256'] and a['source_seed'] == 2026
    assert m.state_hash(first) == m.state_hash(second)
    assert all(p.requires_grad for p in first.encoder.parameters())
    assert not hasattr(first, 'head')
    first.eval()
    # A real backbone forward verifies the adapter/token integration on CPU.
    assert first(torch.randn(1, 1, 90, 378)).shape == (1, 12)


@pytest.mark.parametrize('field,value', [('fold', 2), ('views', [0]), ('seed', 2027), ('mode', 'off')])
def test_checkpoint_lineage_rejects_wrong_fold_views_source_seed_and_mode(field, value):
    m = module()
    checkpoint = {'fold': 1, 'args': {'views': [2], 'seed': 2026, 'mode': 'concat',
                  'readout': 'mean', 'tag': 'sv_R_label_global', 'teacher': '', 'aux': 'none',
                  'soft_target': 'training_onehot_global_projection'}}
    if field == 'fold':
        checkpoint['fold'] = value
    else:
        checkpoint['args'][field] = value
    with pytest.raises(ValueError):
        m.validate_checkpoint(checkpoint, 1)
