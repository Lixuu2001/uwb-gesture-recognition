"""Contracts for training-only multiview residual adapters."""
import importlib.util
from pathlib import Path

import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[1] / "models.py"
    assert path.exists(), "The multiview adapter implementation is missing"
    spec = importlib.util.spec_from_file_location("fusion_models_contract", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("learned", [False, True])
def test_missing_fusion_starts_at_anchor_and_all_masks_block_hidden_inputs(learned):
    m = module()
    assert m.MASKS == ((False, False, True), (True, False, True),
                       (False, True, True), (True, True, True))
    model = m.MissingFusion(8, learned_tokens=learned)
    x = torch.randn(4, 3, 8, requires_grad=True)
    for mask in m.MASKS:
        correction, code = model(x, mask)
        assert correction.shape == (4, 12)
        assert code.shape == (4, 128)
        assert torch.count_nonzero(correction) == 0
        changed = x.detach().clone()
        changed[:, [not present for present in mask]] = float("nan")
        other, other_code = model(changed, mask)
        torch.testing.assert_close(other, correction)
        torch.testing.assert_close(other_code, code)
        grad = torch.autograd.grad(code.square().sum(), x)[0]
        for view, present in enumerate(mask):
            assert bool(grad[:, view].abs().sum() > 0) == present


def test_missing_tokens_do_not_change_initialization_and_receive_code_gradient():
    m = module()
    torch.manual_seed(23)
    zero = m.MissingFusion(8)
    torch.manual_seed(23)
    learned = m.MissingFusion(8, learned_tokens=True)
    x = torch.randn(4, 3, 8)
    torch.testing.assert_close(zero(x, m.MASKS[0])[1], learned(x, m.MASKS[0])[1])
    learned(x, m.MASKS[0])[1].square().sum().backward()
    marker = learned.missing_tokens
    assert marker.grad[:2].abs().sum() > 0
    assert marker.grad[2].abs().sum() == 0


@pytest.mark.parametrize("kind", ["MissingFusion", "SharedSpecific"])
def test_right_only_deployment_parity_even_after_nonzero_correction(kind):
    m = module()
    model = getattr(m, kind)(8)
    torch.nn.init.normal_(model.head.weight)
    x = torch.randn(4, 3, 8)
    expected = model(x, m.MASKS[0])[0]
    torch.testing.assert_close(expected, model.deploy(x[:, 2]))
    with pytest.raises(ValueError):
        model.deploy(x)
    with pytest.raises(ValueError):
        model(torch.randn(4, 2, 8), m.MASKS[0])
    with pytest.raises(ValueError):
        model(x, (False, False, False))


def test_shared_specific_missing_invariance_and_live_shared_private_paths():
    m = module()
    model = m.SharedSpecific(8)
    x = torch.randn(4, 3, 8, requires_grad=True)
    for mask in m.MASKS:
        correction, code = model(x, mask)
        assert correction.shape == (4, 12)
        assert code.shape == (4, 128)
        assert torch.count_nonzero(correction) == 0
        changed = x.detach().clone()
        changed[:, [not v for v in mask]] = float("nan")
        torch.testing.assert_close(model(changed, mask)[1], code)
    model(x, m.MASKS[-1])[1].square().sum().backward()
    assert model.shared.weight.grad.abs().sum() > 0
    assert all(layer.weight.grad.abs().sum() > 0 for layer in model.private)
    model.zero_grad(set_to_none=True)
    aux = model.auxiliary(x)
    assert set(aux) == {"align", "view"}
    assert all(torch.isfinite(value) for value in aux.values())
    (aux["align"] + aux["view"]).backward()
    assert model.shared.weight.grad.abs().sum() > 0
    assert all(layer.weight.grad.abs().sum() > 0 for layer in model.private)
    repeated = x[:, :1].expand(-1, 3, -1)
    assert model.auxiliary(repeated)["align"].item() == pytest.approx(0)


@pytest.mark.parametrize("trainable", [False, True])
def test_prototype_completion_uses_banks_without_deployment_labels(trainable):
    m = module()
    banks = torch.randn(3, 12, 8)
    model = m.PrototypeCompletion(8, banks, trainable_bank=trainable)
    right = torch.randn(4, 8)
    correction, code = model(right)
    assert correction.shape == (4, 12)
    assert code.shape == (4, 128)
    assert torch.count_nonzero(correction) == 0
    assert isinstance(model.banks, torch.nn.Parameter) == trainable
    torch.testing.assert_close(model.deploy(right), correction)
    with torch.no_grad():
        model.banks[0].add_(3)
    assert not torch.allclose(model(right)[1], code)
    if trainable:
        model(right)[1].square().sum().backward()
        assert model.banks.grad[:2].abs().sum() > 0
    with pytest.raises(ValueError):
        model.deploy(torch.randn(4, 3, 8))
    with pytest.raises(TypeError):
        model.deploy(right, torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        m.PrototypeCompletion(8, torch.randn(2, 12, 8))


@pytest.mark.parametrize("labels", [[0], [0, 0, 0], [0, 1, 2], [0, 0, 1]])
def test_sigmoid_contrast_is_finite_with_absent_pair_types_and_large_codes(labels):
    m = module()
    codes = (torch.randn(len(labels), 8) * 1e10).requires_grad_()
    loss = m.sigmoid_contrast(codes, torch.tensor(labels))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(codes.grad).all()
    if len(labels) == 1:
        assert loss.item() == 0


def test_sigmoid_contrast_formula_excludes_identity_and_balances_pair_types():
    m = module()
    codes = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    loss = m.sigmoid_contrast(codes, labels)
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(-5.)))
    with pytest.raises(ValueError):
        m.sigmoid_contrast(codes, labels[:2])
    with pytest.raises(ValueError):
        m.sigmoid_contrast(codes, labels, temperature=0)
