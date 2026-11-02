import importlib.util
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def module():
    assert (HERE/'train_e2e.py').exists(), 'missing end-to-end training loop'
    spec = importlib.util.spec_from_file_location('e2e_training', HERE/'train_e2e.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_all_subset_losses_contribute_with_unit_total_weight():
    m = module()
    torch.manual_seed(17)
    logits = [torch.randn(3, 12, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([0, 1, 2]); matrix = torch.eye(12)
    outputs = [(z, None) for z in logits]
    loss = m.objective(outputs, labels, matrix)
    reference = sum(m.classification_loss(z, labels, matrix) for z in logits)/4
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert all(z.grad.abs().sum() > 0 for z in logits)


def test_stopping_rule_respects_minimum_and_patience():
    m = module()
    assert not m.should_stop(8, 0, 12, 8)
    assert m.should_stop(12, 4, 12, 8)
    assert not m.should_stop(12, 8, 12, 8)
