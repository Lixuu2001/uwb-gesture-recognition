import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def module():
    assert (HERE/'runner.py').exists(), 'missing sequential experiment trainer'
    spec = importlib.util.spec_from_file_location('fusion_runner', HERE/'runner.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_subset_schedule_keeps_right_and_equal_forward_budget():
    m = module()
    g = torch.Generator().manual_seed(9)
    for arm in m.STAGES[1]:
        masks = m.subset_schedule(arm, g)
        assert len(masks) == 4
        assert all(mask[2] for mask in masks)
    assert len(set(m.subset_schedule('JOINT_ZERO', g))) == 4
    assert set(m.subset_schedule('R_ONLY', g)) == {(False, False, True)}


def test_derangement_preserves_class_subject_changes_every_row():
    m = module()
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([1, 1, 1, 2, 2, 2])
    permutation = m.derangement(y, s, 2026)
    assert np.all(permutation != np.arange(6))
    np.testing.assert_array_equal(y, y[permutation])
    np.testing.assert_array_equal(s, s[permutation])
    with pytest.raises(ValueError):
        m.derangement(np.arange(2), np.arange(2), 2026)


def test_classification_loss_matches_existing_projection_formula():
    m = module()
    z = torch.randn(4, 12, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3])
    u = .8*torch.eye(12)+.2/12*torch.ones(12, 12)
    p = torch.nn.functional.one_hot(y, 12).float()
    ref = torch.nn.functional.cross_entropy(z, y)+2*torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(z/2, -1), .5*(p+p@u), reduction='batchmean')
    torch.testing.assert_close(m.classification_loss(z, y, u), ref)


def test_checkpoint_epoch_zero_is_eligible():
    m = module()
    epoch, score = m.select_best([{'epoch': 0, 'val_ce': 1.0},
                                 {'epoch': 1, 'val_ce': 1.1}])
    assert epoch == 0 and score == 1.0


def test_completed_run_rejects_missing_or_corrupt_artifact(tmp_path):
    m = module()
    status = {'predictions_sha256': 'wrong', 'deployment_sha256': 'wrong'}
    with pytest.raises((ValueError, FileNotFoundError)):
        m.verify_completed(tmp_path, status)
    (tmp_path/'predictions.npz').write_bytes(b'changed')
    (tmp_path/'deployment.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):
        m.verify_completed(tmp_path, status)


def test_direct_readout_excludes_frozen_logits():
    m = module()
    base = torch.randn(3, 12)
    correction = torch.randn(3, 12)
    torch.testing.assert_close(m.combine_logits(correction, base, 'direct'), correction)
    torch.testing.assert_close(m.combine_logits(correction, base, 'residual'), base+correction)
    with pytest.raises(ValueError):
        m.combine_logits(correction, base, 'unknown')
