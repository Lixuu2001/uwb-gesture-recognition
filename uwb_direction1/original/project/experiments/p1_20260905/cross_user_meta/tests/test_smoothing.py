import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('_cross_smoothing_tests', HERE/'smoothing.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_smoothing_fit_never_uses_excluded_users_and_matches_weighted_oof_mass():
    m = module(); rng = np.random.default_rng(3)
    subjects = np.repeat(np.arange(1,9), 20)
    labels = np.tile(np.repeat([0,1],10),8)
    x = rng.normal(size=(len(labels),3)); x[:,0] += labels*2
    ids = np.where(subjects > 2)[0]
    matrix, oof, history = m.fit_matrix(x, labels, subjects, ids, n_classes=2)
    assert np.isnan(oof[subjects <= 2]).all()
    assert len(history) == 6
    for row in history:
        assert not set(subjects[row['fit_ids']]) & set(subjects[row['predict_ids']])
        assert set(np.r_[row['fit_ids'],row['predict_ids']]) == set(ids)
    mass = np.mean(oof[ids,labels[ids]])
    np.testing.assert_allclose(np.diag(matrix), mass)
    np.testing.assert_allclose(matrix.sum(1),1)
    changed_x = x.copy(); changed_x[subjects<=2] = np.nan
    changed_labels = labels.copy(); changed_labels[subjects<=2] = 99
    other, _, _ = m.fit_matrix(changed_x, changed_labels, subjects, ids, n_classes=2)
    np.testing.assert_array_equal(other,matrix)


def test_criterion_matches_original_soft_target_definition():
    import torch
    from torch.nn import functional as F
    m = module(); torch.manual_seed(2)
    logits = torch.randn(4,3,requires_grad=True); labels = torch.tensor([0,1,2,0])
    matrix = torch.eye(3)*.7 + torch.ones(3,3)*.1
    q = .5*(F.one_hot(labels,3).float()+F.one_hot(labels,3).float()@matrix)
    expected = F.cross_entropy(logits,labels)+.5*F.kl_div(F.log_softmax(logits/2,-1),q,reduction='batchmean')*4
    actual = m.classification_loss(logits,labels,matrix)
    torch.testing.assert_close(actual,expected)
    actual.backward(); assert torch.isfinite(logits.grad).all()
