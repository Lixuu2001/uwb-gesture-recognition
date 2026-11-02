"""Attribute-prior plumbing (preregistration 2026-07-26, Gate 1).

Checks the parts that can silently produce a wrong experiment:
  * the case mapping picks parents that are BASE classes of that case, and drops
    targets whose parents are not available
  * the parallelogram arithmetic is exact in feature space
  * the synthetic sample cloud has spread (a repeated mean point would collapse
    the few-shot inspector)
  * attr_prior="off" changes nothing
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from config import ATTR_PARENTS, REG_TEST_CASES                      # noqa: E402
from uwb_osr.registration import (_map_attr_spec, synth_attr_prototype,  # noqa: E402
                                  synth_attr_samples)


def label_map_of(case):
    known = list(case["base"]) + list(case["register"])
    return {g: i for i, g in enumerate(known)}


def test_gate1_mapping():
    """REG case 1: G7 (id 6) is registered, parents G5/G6/G8 are all base."""
    case = REG_TEST_CASES[1]
    lm = label_map_of(case)
    spec = _map_attr_spec(ATTR_PARENTS, lm, n_base=len(case["base"]))
    assert set(spec) == {lm[6]}, f"only G7 should be usable, got {spec}"
    plus, minus = spec[lm[6]]
    assert plus == [lm[4], lm[7]] and minus == [lm[5]]
    assert all(i < len(case["base"]) for i in plus + minus)


def test_gate2_mapping_uses_single_axis_parents():
    """REG case 6: all three diagonals resolve, via the single-axis candidate."""
    case = REG_TEST_CASES[6]
    lm = label_map_of(case)
    spec = _map_attr_spec(ATTR_PARENTS, lm, n_base=len(case["base"]))
    assert set(spec) == {lm[4], lm[5], lm[6]}, spec
    plus, minus = spec[lm[6]]
    assert plus == [lm[1], lm[2]]                      # RL + UD, not the diagonals
    assert minus == [lm[0], lm[1], lm[2], lm[3]]


def test_case0_has_no_usable_target():
    """REG case 0 registers G1/G3/G4 -- single-axis classes, no parents."""
    case = REG_TEST_CASES[0]
    spec = _map_attr_spec(ATTR_PARENTS, label_map_of(case), n_base=len(case["base"]))
    assert spec == {}


def test_parallelogram_arithmetic():
    torch.manual_seed(0)
    V, D = 3, 8
    Ft = torch.randn(120, V, D)
    yt = np.repeat(np.arange(6), 20)
    phi = synth_attr_prototype(Ft, yt, [1, 3], [2], V)
    assert phi.shape == (V, D)
    for v in range(V):
        want = (Ft[yt == 1, v].mean(0) + Ft[yt == 3, v].mean(0)
                - Ft[yt == 2, v].mean(0))
        assert torch.allclose(phi[v], want, atol=1e-6)


def test_multi_term_minus_is_averaged():
    torch.manual_seed(1)
    V, D = 2, 5
    Ft = torch.randn(80, V, D)
    yt = np.repeat(np.arange(4), 20)
    phi = synth_attr_prototype(Ft, yt, [0, 2], [0, 1, 2, 3], V)
    m = torch.stack([Ft[yt == k, 0].mean(0) for k in range(4)]).mean(0)
    want = Ft[yt == 0, 0].mean(0) + Ft[yt == 2, 0].mean(0) - m
    assert torch.allclose(phi[0], want, atol=1e-6)


def test_synth_samples_have_spread_and_right_centre():
    torch.manual_seed(2)
    V, D = 3, 6
    Ft = torch.randn(150, V, D)
    yt = np.repeat(np.arange(5), 30)
    S = synth_attr_samples(Ft, yt, [1, 3], [2], V, n=200,
                           rng=np.random.RandomState(0))
    assert S.shape == (200, V, D)
    assert S.std(0).mean() > 0.5, "synthetic positives collapsed to a point"
    centre = S.mean(0)
    phi = synth_attr_prototype(Ft, yt, [1, 3], [2], V)
    # the sample cloud must be centred on the analytic prototype
    assert torch.allclose(centre, phi, atol=0.35)


def test_per_class_seed_isolates_rng():
    """Re-seeding must make an inspector independent of earlier RNG consumption.

    This is what protects the gate's negative controls: giving class 6 synthetic
    positives changes len(pos), hence the draws AND the batch count inside
    train_inspector, which would otherwise perturb every class trained after it.
    """
    from config import Config
    from uwb_osr.inspector import train_inspector

    cfg = Config()
    cfg.inspector_epochs, cfg.batch_size_insp = 2, 64
    cfg.n_pos_syn, cfg.n_neg_syn = 32, 32
    pos = torch.randn(6, 16)
    neg = torch.randn(60, 16)

    def run(burn, reseed):
        torch.manual_seed(7)
        torch.randn(burn)                      # earlier classes consuming the RNG
        if reseed:
            torch.manual_seed(1234)
        _, phi, sigma = train_inspector(pos, neg, cfg, "cpu")
        return phi.detach().clone(), sigma

    a_phi, a_s = run(0, True)
    b_phi, b_s = run(500, True)
    assert torch.allclose(a_phi, b_phi, atol=0), "re-seeded runs must be identical"
    assert a_s == b_s

    c_phi, _ = run(0, False)
    d_phi, _ = run(500, False)
    assert not torch.allclose(c_phi, d_phi, atol=1e-6), \
        "without re-seeding the runs should differ (otherwise the guard is moot)"


def test_off_switch_maps_nothing():
    case = REG_TEST_CASES[1]
    lm = label_map_of(case)
    assert _map_attr_spec(None, lm, n_base=6) == {}
    assert _map_attr_spec({}, lm, n_base=6) == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok  {f.__name__}")
    print(f"{len(fns)} passed")
