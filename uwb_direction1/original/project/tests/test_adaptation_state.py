"""P4 state tests (design doc SS11.2 + budget/slice discipline of SS5.5).

Runs on tiny synthetic tensors with identity inspectors (insp=None), no
GPU and no training, mirroring how proto_dist handles an NCM bank.
"""
import copy
import unittest

import numpy as np
import torch

from uwb_osr.adaptation import DualMemory
from uwb_osr.risk_control import OnlineConformal, RiskController, route

NB, NR, V, D = 3, 2, 2, 4                 # 3 base + 2 register classes


def make_bank(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {(k, v): (None, torch.randn(D, generator=g), 1.0)
            for k in range(NB + NR) for v in range(V)}


def make_memory(cap=20, w_auto=0.25):
    bank = make_bank()
    sup = torch.randn(2 * 3, V, D, generator=torch.Generator().manual_seed(1))
    sup_y = [NB, NB, NB, NB + 1, NB + 1, NB + 1]      # 3 shots per new class
    mem = DualMemory(bank, NB, NR, sup, sup_y, "cpu", cap=cap, w_auto=w_auto)
    return bank, mem


def snapshot(bank):
    return {k: (v[0], v[1].clone(), v[2]) for k, v in bank.items()}


class TestDualMemory(unittest.TestCase):
    def test_stable_bank_never_mutated(self):
        bank, mem = make_memory()
        before = snapshot(bank)
        for i in range(5):
            mem.add(NB, torch.randn(V, D), "auto")
            mem.add(NB + 1, torch.randn(V, D), "confirm")
        for k in bank:
            self.assertTrue(torch.equal(bank[k][1], before[k][1]))
            self.assertEqual(bank[k][2], before[k][2])

    def test_overlay_covers_only_register_classes(self):
        _, mem = make_memory()
        mem.add(NB, torch.randn(V, D), "confirm")
        self.assertTrue(all(c >= NB for (c, _) in mem.overlay))
        with self.assertRaises(ValueError):
            mem.add(0, torch.randn(V, D), "confirm")

    def test_auto_weight_capped(self):
        with self.assertRaises(ValueError):
            make_memory(w_auto=0.5)

    def test_rollback_restores_initial_prototypes(self):
        bank, mem = make_memory()
        eff0 = {k: (v[1].clone(), v[2]) for k, v in mem.bank().items()}
        for i in range(4):
            mem.add(NB, torch.randn(V, D), "auto")
        changed = mem.bank()[(NB, 0)]
        self.assertFalse(torch.equal(changed[1], eff0[(NB, 0)][0]))
        mem.rollback()
        eff1 = mem.bank()
        for k in eff0:
            self.assertTrue(torch.equal(eff1[k][1], eff0[k][0]))
            self.assertEqual(eff1[k][2], eff0[k][1])
        stats = mem.pool_stats()
        self.assertEqual(stats[NB]["auto"], 0)
        self.assertEqual(stats[NB]["confirm"], 0)

    def test_capacity_evicts_auto_first_support_never(self):
        _, mem = make_memory(cap=3)
        for i in range(3):
            mem.add(NB, torch.randn(V, D), "auto")
        mem.add(NB, torch.randn(V, D), "confirm")
        st = mem.pool_stats()[NB]
        self.assertEqual(st["auto"] + st["confirm"], 3)   # cap respected
        self.assertEqual(st["confirm"], 1)                # auto evicted first
        self.assertEqual(st["n"] - st["auto"] - st["confirm"], 3)  # supports

    def test_weighted_prototype_matches_manual(self):
        bank, mem = make_memory()
        x = torch.randn(V, D)
        mem.add(NB, x, "auto")
        pool = mem.pools[NB]
        feats = torch.stack([f for f, _, _ in pool])
        w = torch.tensor([wi for _, wi, _ in pool])
        w = w / w.sum()
        for v in range(V):
            phi = (w[:, None] * feats[:, v]).sum(0)
            self.assertTrue(torch.allclose(mem.bank()[(NB, v)][1], phi,
                                           atol=1e-6))

    def test_sigma_frozen_under_plastic_updates(self):
        # sigma recompute is the runaway lever (G2 smoke: far-away adopted
        # features inflate sigma, r=d/sigma shrinks, the column swallows
        # everything) -- plastic updates must move phi only.
        bank, mem = make_memory()
        for _ in range(6):
            mem.add(NB, 100.0 * torch.randn(V, D), "confirm")
        for v in range(V):
            self.assertEqual(mem.bank()[(NB, v)][2], bank[(NB, v)][2])


class TestOnlineConformal(unittest.TestCase):
    def test_uncertainty_slice_cannot_touch_state(self):
        oc = OnlineConformal(0.1)
        with self.assertRaises(ValueError):
            oc.update(True)
        with self.assertRaises(ValueError):
            oc.update(True, from_random_slice=False)
        self.assertEqual(oc.t, 0)

    def test_direction_and_decay(self):
        oc = OnlineConformal(0.1)
        oc.update(False, from_random_slice=True)          # miss -> widen
        self.assertGreater(oc.delta, 0)
        d1 = oc.delta
        oc.update(True, from_random_slice=True)           # hit -> shrink
        self.assertLess(oc.delta, d1)
        # step size decays as 1/sqrt(t+1)
        oc2 = OnlineConformal(0.1)
        oc2.update(False, from_random_slice=True)
        step1 = oc2.delta
        oc2.update(False, from_random_slice=True)
        self.assertLess(oc2.delta - step1, step1)

    def test_q_eff_offsets_every_class(self):
        oc = OnlineConformal(0.1)
        oc.update(False, from_random_slice=True)
        q = np.zeros(NB + NR)
        self.assertTrue(np.allclose(oc.q_eff(q), oc.delta))


class TestRiskController(unittest.TestCase):
    def make(self, B=5, b_rand=1, q80=0.5):
        return RiskController(0.1, B, b_rand, q80)

    def test_adopt_never_touches_conformal_queue(self):
        rc = self.make()
        S = np.array([9, 9, 9, 0.1, 9], float)            # singleton on col 3
        a, col = rc.decide(0, S, np.ones(NB + NR), NB)
        self.assertEqual(a, "adopt")
        self.assertEqual(col, 3)
        self.assertEqual(rc.online.t, 0)
        self.assertEqual(len(rc.online.hits), 0)

    def test_base_singleton_accepts_instead_of_adopting(self):
        rc = self.make()
        S = np.array([0.1, 9, 9, 9, 9], float)
        a, col = rc.decide(0, S, np.ones(NB + NR), NB)
        self.assertEqual(a, "accept")
        self.assertEqual(col, 0)

    def test_budget_never_exceeded_per_block(self):
        B, b_rand = 5, 1
        rc = self.make(B=B, b_rand=b_rand)
        S_unc = np.array([0.1, 0.2, 9, 9, 9], float)      # |C|=2 -> query_unc
        q = np.ones(NB + NR)
        for t in range(300):
            rc.decide(t, S_unc, q, NB, random_query=(t % 100 == 7))
        for blk, sp in rc.spend_summary().items():
            self.assertLessEqual(sp["rand"] + sp["unc"], B)
            self.assertLessEqual(sp["rand"], b_rand)
            self.assertLessEqual(sp["unc"], B - b_rand)

    def test_random_slot_served_even_on_easy_sample(self):
        rc = self.make()
        S_easy = np.array([0.1, 9, 9, 9, 9], float)
        a, _ = rc.decide(0, S_easy, np.ones(NB + NR), NB, random_query=True)
        self.assertEqual(a, "query_rand")

    def test_views_alert_blocks_adoption(self):
        rc = self.make()
        S = np.array([9, 9, 9, 0.1, 9], float)
        a, _ = rc.decide(0, S, np.ones(NB + NR), NB, views_alert=True)
        self.assertNotEqual(a, "adopt")

    def test_conservative_mode_after_coverage_alarm(self):
        rc = self.make()
        for _ in range(20):
            rc.record_confirmation(False)                 # coverage 0 < 0.85
        self.assertTrue(rc.conservative)
        S = np.array([9, 9, 9, 0.1, 9], float)
        a, _ = rc.decide(0, S, np.ones(NB + NR), NB)
        self.assertNotEqual(a, "adopt")
        rc2 = self.make()
        for _ in range(19):                               # < window: no alarm
            rc2.record_confirmation(False)
        self.assertFalse(rc2.conservative)

    def test_empty_set_rejects_without_nan(self):
        rc = self.make()
        S = np.full(NB + NR, 50.0)
        a, col = rc.decide(0, S, np.ones(NB + NR), NB, views_alert=True)
        self.assertEqual(a, "reject")
        self.assertIsNone(col)

    def test_routing_scale_never_changes_accept_sets(self):
        R = np.array([1.0, 3, 3, 1.1, 3])
        q = np.full(NB + NR, 1.2)
        member_unscaled = R <= q
        for s in (1.0, 1.5, 2.5):
            pred = route(R, NB, s)
            self.assertTrue(np.array_equal(R <= q, member_unscaled))
        self.assertEqual(route(R, NB, 1.0), 0)            # s flips routing...
        self.assertEqual(route(R, NB, 1.5), 3)            # ...but not the set


if __name__ == "__main__":
    unittest.main()
