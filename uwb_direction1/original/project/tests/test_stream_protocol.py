"""Protocol tests for the P4 controlled stream (design doc SS11.1).

Covers: determinism per seed, seed-independence of the structure, index
integrity (every pool index exactly once, disjointness enforced), severity
grid + per-segment monotonicity, random-calibration-slot budget and its
independence from pool contents (label-freedom by construction).
"""
import unittest

import numpy as np

from uwb_osr.stream import (SEVERITY_GRID, StreamSpec, apply_degradation,
                            build_stream, snap_severity)


def _pools(n_base=600, n_reg=300, n_unk=300, offset=0):
    base = list(range(offset, offset + n_base))
    reg = list(range(offset + n_base, offset + n_base + n_reg))
    unk = list(range(offset + n_base + n_reg,
                     offset + n_base + n_reg + n_unk))
    return base, reg, unk


class StreamScheduleTest(unittest.TestCase):
    def setUp(self):
        self.spec = StreamSpec(seed=7)
        self.base, self.reg, self.unk = _pools()

    def test_same_seed_identical_schedule(self):
        a = build_stream(self.base, self.reg, self.unk, self.spec)
        b = build_stream(self.base, self.reg, self.unk, self.spec)
        self.assertEqual(a, b)

    def test_different_seed_same_structure_different_order(self):
        a = build_stream(self.base, self.reg, self.unk, self.spec)
        b = build_stream(self.base, self.reg, self.unk, StreamSpec(seed=8))
        self.assertNotEqual([e.sample_idx for e in a],
                            [e.sample_idx for e in b])
        # segment lengths (the pre-registered structure) must not move
        for s in range(1, 6):
            self.assertEqual(len([e for e in a if e.segment == s]),
                             len([e for e in b if e.segment == s]))

    def test_every_index_exactly_once(self):
        ev = build_stream(self.base, self.reg, self.unk, self.spec)
        seen = sorted(e.sample_idx for e in ev)
        self.assertEqual(seen, sorted(self.base + self.reg + self.unk))

    def test_overlapping_pools_rejected(self):
        with self.assertRaises(ValueError):
            build_stream([1, 2], [2, 3], [4], self.spec)

    def test_severity_on_grid_and_segment_shape(self):
        ev = build_stream(self.base, self.reg, self.unk, self.spec)
        for e in ev:
            self.assertIn(e.severity, SEVERITY_GRID)
            if e.severity == 0:
                self.assertIsNone(e.deg_type)
                self.assertIsNone(e.deg_view)
        seg = {s: [e.severity for e in ev if e.segment == s]
               for s in range(1, 6)}
        self.assertTrue(all(v == 0 for v in seg[1]))          # cold is clean
        self.assertEqual(seg[2], sorted(seg[2]))              # ramp up
        self.assertTrue(all(v == 1.0 for v in seg[3] + seg[4]))
        self.assertEqual(seg[5], sorted(seg[5], reverse=True))  # recovery
        self.assertEqual(seg[5][-1], 0.0)                     # ends clean

    def test_register_mass_concentrates_in_arrival_compound(self):
        ev = build_stream(self.base, self.reg, self.unk, self.spec)
        reg_set = set(self.reg)
        in34 = sum(e.segment in (3, 4) for e in ev
                   if e.sample_idx in reg_set)
        self.assertGreaterEqual(in34 / len(self.reg), 0.7)

    def test_random_slots_per_block_budget(self):
        for b in (0, 2, 5, 10):
            spec = StreamSpec(seed=7, budget_per_100=b)
            ev = build_stream(self.base, self.reg, self.unk, spec)
            expect = 0 if b == 0 else max(1, round(0.2 * b))
            for start in range(0, len(ev), 100):
                block = ev[start:start + 100]
                got = sum(e.random_query for e in block)
                self.assertEqual(got, min(expect, len(block)),
                                 f"B={b} block@{start}")

    def test_random_slots_independent_of_pool_contents(self):
        # same pool SIZES, disjoint different indices -> identical slot
        # positions: slots cannot encode anything about the samples.
        a = build_stream(*_pools(), self.spec)
        b = build_stream(*_pools(offset=10_000), self.spec)
        self.assertEqual([e.random_query for e in a],
                         [e.random_query for e in b])


class DegradationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((3, 90, 189)).astype(np.float32)

    def test_zero_severity_is_identity(self):
        for d in ("atten", "noise", "shift", "occlude", "drop", "domain"):
            out = apply_degradation(self.x, 1, d, 0.0, seed=7, t=3)
            np.testing.assert_array_equal(out, self.x)

    def test_deterministic_per_event(self):
        a = apply_degradation(self.x, 1, "noise", 0.5, seed=7, t=3)
        b = apply_degradation(self.x, 1, "noise", 0.5, seed=7, t=3)
        c = apply_degradation(self.x, 1, "noise", 0.5, seed=7, t=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_only_target_view_touched(self):
        for d in ("atten", "noise", "shift", "occlude", "drop"):
            out = apply_degradation(self.x, 1, d, 0.75, seed=7, t=3)
            np.testing.assert_array_equal(out[0], self.x[0])
            np.testing.assert_array_equal(out[2], self.x[2])
            self.assertFalse(np.array_equal(out[1], self.x[1]), d)

    def test_drop_full_at_severity_one(self):
        out = apply_degradation(self.x, 2, "drop", 1.0, seed=7, t=3)
        np.testing.assert_array_equal(out[2], np.zeros_like(self.x[2]))

    def test_off_grid_severity_rejected(self):
        with self.assertRaises(ValueError):
            apply_degradation(self.x, 0, "noise", 0.3, seed=7, t=3)

    def test_snap_severity(self):
        self.assertEqual(snap_severity(0.3), 0.25)
        self.assertEqual(snap_severity(0.9), 1.0)
        self.assertEqual(snap_severity(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
