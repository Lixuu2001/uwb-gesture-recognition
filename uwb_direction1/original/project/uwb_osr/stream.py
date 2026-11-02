"""P4 controlled open-world stream protocol (design doc SS4.3-4.5, SS5.5).

This module is the *protocol layer* only: it turns a test subject's sample
pools into a deterministic pseudo-time schedule of stream events, with the
degradation timetable and the feedback-budget random-calibration slots baked
in.  It deliberately knows NOTHING about labels, features, or model outputs
(SS11.1 leakage discipline): the builder receives index pools, and every
random draw comes from generators seeded only by (spec.seed, stream role).

Five segments (SS4.3), fractions pre-registered in StreamSpec:
  1 cold      clean mixed base/register/unknown flow
  2 ramp      ONE view's degradation severity ramps 0 -> 1 (snapped to grid)
  3 arrival   register-class frequency rises; severity holds at 1
  4 compound  degradation persists + class-prior shift + unknown surge
  5 recovery  severity ramps back 1 -> 0

Severity is discretised to SEVERITY_GRID (SS4.5, dev 5 levels).  The class-
prior changes are realised through the per-segment allocation matrix, not
through label peeking.  Feedback budget slicing (SS5.5, 2026-07-12 revision):
per 100-sample block, B_rand = max(1, round(0.2*B)) slots are drawn uniformly
without replacement from a dedicated generator -- these are the ONLY events
allowed to update online conformal state downstream.

v1 pre-registered defaults live in StreamSpec; change them only before the
first full run, never after looking at test results.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

SEVERITY_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

# degradation families (SS4.4); "domain" is schedule-tag only (the runner
# swaps Raw/ClutterRemoved inputs), the rest transform the view tensor here.
DEGRADATIONS = ("atten", "noise", "shift", "occlude", "drop", "domain")


def snap_severity(v: float) -> float:
    """Snap a continuous severity to the nearest pre-registered grid level."""
    g = np.asarray(SEVERITY_GRID)
    return float(g[np.argmin(np.abs(g - v))])


@dataclass(frozen=True)
class StreamSpec:
    """Pre-registered v1 stream parameters (design doc SS4.3/SS4.5/SS5.5)."""
    seed: int
    budget_per_100: int = 5                 # B; 0 disables all querying
    deg_type: str = "noise"                 # segment-2/4 degradation family
    deg_view: int = 0                       # view index degraded in seg 2-5
    # segment fractions of the stream length (cold/ramp/arrival/compound/rec)
    seg_frac: Sequence[float] = (0.15, 0.20, 0.20, 0.30, 0.15)
    # per-group allocation across the 5 segments (rows sum to 1):
    # register mass concentrates in arrival/compound (class arrival + prior
    # shift); unknown mass in cold and compound (SS4.3).
    alloc_base: Sequence[float] = (0.20, 0.20, 0.15, 0.25, 0.20)
    alloc_reg: Sequence[float] = (0.05, 0.10, 0.40, 0.35, 0.10)
    alloc_unk: Sequence[float] = (0.25, 0.15, 0.10, 0.35, 0.15)
    max_shift_bins: int = 10                # "shift" at sev=1 rolls this many
    occlude_frac: float = 0.5               # "occlude" at sev=1 zeros T/2

    @property
    def b_rand(self) -> int:
        """Random-calibration-slice queries per 100-sample block (SS5.5)."""
        if self.budget_per_100 <= 0:
            return 0
        return max(1, int(round(0.2 * self.budget_per_100)))


@dataclass(frozen=True)
class StreamEvent:
    t: int                      # position in the stream
    sample_idx: int             # row index into the caller's sample store
    segment: int                # 1..5
    severity: float             # SEVERITY_GRID value; 0 => clean
    deg_type: Optional[str]     # None when severity == 0
    deg_view: Optional[int]     # None when severity == 0
    random_query: bool          # random calibration slice slot (SS5.5)


def _quota(n: int, fracs: Sequence[float]) -> List[int]:
    """Largest-remainder split of n items across segments (deterministic)."""
    raw = [n * f for f in fracs]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = np.argsort([b - r for b, r in zip(base, raw)])  # largest frac 1st
    for i in range(rem):
        base[int(order[i])] += 1
    return base


def _segment_severity(seg: int, pos: int, seg_len: int) -> float:
    """Severity for event #pos (0-based) inside segment seg (1-based)."""
    if seg == 1:
        return 0.0
    if seg == 2:                       # ramp 0 -> 1 across the segment
        frac = (pos + 1) / max(seg_len, 1)
        return snap_severity(frac)
    if seg in (3, 4):                  # hold at full severity (SS4.3 v1)
        return 1.0
    frac = (pos + 1) / max(seg_len, 1)  # seg 5: ramp 1 -> 0
    return snap_severity(1.0 - frac)


def build_stream(base_idx: Sequence[int], reg_idx: Sequence[int],
                 unk_idx: Sequence[int], spec: StreamSpec) -> List[StreamEvent]:
    """Deterministic stream schedule from a test subject's index pools.

    The three pools must be disjoint; every index appears exactly once in
    the returned schedule.  No labels are accepted or consulted.
    """
    if spec.deg_type not in DEGRADATIONS:
        raise ValueError(f"unknown degradation {spec.deg_type!r}")
    if abs(sum(spec.seg_frac) - 1.0) > 1e-9:
        raise ValueError("seg_frac must sum to 1")
    pools = [np.asarray(p, dtype=np.int64) for p in
             (base_idx, reg_idx, unk_idx)]
    allocs = (spec.alloc_base, spec.alloc_reg, spec.alloc_unk)
    for a in allocs:
        if abs(sum(a) - 1.0) > 1e-9:
            raise ValueError("allocation rows must sum to 1")
    flat = np.concatenate(pools)
    if len(np.unique(flat)) != len(flat):
        raise ValueError("sample pools must be disjoint")

    # role-separated generators: event order and query slots never share
    # entropy, so slot positions depend only on (seed, stream length).
    rng_order = np.random.default_rng([spec.seed, 1])

    # per-segment membership: draw each pool's quota without replacement,
    # in a fixed pool order, then shuffle the union within the segment.
    per_seg: List[List[int]] = [[] for _ in range(5)]
    for pool, alloc in zip(pools, allocs):
        q = _quota(len(pool), alloc)
        perm = rng_order.permutation(len(pool))
        start = 0
        for s in range(5):
            take = pool[perm[start:start + q[s]]]
            per_seg[s].extend(int(i) for i in take)
            start += q[s]
    for s in range(5):
        order = rng_order.permutation(len(per_seg[s]))
        per_seg[s] = [per_seg[s][int(i)] for i in order]

    n_total = sum(len(s) for s in per_seg)
    slots = _random_query_slots(n_total, spec)

    events: List[StreamEvent] = []
    t = 0
    for s in range(5):
        seg_len = len(per_seg[s])
        for pos, idx in enumerate(per_seg[s]):
            sev = _segment_severity(s + 1, pos, seg_len)
            events.append(StreamEvent(
                t=t, sample_idx=idx, segment=s + 1, severity=sev,
                deg_type=spec.deg_type if sev > 0 else None,
                deg_view=spec.deg_view if sev > 0 else None,
                random_query=t in slots))
            t += 1
    return events


def _random_query_slots(n_events: int, spec: StreamSpec) -> set:
    """Random-calibration slots: b_rand per 100-block, seeded, label-free."""
    if spec.b_rand == 0:
        return set()
    rng = np.random.default_rng([spec.seed, 2])
    slots = set()
    for start in range(0, n_events, 100):
        block = min(100, n_events - start)
        k = min(spec.b_rand, block)
        picks = rng.choice(block, size=k, replace=False)
        slots.update(int(start + p) for p in picks)
    return slots


def apply_degradation(x: np.ndarray, view: int, deg_type: str,
                      severity: float, seed: int, t: int,
                      max_shift_bins: int = 10,
                      occlude_frac: float = 0.5) -> np.ndarray:
    """Deterministically degrade ONE view of a (V, T, M) sample.

    severity must be a SEVERITY_GRID value; 0 returns the input unchanged.
    Stochastic families (noise, occlude position) draw from a generator
    seeded by (seed, t, view) so the same stream event always produces the
    same corruption.  "domain" is a schedule tag handled by the runner and
    is an identity here.
    """
    if severity not in SEVERITY_GRID:
        raise ValueError(f"severity {severity} not on grid")
    if deg_type not in DEGRADATIONS:
        raise ValueError(f"unknown degradation {deg_type!r}")
    if severity == 0.0 or deg_type == "domain":
        return x.copy()
    out = x.copy()
    v = out[view]
    rng = np.random.default_rng([seed, t, view])
    if deg_type == "atten":
        out[view] = v * (1.0 - 0.9 * severity)
    elif deg_type == "drop":                  # partial -> full view drop
        out[view] = v * (1.0 - severity)
    elif deg_type == "noise":
        out[view] = v + severity * v.std() * rng.standard_normal(v.shape)
    elif deg_type == "shift":                 # fast-time/range-bin shift
        out[view] = np.roll(v, int(round(severity * max_shift_bins)), axis=-1)
    elif deg_type == "occlude":               # slow-time block occlusion
        T = v.shape[0]
        blk = int(round(severity * occlude_frac * T))
        if blk > 0:
            start = int(rng.integers(0, max(T - blk, 1)))
            out[view, start:start + blk, :] = 0.0
    return out
