"""P4 three-action risk controller + online conformal state (SS5.5-5.6).

Label discipline (SS11.1): the controller sees scores, thresholds and flags
only. Hidden labels surface exactly twice -- when the caller answers a
query (the user's label) and inside the offline evaluator -- and the ONLY
path into the online conformal state is a random-calibration-slice
confirmation (`OnlineConformal.update(..., from_random_slice=True)`);
uncertainty-slice confirmations touching threshold state is a test
failure, enforced here by a hard raise.

Actions per stream event (SS5.5):
  adopt       singleton set on a register class, top-1 score <= the fixed
              q80 gate (80th pct of calib true-class scores), no view
              alarm, not in conservative mode -> plastic library at low
              weight; NEVER enters the conformal queue.
  query_rand  pre-scheduled random-calibration slot (stream.py chose it,
              score-independent): label updates online conformal AND may
              enter the plastic pool at weight 1.
  query_unc   uncertainty trigger (|C|>1, scale fallback, coverage alarm,
              or q80 < top-1 <= q) within the remaining per-100 budget:
              label updates plastic memory ONLY.
  accept      covered sample, no state change.
  reject      empty set (or nothing else applies): no state change.

Online conformal (SS5.6): additive offset delta on the z-score threshold
vector, ACI-style update delta += eta_t * (miss - alpha) with
eta_t = 1/sqrt(t+1), t counting random-slice confirmations only.

Coverage-drift alarm (SS5.6 level 3, biased monitor): empirical coverage
of the last `alarm_window`=20 confirmed samples (both slices) outside
[0.85, 0.95] -> conservative mode: auto-adoption off, queries prioritised;
never triggered with fewer than 20 confirmations.

Routing stays decoupled (SS5.1/SS11.2): `route()` applies the scale s to
base columns for argmin only; acceptance always reads unscaled scores.
"""
from collections import defaultdict, deque
from typing import Optional

import numpy as np


def route(R_row: np.ndarray, nb: int, s: float) -> int:
    """Routing decision: argmin with base columns handicapped by s.
    The prediction-set / acceptance side never sees s."""
    Rs = np.asarray(R_row, float).copy()
    Rs[:nb] *= s
    return int(np.argmin(Rs))


class OnlineConformal:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.delta = 0.0
        self.t = 0
        self.hits = []

    def update(self, hit: bool, from_random_slice: bool = False):
        if not from_random_slice:
            raise ValueError("online conformal state only accepts random-"
                             "calibration-slice labels (SS5.5/SS11.1)")
        eta = 1.0 / np.sqrt(self.t + 1)
        self.delta += eta * ((0.0 if hit else 1.0) - self.alpha)
        self.t += 1
        self.hits.append(bool(hit))

    def q_eff(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, float) + self.delta


class RiskController:
    def __init__(self, alpha: float, budget_per_100: int, b_rand: int,
                 q80: float, online: Optional[OnlineConformal] = None,
                 alarm_window: int = 20, alarm_lo: float = 0.85,
                 alarm_hi: float = 0.95):
        self.alpha = float(alpha)
        self.B = int(budget_per_100)
        self.b_rand = int(b_rand)
        self.q80 = float(q80)
        self.online = online if online is not None else OnlineConformal(alpha)
        self._spend = defaultdict(lambda: {"rand": 0, "unc": 0})
        self._confirmed = deque(maxlen=alarm_window)
        self.alarm_window = alarm_window
        self.alarm_lo, self.alarm_hi = alarm_lo, alarm_hi

    # -- monitoring ---------------------------------------------------------
    @property
    def coverage_alarm(self) -> bool:
        if len(self._confirmed) < self.alarm_window:
            return False
        c = float(np.mean(self._confirmed))
        return c < self.alarm_lo or c > self.alarm_hi

    @property
    def conservative(self) -> bool:
        return self.coverage_alarm

    def record_confirmation(self, hit: bool):
        """Every answered query (either slice) feeds the biased alarm
        monitor; it never touches threshold state."""
        self._confirmed.append(bool(hit))

    def spend_summary(self):
        return {b: dict(v) for b, v in self._spend.items()}

    # -- the decision -------------------------------------------------------
    def decide(self, t: int, S_row: np.ndarray, q_static: np.ndarray,
               nb: int, random_query: bool = False,
               views_alert: bool = False, scale_fallback: bool = False):
        """One stream event -> (action, pred_col). pred_col is the singleton
        column for adopt, the tentative top-1 for queries/accept, None for
        reject. The caller owns all state updates that need labels."""
        S_row = np.asarray(S_row, float)
        q = self.online.q_eff(np.asarray(q_static, float))
        member = S_row <= q
        size = int(member.sum())
        top1 = int(np.argmin(S_row))
        top1_score = float(S_row[top1])
        block = t // 100
        sp = self._spend[block]

        if random_query and sp["rand"] < self.b_rand \
                and sp["rand"] + sp["unc"] < self.B:
            sp["rand"] += 1
            return "query_rand", top1

        if (size == 1 and not views_alert and not self.conservative
                and top1_score <= self.q80):
            col = int(np.argmax(member))
            if col >= nb:                          # plastic covers new classes
                return "adopt", col
            return "accept", col

        uncertain = (size > 1 or scale_fallback or self.coverage_alarm
                     or (size == 1 and self.q80 < top1_score))
        if uncertain and sp["unc"] < self.B - self.b_rand \
                and sp["rand"] + sp["unc"] < self.B:
            sp["unc"] += 1
            return "query_unc", top1

        if size == 0:
            return "reject", None
        return "accept", int(np.argmax(member)) if size == 1 else top1
