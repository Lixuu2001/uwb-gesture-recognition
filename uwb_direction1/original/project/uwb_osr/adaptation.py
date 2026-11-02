"""P4 stable-plastic dual memory (design doc SS5.2, SS11.2).

Stable library = the frozen registration bank (base + initial new-class
inspectors, prototypes, scales). It is READ-ONLY for the whole stream:
this module never mutates the caller's dict or its tensors; the effective
bank is stable merged with a plastic overlay that only ever covers
register-class keys.

Plastic library per register class: a bounded pool of
(f_views, weight, source) with source in {"support", "confirm", "auto"}.
Confirmed samples weigh 1.0, auto-adopted samples w_auto <= 0.25 (SS5.2).
Prototype recompute keeps the inspector nets FROZEN (a T3A-style template
update -- the SS6.1 TTA baseline's mechanism, which keeps the method's
delta attributable to the risk gate / dual library / budget):

    phi_v = sum_i w_i * insp_v(f_i^v) / sum_i w_i

SIGMA STAYS STABLE. Recomputing sigma from the pool is a runaway lever:
cross-user (and under drift, degraded) features inflate the weighted RMS,
r = d/sigma shrinks for EVERYONE, the class column swallows base samples
and its own contaminated adoptions, which inflate sigma further (observed
on the first real G2 smoke: clean-stream H 92.5 -> 58.2, pool
contamination up to 83%). Scale re-estimation needs an inspector-level
retrain and belongs to an offline end-of-stream consolidation, not the
online loop. Capacity: `cap` non-support samples per class, oldest "auto"
evicted first, "support" never evicted (source-balanced retention).
rollback() restores the initial registration state exactly (SS11.2).
"""
from typing import Dict, List, Tuple

import torch

Source = str                              # "support" | "confirm" | "auto"


class DualMemory:
    def __init__(self, bank: Dict, n_base: int, n_reg: int,
                 support_feats: torch.Tensor, support_y, device,
                 cap: int = 20, w_auto: float = 0.25):
        if w_auto > 0.25:
            raise ValueError("auto-adopt weight must be <= 0.25 (SS5.2)")
        self.stable = dict(bank)          # shallow copy; entries never replaced
        self.nb, self.nr = n_base, n_reg
        self.V = max(v for (_, v) in bank) + 1
        self.device = device
        self.cap, self.w_auto = cap, w_auto
        self.pools: Dict[int, List[Tuple[torch.Tensor, float, Source]]] = {}
        for c in range(n_base, n_base + n_reg):
            sel = [i for i, y in enumerate(support_y) if int(y) == c]
            self.pools[c] = [(support_feats[i].detach().cpu(), 1.0, "support")
                             for i in sel]
        self.overlay: Dict = {}
        self.n_updates = 0

    # -- queries ------------------------------------------------------------
    def bank(self) -> Dict:
        """Effective bank: stable entries shadowed by the plastic overlay."""
        merged = dict(self.stable)
        merged.update(self.overlay)
        return merged

    def pool_stats(self):
        return {c: {"n": len(p),
                    "auto": sum(1 for _, _, s in p if s == "auto"),
                    "confirm": sum(1 for _, _, s in p if s == "confirm")}
                for c, p in self.pools.items()}

    # -- updates ------------------------------------------------------------
    def add(self, c: int, f_views: torch.Tensor, source: Source):
        """Add one sample (V, D feature per view) to class c's plastic pool
        and refresh that class's prototypes/scales. Base classes are stable
        and must never arrive here."""
        if not (self.nb <= c < self.nb + self.nr):
            raise ValueError(f"class {c} is stable (base); plastic only "
                             "covers register classes")
        if source not in ("confirm", "auto"):
            raise ValueError(f"bad source {source!r}")
        w = 1.0 if source == "confirm" else self.w_auto
        self.pools[c].append((f_views.detach().cpu(), w, source))
        self._evict(c)
        self._recompute(c)
        self.n_updates += 1

    def rollback(self):
        """Restore the initial registration state (SS11.2): support-only
        pools, no overlay."""
        for c in self.pools:
            self.pools[c] = [e for e in self.pools[c] if e[2] == "support"]
        self.overlay.clear()

    # -- internals ----------------------------------------------------------
    def _evict(self, c):
        pool = self.pools[c]
        while sum(1 for _, _, s in pool if s != "support") > self.cap:
            for kind in ("auto", "confirm"):       # oldest auto first
                idx = next((i for i, e in enumerate(pool) if e[2] == kind),
                           None)
                if idx is not None:
                    pool.pop(idx)
                    break

    @torch.no_grad()
    def _recompute(self, c):
        feats = torch.stack([f for f, _, _ in self.pools[c]]).to(self.device)
        w = torch.tensor([wi for _, wi, _ in self.pools[c]],
                         device=self.device, dtype=feats.dtype)
        w = w / w.sum()
        for v in range(self.V):
            insp, _, stable_sigma = self.stable[(c, v)]
            lat = insp(feats[:, v]) if insp is not None else feats[:, v]
            phi = (w[:, None] * lat).sum(0)
            self.overlay[(c, v)] = (insp, phi.detach(), stable_sigma)
