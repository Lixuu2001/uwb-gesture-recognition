"""Global configuration for the Multi-View UWB Open-Set Recognition (MV-OSR) project.

A single dataclass drives every module so ablations are one-line changes.
"""
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
DATA_ROOT = r"D:\uwb_gestures\12652592"
CACHE_DIR = r"D:\uwb_gestures\cache"

# Gesture index (0..11) -> name. Index == (G number - 1) == Eom letter (a..l).
GESTURE_NAMES = {
    0: "Left-right swipe",      # G1  a
    1: "Right-left swipe",      # G2  b
    2: "Up-down swipe",         # G3  c
    3: "Down-up swipe",         # G4  d
    4: "Diag LR-UD swipe",      # G5  e
    5: "Diag LR-DU swipe",      # G6  f
    6: "Diag RL-UD swipe",      # G7  g
    7: "Diag RL-DU swipe",      # G8  h
    8: "Clockwise",             # G9  i
    9: "Anti-clockwise",        # G10 j
    10: "Inward push",          # G11 k
    11: "Empty gesture",        # G12 l
}

# View order used everywhere (must stay consistent: slot 0/1/2).
VIEW_ORDER = ["Left", "Top", "Right"]

# Approximate radar geometry from the dataset description (metres). Used only by
# the optional physical-bias module; left/right ~1.1 m apart, top ~0.55 m above
# the horizontal midpoint.
VIEW_POSITIONS = {
    "Left":  (-0.55, 0.0),
    "Right": (0.55, 0.0),
    "Top":   (0.0, 0.55),
}

# ---------------------------------------------------------------------------
# Eom IoTJ-2025 Table I open-set test cases (letter a..l -> index 0..11).
# Each case: 6 known classes / 6 unknown classes.
# ---------------------------------------------------------------------------
OSR_TEST_CASES = [
    {"known": [1, 4, 5, 7, 8, 11], "unknown": [0, 2, 3, 6, 9, 10]},   # case 1
    {"known": [1, 4, 5, 7, 10, 11], "unknown": [0, 2, 3, 6, 8, 9]},   # case 2
    {"known": [2, 5, 6, 7, 9, 11], "unknown": [0, 1, 3, 4, 8, 10]},   # case 3
    {"known": [0, 1, 7, 8, 10, 11], "unknown": [2, 3, 4, 5, 6, 9]},   # case 4
    {"known": [1, 2, 7, 8, 9, 10], "unknown": [0, 3, 4, 5, 6, 11]},   # case 5
]

# ---------------------------------------------------------------------------
# Paper-3 few-shot gesture registration protocol (N-series, see
# docs/RESEARCH_PLAN.md SS4). Each case splits the 12 gestures into:
#   base     -> trains the encoder + base inspectors (the deployed system)
#   register -> the new gestures added few-shot at registration time
#   unknown  -> never registered; must still be rejected (open-set stays intact)
# Derived from the OSR cases: base reuses case-1's known set, its 6 unknowns are
# split 3 (register) / 3 (true unknown). Add more cases for the N3 sweep later.
# ---------------------------------------------------------------------------
REG_TEST_CASES = [
    {"base": [1, 4, 5, 7, 8, 11], "register": [0, 2, 3], "unknown": [6, 9, 10]},
    # case 1: register/unknown roles swapped (same base -> same encoder family);
    # tests the paper-3 conclusions' sensitivity to WHICH classes are registered.
    {"base": [1, 4, 5, 7, 8, 11], "register": [6, 9, 10], "unknown": [0, 2, 3]},
    # cases 2-5 (paper-4 final phase, design doc 2026-07-10 SS4.1): primary
    # vocab for each remaining OSR base split -- base = OSR known set, its
    # unknowns split in listed order: first 3 register / last 3 true-unknown.
    # Swapped-role twins become c6..c9 later and must REUSE these encoders
    # (SS4.5 cache-key migration); do not retrain per vocab.
    {"base": [1, 4, 5, 7, 10, 11], "register": [0, 2, 3], "unknown": [6, 8, 9]},
    {"base": [2, 5, 6, 7, 9, 11], "register": [0, 1, 3], "unknown": [4, 8, 10]},
    {"base": [0, 1, 7, 8, 10, 11], "register": [2, 3, 4], "unknown": [5, 6, 9]},
    {"base": [1, 2, 7, 8, 9, 10], "register": [0, 3, 4], "unknown": [5, 6, 11]},
    # case 6 = Gate 2 of the attribute-prior preregistration (2026-07-26): the
    # registered classes are ALL unseen attribute combinations and every parent
    # is in base, so the parallelogram prior is defined for all three of them.
    # Needs its own encoder family (new base split); do not reuse c0-c5 caches.
    {"base": [0, 1, 2, 3, 8, 11], "register": [4, 5, 6], "unknown": [7, 9, 10]},
]

# ---------------------------------------------------------------------------
# Attribute parents for the parallelogram prior (preregistration 2026-07-26).
#   target -> ([plus_a, plus_b], [minus...])
#   phi_hat(target) = mu(plus_a) + mu(plus_b) - mean_j mu(minus_j)
# Ids are ORIGINAL gesture ids (0..11); `_map_attr_spec` translates them per case
# and silently drops a target whose parents are not all in that case's base set.
# G5..G8 are the two-axis combinations of the four single-axis swipes G1..G4:
#   G5 = LR+UD, G6 = LR+DU, G7 = RL+UD, G8 = RL+DU
# ---------------------------------------------------------------------------
# Each target carries a LIST of candidate parent sets; the first one whose parents
# all live in the current case's base set is used, so one table serves both gates.
ATTR_PARENTS = {
    # G7 = RL+UD.  Gate 1 (REG case 1, base = [1,4,5,7,8,11]) reaches it as the
    # fourth corner of the G5/G6/G8 parallelogram; Gate 2 reaches it from the two
    # single-axis parents.
    6: [([4, 7], [5]),           # (LR+UD) + (RL+DU) - (LR+DU)
        ([1, 2], [0, 1, 2, 3])],  # (RL) + (UD) - mean(single-axis)
    # Gate 2 (REG case 6, base = [0,1,2,3,8,11]): the other two diagonals.
    4: [([0, 2], [0, 1, 2, 3])],  # LR+UD
    5: [([0, 3], [0, 1, 2, 3])],  # LR+DU
}


@dataclass
class Config:
    # --- data ---
    data_root: str = DATA_ROOT
    cache_dir: str = CACHE_DIR
    # Keep this visible in result logs: the paper reports on UWB-Gestures but
    # does not fully pin down preprocessing. Switching Raw/ClutterRemoved is a
    # high-impact diagnostic when the single-view OSR gap is investigated.
    data_type: str = "ClutterRemoved"          # ClutterRemoved | Raw
    views: List[str] = field(default_factory=lambda: list(VIEW_ORDER))
    T: int = 90                                 # slow-time frames per sample
    M: int = 189                               # fast-time / range bins
    # Per-sample, per-view z-score removes radar amplitude scale differences.
    # It is useful for multi-view stability, but it is another nontrivial
    # protocol choice versus the paper, so keep it easy to audit.
    # input normalisation. abs_std = |x| then per-sample per-view std-normalise,
    # matches Eom's "amplitude-based recognition" and gives the cleanest OSR
    # geometry; per_view_std is the old default (kept for ablation), none/abs
    # are diagnostics. CLI: --normalize {none|abs|per_view_std|abs_std}.
    normalize: str = "abs_std"
    # Coherent Doppler channel. The CSVs hold RF frames with the carrier intact
    # (fast-time spectrum peaks at 0.74 of Nyquist), so the analytic signal
    # carries phase, and the frame-to-frame phase increment is a SIGNED radial
    # velocity map on the same (T,M) grid. `normalize` throws that sign away
    # (abs_std takes |x| first), which is why direction of motion is unreadable
    # from the learned features. CLI: --doppler {off|only|concat}.
    #   off    -> unchanged amplitude pipeline
    #   only   -> the velocity map replaces the amplitude map (same shape)
    #   concat -> both, side by side on the range axis (M_in = 2M)
    doppler: str = "off"

    # --- encoder (Eom dual-path Transformer) ---
    dim: int = 32                              # D
    depth: int = 4                            # L transformer layers per path
    heads: int = 4                            # attention heads
    # Per-head key/query/value width (Eom D_key). 0 -> PyTorch default split
    # (dim/heads per head, i.e. 8). 32 -> paper-faithful custom MSA: each head
    # projects the 32-dim token to 32 dims, concat 4x32=128, project back to 32.
    # Token dim and feature-code size stay unchanged (unlike the old --dim 128
    # E3 proxy, which widened the tokens themselves to 128).
    d_key: int = 0
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    # Feature-code pooling:
    #   "mean"    -> 2*D (64), fast ablation with a much smaller inspector input.
    #   "flatten" -> D*(T+M) (8928), closest to Eom's feature code c in the paper.
    # Use --pool flatten for paper-facing single-view and multi-view comparisons.
    pool: str = "mean"

    # --- fusion (Method A: time-axis cross-view attention) ---
    use_fusion: bool = True
    use_physical_bias: bool = False           # Method B physical prior (v1 off)
    # F1: which per-view feature codes the Stage-2 inspectors consume.
    #   "pre"   -> straight from the shared encoder, before any cross-view
    #              exchange (default; byte-for-byte the old pipeline).
    #   "fused" -> per-view readout AFTER CrossViewFusion's time-axis attention
    #              (spatial path unchanged). Same dim, so the inspector stack is
    #              untouched; only its input features change. Tests whether
    #              exposing cross-view information to the OSR head helps.
    inspector_feats: str = "pre"               # pre | fused
    # F2: X-MV bottleneck mid-fusion (MBT-style, Nagrani et al. NeurIPS'21).
    # The last `xmv_layers` blocks of each TransformerPath exchange information
    # across views through `xmv_bottleneck` shared tokens: per view, tokens
    # attend over [own tokens ; bottleneck]; the per-view updated bottleneck
    # copies are then averaged, so the bottleneck is the only cross-view
    # channel. Per-view feature codes therefore become cross-view-informed
    # while staying view-specific (the FUSION_DESIGN.md SS3 trade-off).
    # 0 -> off, byte-for-byte the old per-view encoder.
    xmv_layers: int = 0
    xmv_bottleneck: int = 4

    # --- OSR head ---
    osr_head: str = "inspector"              # inspector | evidential | msp
    edl_kl_anneal_epochs: int = 10
    # deep-inspector head (paper-faithful one-vs-rest, Eom Stage-2)
    inspector_epochs: int = 40
    inspector_hidden: int = 1024
    inspector_latent: int = 128
    # Eom describes a plain DNN; 0.1 is the historical local default. Set 0 via
    # --insp-dropout for the paper-faithful (no-dropout) variant.
    inspector_dropout: float = 0.1
    n_pos_syn: int = 4000                     # Gaussian positives  (Eom Nps)
    n_neg_syn: int = 3000                     # manifold-mixup negatives (Eom Nns)
    # Stage-2 batch size is an experiment switch. Eom uses 16; the local 5-case
    # runs were stronger with 256, so the default stays empirical and
    # --batch-insp 16 is available for paper-faithful diagnostics.
    batch_size_insp: int = 256
    # Off by default because it lowered the local single-view 5-case result.
    # Enable --paper-neg-mixup to reproduce the Eq.12-style negative synthesis:
    # mix feature codes from any two different known gesture labels.
    paper_neg_mixup: bool = False
    osr_eta: float = 3.0                      # 3-sigma threshold scale
    osr_beta: float = 1.0                     # multi-view variance-signal weight
    # manifold-mixup synthetic negatives (Eom Eq.12): teach the head to reject
    # near-OOD by forcing max uncertainty on mixtures of different known classes.
    lambda_ood: float = 1.0
    ood_mix_low: float = 0.4
    ood_mix_high: float = 0.6
    # Gaussian-jitter synthetic positives (Eom Eq.13), optional
    lambda_pos: float = 0.0
    pos_jitter: float = 0.1

    # --- B2: cross-view masked-reconstruction pretraining (Stage-0, off by default) ---
    # Master switch. When False every B2 branch is skipped and the run is
    # byte-for-byte identical to the pre-B2 pipeline. When True a Stage-0 mask-one-
    # view reconstruction pretrain runs on train_ds (open-set clean: known-train
    # only) and its encoder weights initialise Stage-1.
    use_mae_pretrain: bool = False
    mae_epochs: int = 100
    # raw   -> decoder predicts the masked view's echo cube X[v*]  (B,T,M); grounded,
    #          no representation-collapse risk, most interpretable error.
    # token -> decoder predicts the masked view's temporal tokens (B,T,D); smoother
    #          but target comes from a stop-grad copy of the *training* encoder, so
    #          it carries a collapse caveat (no EMA target in v1).
    mae_target: str = "raw"                    # raw | token
    # Weight of the reconstruction-error OOD signal when fused into the inspector
    # anomaly score. Default 0 => recon error is reported as a STANDALONE head only
    # and never contaminates the main result. This mirrors osr_beta's lesson (the
    # cross-view-disagreement term hurt at beta>0), so fusion is opt-in and swept.
    lambda_recon_osr: float = 0.0

    # --- R1: missing-view robustness via training-time view dropout (off by default) ---
    # Probability that a training batch drops views (sensor-failure simulation).
    # When it fires, a random proper subset of 1..n_views-1 views is kept and the
    # model is trained to classify from that subset (view_emb sliced to match), so
    # at test time a dead sensor degrades gracefully instead of catastrophically.
    # 0.0 => disabled, run is byte-for-byte identical to the pre-R1 pipeline.
    view_dropout: float = 0.0

    # --- KD: cross-view self-distillation (off by default) ---
    # On a view-dropped batch, the full-view forward of the *same* model (no_grad)
    # is the teacher; the kept-subset forward is the student. The student's
    # closed-set logits are distilled toward the teacher's softened logits so a
    # degraded-view deployment mimics the strong full-view readout. Requires
    # view_dropout>0 (KD only fires on dropped batches). 0.0 => disabled,
    # byte-for-byte identical to the pre-KD pipeline.
    kd_distill: float = 0.0                    # logits-KL weight
    kd_temp: float = 2.0                       # softmax temperature T
    # delta term: match the student's evidential uncertainty to the teacher's so a
    # missing view does not inflate rejection. Experimental (acts on the evidential
    # head, secondary to the inspector); default off.
    kd_uncert: float = 0.0

    # --- auxiliary contrastive loss (optional, off by default) ---
    use_supcon: bool = False
    lambda_supcon: float = 0.5
    supcon_temp: float = 0.1

    # --- N4c (paper-3): subject-adversarial encoder training (off by default) ---
    # A gradient-reversal subject head on f_views scrubs "who is this" from the
    # features; targets the few-shot head's cross-subject distance inflation
    # (RESULTS SS33.3: subject shift = 2/3 of the new-class error). The reversed
    # gradient ramps DANN-style from 0 to this value over training. 0 = off.
    # SHELVED: collapses the representation at 0.5 AND 0.1 (RESULTS SS34.6).
    lambda_subj_adv: float = 0.0
    # --- N4b (paper-3): episodic (prototypical) auxiliary loss (off by default) ---
    # Within each batch, a random half of every class builds a prototype (mean
    # f_views per view) and the other half is classified among the in-batch
    # prototypes by -squared distance. Trains the exact deployment condition of
    # few-shot registration ("a mean of a few samples must classify unseen
    # samples") without the adversarial self-destruction mode. 0 = off.
    lambda_episodic: float = 0.0

    # --- training ---
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 150
    seed: int = 0
    device: str = "cuda"                       # auto-falls back to cpu
    num_workers: int = 0

    # --- evaluation ---
    # These counts implement the Table-I-style OSR protocol: per case, 6 known
    # and 6 unknown gestures, with 600 train + 200 known-test samples per known
    # gesture and 1200 unknown-test samples total.
    n_unknown_test: int = 1200                 # unknown samples drawn for test
    test_per_known: int = 200                  # test split per known gesture
    train_per_known: int = 600                 # train split per known gesture

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def M_in(self) -> int:
        """Range-axis width the encoder actually sees. Only `doppler=concat`
        widens it; M itself stays the physical bin count used by the cache."""
        return self.M * (2 if self.doppler == "concat" else 1)
