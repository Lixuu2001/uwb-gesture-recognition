"""Multi-View UWB Open-Set Gesture Recognition - entry point.

Examples
--------
  # smoke test on synthetic data (no real files, a few minutes on CPU)
  python main.py --smoke

  # build the cache once from the real CSVs, then run the 5 OSR test cases
  python main.py --build-cache
  python main.py                       # 3-view evidential (main result, B3)

  # baselines / ablations
  python main.py --single-view         # Eom single-view baseline (B0)
  python main.py --no-fusion           # drop cross-view attention
  python main.py --physical-bias       # add Method-B physical prior
  python main.py --supcon              # add supervised contrastive aux loss
  python main.py --case 0              # only run test case 1
"""
import argparse
import gc
import numpy as np
import torch

from config import Config, OSR_TEST_CASES, REG_TEST_CASES, VIEW_ORDER
from uwb_osr.data import (load_cache, build_cache, make_synthetic,
                          MVDataset, osr_split, loso_split, register_split,
                          register_split_conf)
from uwb_osr.engine import train_one_case, evaluate_osr, get_device
from uwb_osr.inspector import evaluate_inspector
from uwb_osr.model import MultiViewOSRNet, SharedDualPathEncoder
from uwb_osr.pretrain import (pretrain_encoder, recon_osr_heads, recon_error,
                              MaskedViewDecoder)
from uwb_osr.synth import evaluate_synth_views
from uwb_osr.registration import (evaluate_registration, register_and_score,
                                  conformal_report)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_case(cfg, X, y, split, device, verbose=True, tag="",
             ablate_views=False, ablate_score=False, eta_sweep=None,
             synth_view=False):
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    nrm = cfg.normalize
    train_ds = MVDataset(X, y, split["train_idx"], view_idx, split["label_map"], nrm)
    known_ds = MVDataset(X, y, split["test_known_idx"], view_idx, split["label_map"], nrm)
    unk_ds = MVDataset(X, y, split["test_unknown_idx"], view_idx, None, nrm)

    # B2 Stage-0: cross-view masked-reconstruction pretrain (gated). The trained
    # (enc0, dec0) stay frozen for the e_recon OOD signal; enc0_state initialises
    # the separately fine-tuned Stage-1 model so the two never alias.
    mae_state = enc0 = dec0 = None
    if cfg.use_mae_pretrain and cfg.n_views >= 2:
        mae_state, enc0, dec0 = pretrain_encoder(cfg, train_ds, device, verbose)

    model = train_one_case(cfg, train_ds, split["n_known"], device, verbose,
                           desc=f"{tag}Stage-1", mae_state=mae_state)
    if cfg.osr_head == "inspector":
        # E2 missing-view subsets (only when the run uses all 3 views)
        view_subsets = None
        if ablate_views and len(cfg.views) == 3:
            view_subsets = [
                ("all", [0, 1, 2]),                  # full multi-view (main)
                ("noL", [1, 2]),                      # drop Left  -> 2 views
                ("noT", [0, 2]),                      # drop Top
                ("noR", [0, 1]),                      # drop Right
                ("L",   [0]),                          # only Left  -> 1 view
                ("T",   [1]),
                ("R",   [2]),
            ]
        # E4 score-component ablation
        score_modes = ["combined", "mean_only", "var_only"] if ablate_score else None
        res = evaluate_inspector(cfg, model, train_ds, known_ds, unk_ds,
                                 split["n_known"], device,
                                 view_subsets=view_subsets,
                                 score_modes=score_modes,
                                 eta_sweep=eta_sweep,
                                 return_scores=(dec0 is not None))
        if dec0 is not None:                         # B2: add e_recon head(s)
            insp_scores = res.pop("_mean_scores", None)
            rheads, screen = recon_osr_heads(
                model, enc0, dec0, known_ds, unk_ds, split["n_known"], cfg,
                device, insp_scores=insp_scores)
            if verbose:
                print(f"  [B2] {screen}")
            res.update(rheads)
            if synth_view:                           # S4: virtual-view synthesis
                res.update(evaluate_synth_views(
                    cfg, model, enc0, dec0, train_ds, known_ds, unk_ds,
                    split["n_known"], device, verbose=verbose))
        else:
            res.pop("_mean_scores", None)
        return res
    return evaluate_osr(model, known_ds, unk_ds, cfg, split["n_known"], device)


def fmt(rows, key):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return "    -    "
    arr = np.array(vals) * 100
    return f"{arr.mean():5.2f} +/- {arr.std():4.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build-cache", action="store_true")
    # Single-view means Left radar only, matching the paper's single-radar setup.
    # The default keeps all three paired radar views for the multi-view extension.
    ap.add_argument("--single-view", action="store_true")
    ap.add_argument("--no-fusion", action="store_true")
    ap.add_argument("--physical-bias", action="store_true")
    ap.add_argument("--inspector-feats", choices=["pre", "fused"], default=None,
                    help=("F1: features the Stage-2 inspectors consume: pre "
                          "(default, before cross-view fusion; old pipeline) or "
                          "fused (after the cross-view temporal attention, same "
                          "dim). Tests exposing cross-view info to the OSR "
                          "head."))
    ap.add_argument("--xmv-layers", type=int, default=-1,
                    help=("F2: number of trailing encoder blocks per path doing "
                          "MBT-style cross-view bottleneck exchange (total "
                          "depth unchanged). 0 = off (default). Try 2."))
    ap.add_argument("--xmv-bottleneck", type=int, default=-1,
                    help="F2: number of shared bottleneck tokens NB (default 4).")
    ap.add_argument("--supcon", action="store_true")
    ap.add_argument("--osr-head", choices=["inspector", "evidential", "msp"],
                    default=None)
    ap.add_argument("--pool", choices=["mean", "flatten"], default=None,
                    help="feature code: mean is light; flatten is paper-facing")
    ap.add_argument("--batch-insp", type=int, default=-1,
                    help=("inspector Stage-2 batch size; Eom uses 16, "
                          "local default is in config.py"))
    ap.add_argument("--paper-neg-mixup", action="store_true",
                    help=("use Eq.12-style negatives mixed from all known labels "
                          "instead of the local neg-only default"))
    ap.add_argument("--normalize",
                    choices=["none", "abs", "per_view_std", "abs_std"],
                    default=None,
                    help=("input normalisation: none (raw signed), abs (|x|, "
                          "Eom 'amplitude'), per_view_std (our default), "
                          "abs_std (|x| then std-normalise)"))
    ap.add_argument("--ablate-views", action="store_true",
                    help=("E2: also evaluate on 6 view subsets (noL/noT/noR + "
                          "only L/T/R) to plot graceful degradation. Inspector "
                          "head only. Trained once with all 3 views, eval-time "
                          "subset switch."))
    ap.add_argument("--ablate-score", action="store_true",
                    help=("E4: also evaluate with mean_only and var_only "
                          "anomaly scores in addition to combined. Tests how "
                          "much the cross-view-disagreement (var) signal "
                          "contributes vs averaged distance (mean)."))
    ap.add_argument("--dim", type=int, default=-1,
                    help=("E3 (legacy proxy): encoder token dim. Default 32. "
                          "Widening this to 128 changes the tokens themselves; "
                          "for the paper's D_key=32 use --d-key 32 instead."))
    ap.add_argument("--d-key", type=int, default=-1,
                    help=("E3b: paper-faithful per-head key width (Eom D_key). "
                          "0/unset keeps PyTorch's dim/heads split (per-head 8);"
                          " 32 gives each head a 32-dim projection with token "
                          "dim and feature code unchanged."))
    ap.add_argument("--insp-dropout", type=float, default=-1.0,
                    help=("Stage-2 inspector dropout. Local default 0.1; the "
                          "paper describes a plain DNN, use 0 for that."))
    ap.add_argument("--heads", type=int, default=-1,
                    help="E3: encoder attention heads. Default 4.")
    ap.add_argument("--data-type", choices=["ClutterRemoved", "Raw"], default=None,
                    help=("D1: which preprocessed data to load. ClutterRemoved is "
                          "the default; Raw is the no-preprocess baseline."))
    ap.add_argument("--eta-sweep", action="store_true",
                    help=("A7: also report acc/f1 at fixed eta thresholds "
                          "(eta in {1.5,2,2.5,3,3.5,4,5,7}), matching Eom Fig 10 "
                          "heuristic 3-sigma rule. Eval-only, no retrain."))
    ap.add_argument("--mae-pretrain", action="store_true",
                    help=("B2: run Stage-0 cross-view masked-reconstruction "
                          "pretraining (mask 1 view) before Stage-1, and report "
                          "the e_recon OOD head. Inspector head, >=2 views."))
    ap.add_argument("--mae-target", choices=["raw", "token"], default=None,
                    help="B2: reconstruct the raw echo cube (default) or tokens.")
    ap.add_argument("--lambda-recon", type=float, default=-1.0,
                    help=("B2: weight to fuse e_recon into the inspector "
                          "mean_only score (z-normalised, also swept). 0 keeps "
                          "e_recon standalone. Default from config (0)."))
    ap.add_argument("--synth-view", action="store_true",
                    help=("S4: also evaluate virtual-view synthesis. For each "
                          "view, replace the real view with the B2 decoder's "
                          "synthesis (other two views) and score open-set; "
                          "reports synth_real3 / synth_<V> / synth_drop<V>. "
                          "Needs --mae-pretrain and all 3 views."))
    ap.add_argument("--view-dropout", type=float, default=-1.0,
                    help=("R1: probability a training batch drops views "
                          "(sensor-failure sim); keeps a random 1..V-1 subset so "
                          "missing views degrade gracefully. Default from config "
                          "(0). Pair with --ablate-views to read the matrix."))
    ap.add_argument("--kd-distill", type=float, default=-1.0,
                    help=("KD: weight for distilling the view-dropped student "
                          "toward the full-view self-teacher's softened logits. "
                          "Only fires on dropped batches, so REQUIRES "
                          "--view-dropout>0. Default from config (0)."))
    ap.add_argument("--kd-temp", type=float, default=-1.0,
                    help="KD: softmax temperature T (default from config, 2.0).")
    ap.add_argument("--kd-uncert", type=float, default=-1.0,
                    help=("KD: weight to match student/teacher evidential "
                          "uncertainty (open-set score distillation, "
                          "experimental). Default from config (0)."))
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help=("force the compute device. cpu lets smoke tests run "
                          "alongside a busy GPU. Default from config (cuda, "
                          "auto-falls back to cpu)."))
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--register", action="store_true",
                    help=("N1 (paper-3): few-shot new-gesture registration eval. "
                          "Trains the encoder on REG_TEST_CASES base classes, "
                          "registers the register classes from --n-shot samples, "
                          "then classifies the held-out subject by nearest "
                          "prototype across base+registered. Always LOSO."))
    ap.add_argument("--n-shot", type=int, default=5,
                    help="N1: support samples per registered class (default 5).")
    ap.add_argument("--reg-method", choices=["A", "B", "both"], default="both",
                    help=("N1: A=trained few-shot inspector (same family as base "
                          "heads), B=NCM mean prototype in frozen feature space "
                          "(more robust at tiny shot counts), both=evaluate A and "
                          "B on the same per-fold encoder (default, paired)."))
    ap.add_argument("--conformal", action="store_true",
                    help=("N2 (paper-3): replace the fixed eta=3 accept/reject "
                          "rule with a split-conformal threshold (held-out base "
                          "calibration). Reports coverage vs 1-alpha, prediction-"
                          "set size, unknown empty-set (reject) rate and singleton "
                          "accuracy for mondrian/global/eta modes. Dumps per-fold "
                          "score matrices to logs/n2_scores_*.npz for post-hoc."))
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="N2: conformal miscoverage target (default 0.1 => 90%%).")
    ap.add_argument("--calib-frac", type=float, default=0.25,
                    help="N2: fraction of base-train held out for calibration.")
    ap.add_argument("--support-balanced", action="store_true",
                    help=("N3 P1.5: draw each new-class support shot from a "
                          "different subject (cross-person spread) instead of "
                          "random, to de-bias the few-shot sigma."))
    ap.add_argument("--support-personal", action="store_true",
                    help=("N4 diagnostic: draw the new-class support shots from "
                          "the TEST subject (excluded from the test set) -- the "
                          "personalization scenario ('register on your own "
                          "device'). Decomposes the new-class error into subject "
                          "shift vs class hardness. Overrides --support-balanced."))
    ap.add_argument("--subj-adv", type=float, default=-1.0,
                    help=("N4c: subject-adversarial encoder training. Value = "
                          "max reversed-gradient weight (DANN ramp 0->value); "
                          "scrubs subject identity from f_views to shrink the "
                          "few-shot head's cross-subject distance inflation. "
                          "0/unset = off. SHELVED: collapses at 0.5 AND 0.1."))
    ap.add_argument("--episodic", type=float, default=-1.0,
                    help=("N4b: weight of the in-batch prototypical episode "
                          "loss (support-half prototype classifies query-half "
                          "by -distance) -- trains the few-shot deployment "
                          "condition directly. 0/unset = off."))
    ap.add_argument("--dump-tag", default="",
                    help=("suffix inserted into the per-fold npz dump names, e.g. "
                          "'b2' -> n3_scores_bal_b2_c0_s*.npz. REQUIRED for "
                          "encoder-variant runs (N4a/b/c) so they do not "
                          "overwrite the baseline dumps."))
    ap.add_argument("--enc-cache", action="store_true",
                    help=("P3: cache the trained fold encoder (and the Stage-0 "
                          "enc/dec when --mae-pretrain) under cache/enc/, keyed "
                          "by case/test-subject/calib-subject/encoder config. "
                          "A later run with the same key skips Stage-0+Stage-1 "
                          "entirely, turning Stage-2 ablations (shots, delta, "
                          "sim-task) into minute-scale jobs."))
    ap.add_argument("--delta-aug", action="store_true",
                    help=("P3: subject-delta support augmentation (method A). "
                          "Base classes give per-subject centroid offsets; each "
                          "support shot is translated to every other training "
                          "subject before the few-shot inspector trains, so "
                          "prototype/sigma see cross-person spread (the "
                          "balanced-support mechanism, amplified, encoder "
                          "untouched)."))
    ap.add_argument("--sim-task", action="store_true",
                    help=("P3: simulated-task conformal raw material. Per fold, "
                          "rehearse the few-shot registration on each base class "
                          "(subject-balanced shots, other classes as negatives) "
                          "and dump the calib subject's own-r distribution -> a "
                          "few-shot-specific threshold for the new-class columns "
                          "without any new-class calibration data."))
    ap.add_argument("--calib-rotate", action="store_true",
                    help=("P3: calibration subject = the next subject after the "
                          "test subject in cyclic order (default: always the "
                          "first non-test subject, which makes nearly every fold "
                          "calibrate on subj1 and produced a degenerate "
                          "accept-everything fold)."))
    ap.add_argument("--n-calib", type=int, default=1,
                    help=("P3 SS38: number of held-out calibration subjects "
                          "pooled per fold (leave-(1+n)-out). One subject is a "
                          "scale lottery (q90 varies 15x with who calibrates, "
                          "per-fold coverage swings 61-100); pooling several "
                          "averages the scale variance away, at the cost of one "
                          "encoder training subject each."))
    ap.add_argument("--mae-no-init", action="store_true",
                    help=("P3 attribution variant: run the Stage-0 pretrain (so "
                          "e_recon exists) but do NOT init Stage-1 from it -- "
                          "separates 'fusion helps' from 'B2 init helps'."))
    ap.add_argument("--mmap", action="store_true",
                    help=("memory-map the data cube instead of loading it to RAM "
                          "(~2 GB saved); use on memory-tight machines to avoid "
                          "the commit-charge spike that kills runs."))
    ap.add_argument("--case", type=int, default=-1, help="run a single case 0..4")
    ap.add_argument("--max-folds", type=int, default=-1,
                    help=("cap the number of LOSO folds (subjects) run; -1 = all. "
                          "Use for a smoke (1) or direction (3) run before the full 8."))
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(seed=args.seed)
    if args.single_view:
        cfg.views = ["Left"]
    if args.no_fusion:
        cfg.use_fusion = False
    if args.physical_bias:
        cfg.use_physical_bias = True
    if args.inspector_feats:
        cfg.inspector_feats = args.inspector_feats
    if args.xmv_layers >= 0:
        cfg.xmv_layers = args.xmv_layers
    if args.xmv_bottleneck > 0:
        cfg.xmv_bottleneck = args.xmv_bottleneck
    if args.supcon:
        cfg.use_supcon = True
    if args.osr_head:
        cfg.osr_head = args.osr_head
    if args.pool:
        cfg.pool = args.pool
    if args.batch_insp > 0:
        cfg.batch_size_insp = args.batch_insp
    if args.paper_neg_mixup:
        cfg.paper_neg_mixup = True
    if args.normalize:
        cfg.normalize = args.normalize
    if args.epochs > 0:
        cfg.epochs = args.epochs
    if args.dim > 0:
        cfg.dim = args.dim
    if args.heads > 0:
        cfg.heads = args.heads
    if args.d_key >= 0:
        cfg.d_key = args.d_key
    if args.insp_dropout >= 0:
        cfg.inspector_dropout = args.insp_dropout
    if args.data_type:
        cfg.data_type = args.data_type
    if args.mae_pretrain:
        cfg.use_mae_pretrain = True
    if args.mae_target:
        cfg.mae_target = args.mae_target
    if args.lambda_recon >= 0:
        cfg.lambda_recon_osr = args.lambda_recon
    if args.subj_adv >= 0:
        cfg.lambda_subj_adv = args.subj_adv
    if args.episodic >= 0:
        cfg.lambda_episodic = args.episodic
    if args.view_dropout >= 0:
        cfg.view_dropout = args.view_dropout
    if args.kd_distill >= 0:
        cfg.kd_distill = args.kd_distill
    if args.kd_temp >= 0:
        cfg.kd_temp = args.kd_temp
    if args.kd_uncert >= 0:
        cfg.kd_uncert = args.kd_uncert
    if args.device:
        cfg.device = args.device
    eta_sweep_list = ([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]
                      if args.eta_sweep else None)

    set_seed(cfg.seed)
    device = get_device(cfg)
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}  (CUDA enabled)")
    else:
        print("[!! CPU !!] CUDA not available. In VSCode pick the 'pytorch' conda "
              "interpreter: Ctrl+Shift+P > Python: Select Interpreter.")
    print(f"== MV-OSR ==  osr_head={cfg.osr_head}  pool={cfg.pool}  "
          f"normalize={cfg.normalize}  "
          f"views={cfg.views}  fusion={cfg.use_fusion}  "
          f"phys_bias={cfg.use_physical_bias}  epochs={cfg.epochs}  "
          f"batch_insp={cfg.batch_size_insp}  "
          f"paper_neg_mixup={cfg.paper_neg_mixup}  "
          f"dim={cfg.dim}  heads={cfg.heads}"
          + (f"  d_key={cfg.d_key}" if cfg.d_key > 0 else "")
          + (f"  insp_dropout={cfg.inspector_dropout}"
             if cfg.inspector_dropout != 0.1 else "")
          + f"  data_type={cfg.data_type}  "
          f"eta_sweep={'yes' if eta_sweep_list else 'no'}  "
          f"mae_pretrain={cfg.use_mae_pretrain}"
          + (f"(target={cfg.mae_target},lambda={cfg.lambda_recon_osr})"
             if cfg.use_mae_pretrain else "")
          + (f"  insp_feats={cfg.inspector_feats}"
             if cfg.inspector_feats != "pre" else "")
          + (f"  xmv={cfg.xmv_layers}Lx{cfg.xmv_bottleneck}B"
             if cfg.xmv_layers > 0 else "")
          + (f"  view_dropout={cfg.view_dropout}" if cfg.view_dropout > 0 else "")
          + (f"  kd={cfg.kd_distill}/T{cfg.kd_temp}"
             + (f"+u{cfg.kd_uncert}" if cfg.kd_uncert > 0 else "")
             if cfg.kd_distill > 0 else ""))

    if args.build_cache:
        build_cache(cfg, force=True)
        print("cache built; rerun without --build-cache to train.")
        return

    # ---- data ----
    if args.smoke:
        cfg.train_per_known, cfg.test_per_known = 50, 20
        cfg.n_unknown_test = 100
        if args.epochs <= 0:
            cfg.epochs = 3
        cfg.mae_epochs = 2                          # keep Stage-0 quick in smoke
        if args.register or args.conformal:         # keep N1/N2 smoke quick on CPU
            cfg.inspector_epochs = 5
            cfg.n_pos_syn, cfg.n_neg_syn = 400, 300
        X, y, subj = make_synthetic(cfg, n_per_class=80, seed=cfg.seed)
        print(f"[smoke] synthetic X{X.shape}")
    else:
        # mmap keeps the ~2 GB cube on disk instead of RAM; essential on a
        # memory-tight box (the commit spike otherwise kills the run). Default
        # loads to RAM to keep the GPU fed when memory allows.
        X, y, subj = load_cache(cfg, mmap=args.mmap)
        print(f"[data] X{X.shape} y{y.shape} {'mmap' if args.mmap else 'loaded to RAM'}")

    if args.conformal:
        run_conformal(cfg, X, y, subj, device, args)
        return

    if args.register:
        run_registration(cfg, X, y, subj, device, args)
        return

    cases = [args.case] if args.case >= 0 else list(range(len(OSR_TEST_CASES)))
    head_rows = {}
    for ci in cases:
        case = OSR_TEST_CASES[ci]
        print(f"\n--- Test case {ci+1}: known={case['known']} "
              f"unknown={case['unknown']} ---")
        if args.loso:
            subj_res = []
            for ts in sorted(set(subj.tolist())):
                sp = loso_split(cfg, y, subj, case["known"], case["unknown"], ts, cfg.seed)
                subj_res.append(run_case(cfg, X, y, sp, device, verbose=False,
                                         ablate_views=args.ablate_views,
                                         ablate_score=args.ablate_score,
                                         eta_sweep=eta_sweep_list,
                                         synth_view=args.synth_view))
                # pick any inspector-like head for the LOSO running line
                _pkey = next((k for k in subj_res[-1]
                              if k != "closed_acc" and not k.startswith("msp")),
                             None)
                if _pkey:
                    print(f"  LOSO subj {ts}: OSCR="
                          f"{subj_res[-1][_pkey]['oscr']*100:.2f}")
            res = _avg_results(subj_res)
        else:
            sp = osr_split(cfg, y, case["known"], case["unknown"], cfg.seed)
            res = run_case(cfg, X, y, sp, device, tag=f"case{ci+1} ",
                           ablate_views=args.ablate_views,
                           ablate_score=args.ablate_score,
                           eta_sweep=eta_sweep_list,
                           synth_view=args.synth_view)
        for hk, hv in res.items():
            if hk not in ("closed_acc", "eta_sweep"):
                head_rows.setdefault(hk, []).append(hv)
        # A7 separate bucket: per-eta acc/f1 records (one entry per case)
        if "eta_sweep" in res:
            for eta, vals in res["eta_sweep"].items():
                head_rows.setdefault(f"eta={eta}", []).append(vals)
        # per-case progress line: prefer "inspector"/"insp_all_combined" headline,
        # fall back to first non-msp head if ablations renamed everything.
        prim_keys = [k for k in res if k != "closed_acc" and not k.startswith("msp")]
        prim = ("inspector" if "inspector" in res
                else "insp_all_combined" if "insp_all_combined" in res
                else "insp_all" if "insp_all" in res
                else "insp_combined" if "insp_combined" in res
                else (prim_keys[0] if prim_keys else None))
        msp_key = ("msp" if "msp" in res
                   else "msp_all" if "msp_all" in res
                   else next((k for k in res if k.startswith("msp")), None))
        if prim:
            msp_str = f" | MSP oscr={res[msp_key]['oscr']*100:.2f}" if msp_key else ""
            print(f"  closed_acc={res['closed_acc']*100:.2f} | "
                  f"{prim.upper()} oscr={res[prim]['oscr']*100:.2f} "
                  f"auroc={res[prim]['auroc']*100:.2f}{msp_str}")

    print("\n================ SUMMARY (mean +/- std over cases, %) ============")
    print(f"{'head':16s} {'Accuracy':>14s} {'F1':>14s} {'AUROC':>14s} {'OSCR':>14s}")
    eta_rows = {k: v for k, v in head_rows.items() if k.startswith("eta=")}
    main_rows = {k: v for k, v in head_rows.items() if not k.startswith("eta=")}
    for name, rows in main_rows.items():
        print(f"{name:16s} {fmt(rows,'acc'):>14s} {fmt(rows,'f1'):>14s} "
              f"{fmt(rows,'auroc'):>14s} {fmt(rows,'oscr'):>14s}")
    if eta_rows:
        print("\n--- A7 eta-sweep (Eom Fig 10 heuristic threshold) ---")
        print(f"{'eta':16s} {'Accuracy':>14s} {'F1':>14s}")
        for name, rows in eta_rows.items():
            print(f"{name:16s} {fmt(rows,'acc'):>14s} {fmt(rows,'f1'):>14s}")


def _enc_cache_path(cfg, args, ci, ts, cs):
    """Cache key for one fold's trained encoder (P3 --enc-cache).

    Everything that changes the Stage-1 weights (data variant, normalisation,
    architecture, epochs, Stage-0 init, auxiliary losses, seed, fold identity)
    is in the name; Stage-2 knobs (shots, support mode, delta/sim/conformal
    settings) deliberately are NOT, so every Stage-2 ablation reuses the same
    encoder and pays minutes instead of the ~30-60 min fold retrain."""
    import os
    tag = (f"c{ci}_ts{ts}_cs{cs}_ep{cfg.epochs}_f{int(cfg.use_fusion)}"
           f"_{cfg.data_type}_{cfg.normalize}_{cfg.pool}"
           f"_d{cfg.dim}x{cfg.depth}h{cfg.heads}k{cfg.d_key}"
           f"_v{len(cfg.views)}_s{cfg.seed}")
    if cfg.use_mae_pretrain:
        tag += f"_mae-{cfg.mae_target}{cfg.mae_epochs}"
        if args.mae_no_init:
            tag += "-noinit"
    if cfg.lambda_subj_adv > 0:
        tag += f"_adv{cfg.lambda_subj_adv}"
    if cfg.lambda_episodic > 0:
        tag += f"_epi{cfg.lambda_episodic}"
    return os.path.join(cfg.cache_dir, "enc", tag + ".pt")


def run_conformal(cfg, X, y, subj, device, args):
    """N2 (paper-3): conformal accept/reject calibration on the N1 registration
    system. Per fold: train the encoder on base classes, register the new classes
    few-shot, score by nearest prototype with a held-out calibration split, then
    build prediction sets under conformal (mondrian/global) and the fixed eta=3
    baseline. Reports coverage vs 1-alpha, set size, unknown reject rate and
    singleton accuracy; dumps per-fold score matrices for post-hoc re-analysis."""
    import os
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    reg_cases = [args.case] if args.case >= 0 else list(range(len(REG_TEST_CASES)))
    subjects = sorted(set(subj.tolist()))
    # cap only how many test folds we run; the calibration subject `cs` is still
    # drawn from the full subject pool (leave-2-out needs >=2 subjects available).
    fold_subjects = subjects[:args.max_folds] if args.max_folds > 0 else subjects
    alpha = args.alpha
    modes = ["hybrid", "mondrian", "global", "eta"]
    eta = float(getattr(cfg, "osr_eta", 3.0))
    sup_mode = ("personal(test-subject)" if args.support_personal
                else "subject-balanced" if args.support_balanced else "random")
    adv_str = (f"  subj_adv={cfg.lambda_subj_adv}" if cfg.lambda_subj_adv > 0
               else "")
    adv_str += (f"  episodic={cfg.lambda_episodic}" if cfg.lambda_episodic > 0
                else "")
    p3_str = "".join(s for s, on in [
        ("  delta_aug", args.delta_aug), ("  sim_task", args.sim_task),
        ("  enc_cache", args.enc_cache), ("  calib_rotate", args.calib_rotate),
        ("  mae_no_init", args.mae_no_init)] if on)
    print(f"== Conformal registration (N2/N3) ==  n_shot={args.n_shot} "
          f"method={args.reg_method if args.reg_method != 'both' else 'A'} "
          f"alpha={alpha} (target cov {100*(1-alpha):.0f}%)  "
          f"calib={args.n_calib} held-out subject(s) "
          f"(leave-{1+args.n_calib}-out)  "
          f"support={sup_mode}{adv_str}{p3_str}  "
          f"views={cfg.views}  folds={len(fold_subjects)}/{len(subjects)}")
    method = "A" if args.reg_method == "both" else args.reg_method
    os.makedirs("logs", exist_ok=True)
    summary = {}                                     # (case, mode) -> [fold dicts]
    for ci in reg_cases:
        rc = REG_TEST_CASES[ci]
        print(f"\n--- Reg case {ci+1}: base={rc['base']} register={rc['register']} "
              f"unknown={rc['unknown']} ---")
        for ts in fold_subjects:
            # leave-(1+n_calib)-out: the calib subject(s) are other unseen
            # users, so calib and test are exchangeable -> valid coverage.
            # --calib-rotate walks the pairing around the roster; the default
            # (first non-test subject) calibrates nearly every fold on subj1,
            # which produced a degenerate accept-everything fold (SS38).
            if args.calib_rotate:
                i0 = subjects.index(ts)
                order = [subjects[(i0 + j) % len(subjects)]
                         for j in range(1, len(subjects))]
            else:
                order = [s for s in subjects if s != ts]
            cs_list = order[:max(1, args.n_calib)]
            cs = cs_list[0] if len(cs_list) == 1 else cs_list
            cs_tag = "+".join(str(s) for s in cs_list)
            sp = register_split_conf(cfg, y, subj, rc["base"], rc["register"],
                                     rc["unknown"], ts, cs, args.n_shot, cfg.seed,
                                     subject_balanced=args.support_balanced,
                                     support_personal=args.support_personal)
            # N4c: subject ids for the adversarial head (encoder subjects only,
            # remapped to 0..n_subj-1; -1 marks rows the train split never sees).
            n_subj, subj_arr = 0, None
            if cfg.lambda_subj_adv > 0:
                enc_subjs = sorted(set(subj[sp["train_idx"]].tolist()))
                smap = {s0: i for i, s0 in enumerate(enc_subjs)}
                subj_arr = np.array([smap.get(int(s0), -1) for s0 in subj])
                n_subj = len(enc_subjs)
            train_ds = MVDataset(X, y, sp["train_idx"], view_idx,
                                 sp["label_map"], cfg.normalize,
                                 subj_labels=subj_arr)
            # Fold encoder: load from cache when available, else train
            # (optionally after the B2 Stage-0) and cache. Unlike the N4a-era
            # code, enc0/dec0 are KEPT when mae is on: the frozen Stage-0 pair
            # produces the per-sample e_recon dumped below (P3 fusion).
            cache_path = (_enc_cache_path(cfg, args, ci, ts, cs_tag)
                          if args.enc_cache else None)
            mae_state, enc0, dec0 = None, None, None
            if cache_path and os.path.exists(cache_path):
                ck = torch.load(cache_path, map_location=device)
                model = MultiViewOSRNet(cfg, sp["n_base"]).to(device)
                model.load_state_dict(ck["model"])
                model.eval()
                if ck.get("mae_enc") is not None:
                    enc0 = SharedDualPathEncoder(cfg).to(device)
                    enc0.load_state_dict(ck["mae_enc"])
                    dec0 = MaskedViewDecoder(cfg).to(device)
                    dec0.load_state_dict(ck["mae_dec"])
                print(f"  subj {ts}: [enc-cache hit] "
                      f"{os.path.basename(cache_path)}", flush=True)
            else:
                if cfg.use_mae_pretrain and cfg.n_views >= 2:
                    mae_state, enc0, dec0 = pretrain_encoder(cfg, train_ds,
                                                             device, False)
                model = train_one_case(
                    cfg, train_ds, sp["n_base"], device, verbose=False,
                    desc=f"conf c{ci+1} s{ts} ",
                    mae_state=(None if args.mae_no_init else mae_state),
                    n_subj=n_subj)
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), "mae_enc": mae_state,
                         "mae_dec": (dec0.state_dict()
                                     if dec0 is not None else None)},
                        cache_path)
            rd = register_and_score(cfg, model, sp, X, y, view_idx, device,
                                    method=method, calib_frac=args.calib_frac,
                                    seed=cfg.seed, subj=subj,
                                    delta_aug=args.delta_aug,
                                    sim_task=args.sim_task, n_shot=args.n_shot)
            # P3: per-sample cross-view reconstruction error for every scored
            # set. e_recon is class-agnostic, so fusing it into the
            # non-conformity is a pure post-hoc lambda sweep on the dump
            # (tools/p3_analyze.py) -- no retraining.
            if dec0 is not None:
                def _er(idx):
                    if len(idx) == 0:
                        return np.zeros(0, np.float32)
                    return recon_error(
                        enc0, dec0,
                        MVDataset(X, y, idx, view_idx, None, cfg.normalize),
                        cfg, device)
                rd["E_cal"] = _er(sp["calib_idx"])
                rd["E_cal_reg"] = _er(sp.get("calib_reg_idx",
                                             np.array([], int)))
                rd["E_base"] = _er(sp["test_base_idx"])
                rd["E_reg"] = _er(sp["test_reg_idx"])
                rd["E_unk"] = _er(sp["test_unknown_idx"])
            sup = ("pers" if args.support_personal
                   else "bal" if args.support_balanced else "rand")
            if args.dump_tag:
                sup = f"{sup}_{args.dump_tag}"
            np.savez(os.path.join("logs", f"n3_scores_{sup}_c{ci}_s{ts}.npz"), **rd)
            line = f"  subj {ts}:"
            for m in modes:
                rep = conformal_report(rd, alpha, m, eta)
                summary.setdefault((ci, m), []).append(rep)
                line += (f"  [{m}] cov_b={rep['cov_base']*100:4.0f} "
                         f"cov_n={rep['cov_new']*100:4.0f} "
                         f"sz={rep['size_base']:.2f}/{rep['size_new']:.2f} "
                         f"rejU={rep['reject_unknown']*100:4.0f}")
            print(line)
            # free the fold's model before the next fold allocates a new one;
            # on a memory-tight box the brief double-allocation otherwise spikes
            # the commit charge and the process dies (no traceback). See §32 log.
            del model, rd, train_ds, mae_state, enc0, dec0
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n===== N2 CONFORMAL SUMMARY (mean over folds, %; target cov "
          f"{100*(1-alpha):.0f}) =====")
    print(f"{'mode':10s} {'cov_base':>9s} {'cov_new':>9s} {'size_b':>7s} "
          f"{'size_n':>7s} {'empty_n':>8s} {'rejU':>7s} {'singAcc_b':>10s} "
          f"{'singAcc_n':>10s}")
    for (ci, m), fr in summary.items():
        def mv(k):
            return float(np.mean([r[k] for r in fr])) * 100
        print(f"c{ci+1}/{m:<8s} {mv('cov_base'):9.1f} {mv('cov_new'):9.1f} "
              f"{mv('size_base')/100:7.2f} {mv('size_new')/100:7.2f} "
              f"{mv('empty_new'):8.1f} {mv('reject_unknown'):7.1f} "
              f"{mv('sing_acc_base'):10.1f} {mv('sing_acc_new'):10.1f}")


def run_registration(cfg, X, y, subj, device, args):
    """N1 (paper-3): few-shot registration eval, LOSO over subjects.

    Per case: train the encoder on base classes (training subjects), register the
    new classes from --n-shot samples, classify the held-out subject by nearest
    prototype across base+registered, and report base/new recognition, their
    harmonic mean, open-set AUROC/OSCR, plus an operating point that also rejects
    true unknowns."""
    view_idx = [VIEW_ORDER.index(v) for v in cfg.views]
    reg_cases = [args.case] if args.case >= 0 else list(range(len(REG_TEST_CASES)))
    methods = ["A", "B"] if args.reg_method == "both" else [args.reg_method]
    subjects = sorted(set(subj.tolist()))
    print(f"== Registration (N1) ==  n_shot={args.n_shot}  methods={methods}"
          f"  views={cfg.views}  subjects={len(subjects)}")
    summary = {}                                     # (case, method) -> [fold dicts]
    for ci in reg_cases:
        rc = REG_TEST_CASES[ci]
        print(f"\n--- Reg case {ci+1}: base={rc['base']} register={rc['register']} "
              f"unknown={rc['unknown']} ---")
        for ts in subjects:
            sp = register_split(cfg, y, subj, rc["base"], rc["register"],
                                rc["unknown"], ts, args.n_shot, cfg.seed)
            train_ds = MVDataset(X, y, sp["train_idx"], view_idx,
                                 sp["label_map"], cfg.normalize)
            # one encoder per fold serves every method (base-only training is
            # method-independent), so A vs B is paired on the same backbone.
            model = train_one_case(cfg, train_ds, sp["n_base"], device,
                                   verbose=False, desc=f"reg c{ci+1} s{ts} ")
            for m in methods:
                res = evaluate_registration(cfg, model, sp, X, y, view_idx,
                                            device, method=m)
                summary.setdefault((ci, m), []).append(res)
                print(f"  subj {ts} [{m}]: acc_base={res['acc_base']*100:5.1f} "
                      f"acc_new={res['acc_new']*100:5.1f} H={res['H']*100:5.1f} | "
                      f"AUROC={res['auroc']*100:5.1f} OSCR={res['oscr']*100:5.1f} | "
                      f"@tau base={res['acc_base_op']*100:5.1f} "
                      f"new={res['acc_new_op']*100:5.1f} "
                      f"rejU={res['rej_unknown']*100:5.1f}")

    print("\n========== REG SUMMARY (mean +/- std over folds, %) ==========")
    print(f"{'case/m':8s} {'acc_base':>13s} {'acc_new':>13s} {'H':>8s} "
          f"{'AUROC':>8s} {'OSCR':>8s} | {'op_base':>8s} {'op_new':>8s} "
          f"{'rejU':>8s} {'op_H':>8s}")
    for (ci, m), fr in summary.items():
        def ms(k):
            a = np.array([r[k] for r in fr]) * 100
            return a.mean(), a.std()
        ab, abs_ = ms("acc_base"); an, ans = ms("acc_new")
        print(f"c{ci+1}/{m:<4s} {ab:6.2f}+/-{abs_:4.2f} {an:6.2f}+/-{ans:4.2f} "
              f"{ms('H')[0]:8.2f} {ms('auroc')[0]:8.2f} {ms('oscr')[0]:8.2f} | "
              f"{ms('acc_base_op')[0]:8.2f} {ms('acc_new_op')[0]:8.2f} "
              f"{ms('rej_unknown')[0]:8.2f} {ms('H_op')[0]:8.2f}")


def _avg_results(subj_res):
    heads = [k for k in subj_res[0] if k != "closed_acc"]
    out = {"closed_acc": float(np.mean([r["closed_acc"] for r in subj_res]))}
    for head in heads:
        out[head] = {k: float(np.mean([r[head][k] for r in subj_res]))
                     for k in subj_res[0][head]}
    return out


if __name__ == "__main__":
    main()
