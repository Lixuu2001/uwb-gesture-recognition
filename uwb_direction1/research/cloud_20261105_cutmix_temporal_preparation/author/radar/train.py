"""Unified training script for sign language recognition.

Supports CE, CE+Triplet, CE+SupCon loss combinations via config.
All modes are deterministic by default (bitwise reproducible across runs).

Config flags control behavior:
  triplet_lambda: 0     -> pure CrossEntropy (default)
  triplet_lambda: 0.2   -> CE + Triplet auxiliary loss
  supcon_lambda: 0.1    -> CE + Triplet + Supervised Contrastive
  pk_sampling: true     -> PK batch sampling (P classes x K samples)
  sampling: reinforce   -> confusion-aware oversampling

Usage:
  python codes/final/train.py configs/exp84_b1_winner_13a.yaml
  python codes/final/train.py configs/r7_b1_trip.yaml --fold 0 --device cuda:2
"""

import argparse
import json
import shutil
import sys
import time as _time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import tqdm

from src.data.dataset import build_dataloaders, _make_worker_init_fn
from src.data.dataset_hdf5 import build_dataloaders_hdf5
from src.losses.triplet import (
    online_hard_triplet_loss, confusion_weighted_triplet_loss, PKSampler,
)
from src.models.classifier import build_model
from src.paths import EXPERIMENTS_DIR, PROJECT_ROOT
from src.training_utils import (
    set_seed, build_scheduler, validate_config, get_git_hash,
)


def load_class_mapping():
    """Load class ID -> name mapping from data/class_mapping.json."""
    mapping_path = PROJECT_ROOT / "data" / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            m = json.load(f)
        return {int(k): v for k, v in m["id_to_name"].items()}
    return {}


# -- Unified forward pass ---------------------------------------------------

def forward_with_features(model, x, mask=None):
    """Single forward pass returning both logits and L2-normalized features.

    Avoids the redundant pattern of model(x) + get_backbone_features(x)
    by doing forward_features -> pool -> classifier in ONE pass.
    Returns (logits, L2-normalized features) with gradients flowing to both.

    Uses timm's forward_head(pre_logits=True) for pooled features,
    which works for all architectures (EfficientNet, RegNet, MobileOne, etc.).
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model

    # Temporal models have their own forward_with_features
    if hasattr(raw, 'forward_with_features'):
        return raw.forward_with_features(x, mask=mask)

    # CNN backbone forward (the expensive part — done ONCE)
    feats = raw.forward_features(x)

    # Pooled features (pre-classifier) via timm's forward_head
    pooled = raw.forward_head(feats, pre_logits=True)

    # Full logits (includes dropout + classifier)
    logits = raw.forward_head(feats)

    # L2-normalized features for metric losses
    features = F.normalize(pooled, p=2, dim=1)

    return logits, features


# -- UMAP feature extraction ------------------------------------------------

def _ce_feature_fn(model, x):
    """Extract backbone features from CE classifier model."""
    raw = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(raw, 'forward_with_features') and not hasattr(raw, 'forward_features'):
        _, feats = raw.forward_with_features(x)
        return feats
    feats = raw.forward_features(x)
    return raw.forward_head(feats, pre_logits=True)


# -- Batch unpacking --------------------------------------------------------

def _unpack_batch(batch, device, channels_last=False):
    """Unpack batch — handles both (x, y) and (x, y, mask) formats."""
    if len(batch) == 3:
        x, y, mask = batch
        x = x.to(device)
        if channels_last and x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
        return x, y.to(device), mask.to(device)
    x, y = batch
    x = x.to(device)
    if channels_last and x.dim() == 4:
        x = x.to(memory_format=torch.channels_last)
    return x, y.to(device), None


# -- Train / Eval -----------------------------------------------------------

def _apply_mixup(x, y, alpha):
    """MixUp: blend samples with beta-distributed lambda. Returns x_mixed, y_a, y_b, lam."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y, y[idx], lam


def _apply_cutmix(x, y, alpha):
    """CutMix on temporal axis: cut-paste a temporal segment. Returns x_mixed, y_a, y_b, lam."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    T = x.size(2)  # (B, C, T, R)
    cut_len = int(T * (1 - lam))
    t_start = torch.randint(0, T - cut_len + 1, (1,)).item() if cut_len < T else 0
    x_mixed = x.clone()
    x_mixed[:, :, t_start:t_start + cut_len, :] = x[idx, :, t_start:t_start + cut_len, :]
    # Adjust lam to actual area ratio
    lam = 1 - cut_len / T
    return x_mixed, y, y[idx], lam


def _mixup_ce_loss(logits, y_a, y_b, lam, label_smoothing=0.1):
    """CE loss for MixUp/CutMix with two targets and blending lambda."""
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + \
           (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_clip=0, triplet_lambda=0, triplet_margin=0.3,
                    supcon_lambda=0, supcon_loss_fn=None,
                    amp_enabled=False, amp_dtype=torch.bfloat16,
                    is_temporal=False, channels_last=False,
                    mixup_alpha=0, cutmix_alpha=0, mix_prob=0.5,
                    label_smoothing=0.1, confusion_weights=None):
    """Train for one epoch. Returns dict with loss/accuracy metrics."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_ce = torch.tensor(0.0, device=device)
    total_trip = torch.tensor(0.0, device=device)
    total_supcon = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = 0

    need_feats = triplet_lambda > 0 or supcon_lambda > 0
    use_mix = (mixup_alpha > 0 or cutmix_alpha > 0) and not is_temporal

    for batch in tqdm(loader, desc="train", leave=False):
        x, y, mask = _unpack_batch(batch, device, channels_last=channels_last)
        batch_size = len(y)

        # -- MixUp / CutMix (applied before forward pass) --
        mixed = False
        y_a = y_b = y
        lam = 1.0
        if use_mix and torch.rand(1).item() < mix_prob:
            if mixup_alpha > 0 and cutmix_alpha > 0:
                # 50/50 choice between MixUp and CutMix
                if torch.rand(1).item() < 0.5:
                    x, y_a, y_b, lam = _apply_mixup(x, y, mixup_alpha)
                else:
                    x, y_a, y_b, lam = _apply_cutmix(x, y, cutmix_alpha)
            elif mixup_alpha > 0:
                x, y_a, y_b, lam = _apply_mixup(x, y, mixup_alpha)
            else:
                x, y_a, y_b, lam = _apply_cutmix(x, y, cutmix_alpha)
            mixed = True

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
            if need_feats:
                logits, features = forward_with_features(model, x, mask=mask)
            else:
                if is_temporal and mask is not None:
                    logits = model(x, mask=mask)
                else:
                    logits = model(x)

            if mixed:
                ce_loss = _mixup_ce_loss(logits, y_a, y_b, lam, label_smoothing)
            else:
                ce_loss = criterion(logits, y)
            loss = ce_loss

            # Cast features to FP32 for metric losses (BF16 insufficient for distances)
            if need_feats and amp_enabled:
                features = features.float()

            if triplet_lambda > 0:
                if confusion_weights is not None:
                    trip_loss = confusion_weighted_triplet_loss(
                        features, y_a if not mixed else y_a,
                        margin=triplet_margin,
                        confusion_weights=confusion_weights)
                else:
                    trip_loss = online_hard_triplet_loss(
                        features, y_a, margin=triplet_margin)
                loss = loss + triplet_lambda * trip_loss
                total_trip += trip_loss.detach() * batch_size

            if supcon_lambda > 0 and supcon_loss_fn is not None:
                sc_loss = supcon_loss_fn(features, y)
                loss = loss + supcon_lambda * sc_loss
                total_supcon += sc_loss.detach() * batch_size

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.detach() * batch_size
        total_ce += ce_loss.detach() * batch_size
        correct += (logits.detach().argmax(1) == y_a).sum()  # use original labels for acc
        total += batch_size

    return {
        "loss": total_loss.item() / total,
        "ce_loss": total_ce.item() / total,
        "trip_loss": total_trip.item() / total if triplet_lambda > 0 else 0,
        "supcon_loss": total_supcon.item() / total if supcon_lambda > 0 else 0,
        "acc": correct.item() / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=126,
             is_temporal=False, channels_last=False):
    """Evaluate model on validation set. Returns loss, top1, top5, per-class accuracy."""
    model.eval()
    total_loss, top1, top5, total = 0.0, 0, 0, 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for batch in tqdm(loader, desc="val", leave=False):
        x, y, pad_mask = _unpack_batch(batch, device, channels_last=channels_last)
        if is_temporal and pad_mask is not None:
            logits = model(x, mask=pad_mask)
        else:
            logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        top1 += (logits.argmax(1) == y).sum().item()
        top5 += (logits.topk(5, dim=1).indices == y.unsqueeze(1)).any(1).sum().item()
        total += len(y)
        for c in range(num_classes):
            cmask = y == c
            if cmask.any():
                class_correct[c] += (logits.argmax(1)[cmask] == c).sum().item()
                class_total[c] += cmask.sum().item()

    per_class = {c: class_correct[c] / max(class_total[c], 1) for c in range(num_classes)}
    return {
        "loss": total_loss / total,
        "top1": top1 / total,
        "top5": top5 / total,
        "per_class": per_class,
    }


# -- DataLoader with persistent_workers ------------------------------------

def _rebuild_loaders_fast(train_loader, val_loader, cfg, fold_idx):
    """Recreate DataLoaders with persistent_workers and configurable num_workers."""
    nw = cfg.get("num_workers", 8)
    pw = cfg.get("persistent_workers", True) and nw > 0

    if not pw and nw == cfg.get("num_workers", 4):
        return train_loader, val_loader

    seed = cfg.get("seed", 42)
    wif = _make_worker_init_fn(seed + fold_idx)
    g = torch.Generator()
    g.manual_seed(seed + fold_idx)

    train_loader_fast = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        worker_init_fn=wif, generator=g,
        persistent_workers=pw,
    )
    val_loader_fast = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True,
        worker_init_fn=wif,
        persistent_workers=pw,
    )
    return train_loader_fast, val_loader_fast


# -- Fold training ----------------------------------------------------------

def train_fold(cfg, fold_idx, exp_dir, device, run_name, id_to_name=None, resume_from=None):
    """Train a single fold. Returns dict with best metrics."""
    deterministic = cfg.get("deterministic", True)
    set_seed(cfg["seed"] + fold_idx, deterministic=deterministic)
    fold_dir = exp_dir / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if id_to_name is None:
        id_to_name = {}

    triplet_lambda = cfg.get("triplet_lambda", 0)
    triplet_margin = cfg.get("triplet_margin", 0.3)
    supcon_lambda = cfg.get("supcon_lambda", 0)

    # AMP settings
    amp_enabled = cfg.get("amp", False)
    amp_dtype = torch.bfloat16

    # Wandb tags derived from config
    tags = []
    if triplet_lambda > 0:
        tags.append("ce_triplet")
        tags.append(f"lambda={triplet_lambda}")
    else:
        tags.append("ce")
    if supcon_lambda > 0:
        tags.append(f"supcon={supcon_lambda}")
    if cfg.get("pk_sampling", False):
        tags.append("pk_sampling")
    if amp_enabled:
        tags.append("amp_bf16")
    if not deterministic:
        tags.append("non_deterministic")
    tags.append(cfg.get("model_name", "unknown"))

    # Route wandb logs to /data/ (avoid bloating project dir)
    wandb_dir = EXPERIMENTS_DIR / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        entity=cfg["wandb_entity"], project=cfg["wandb_project"],
        name=f"{run_name}_fold{fold_idx}", config={**cfg, "fold": fold_idx},
        group=run_name, tags=tags, reinit=True,
        dir=str(wandb_dir),
    )

    # Data loading
    use_hdf5 = cfg.get("use_hdf5", False)
    if use_hdf5:
        train_loader, val_loader = build_dataloaders_hdf5(cfg, fold_idx=fold_idx)
        print("  [data] Using HDF5 backend")
    else:
        sampling = cfg.get("sampling", "")
        if sampling == "reinforce":
            from src.data.dataset import build_dataloaders_v3
            train_loader, val_loader = build_dataloaders_v3(cfg, fold_idx=fold_idx)
            print("  [sampling] Using REINFORCE sampling")
        else:
            train_loader, val_loader = build_dataloaders(cfg, fold_idx=fold_idx)
        train_loader, val_loader = _rebuild_loaders_fast(train_loader, val_loader, cfg, fold_idx)

    # PK Sampling: replace train_loader sampler
    if cfg.get("pk_sampling", False):
        pk_P = cfg.get("pk_P", 32)
        pk_K = cfg.get("pk_K", 2)
        train_labels = [train_loader.dataset[i][1] for i in range(len(train_loader.dataset))]
        pk_sampler = PKSampler(train_labels, P=pk_P, K=pk_K, seed=cfg["seed"])
        nw = cfg.get("num_workers", 8)
        pw = cfg.get("persistent_workers", True) and nw > 0
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=pk_P * pk_K,
            sampler=pk_sampler, num_workers=nw,
            pin_memory=True, drop_last=True,
            worker_init_fn=_make_worker_init_fn(cfg["seed"] + fold_idx),
            persistent_workers=pw,
        )
        print(f"  PK Sampling: P={pk_P} classes, K={pk_K} samples, batch={pk_P * pk_K}")

    is_temporal = cfg.get("model_type", "cnn") == "temporal"
    model = build_model(cfg).to(device)

    # Resume from previous experiment (warm start)
    best_acc = 0.0
    resumed_wandb_id = None
    if resume_from:
        resume_ckpt = EXPERIMENTS_DIR / resume_from / f"fold{fold_idx}" / "best.pt"
        if resume_ckpt.exists():
            ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            best_acc = ckpt.get("best_top1", 0.0)
            resumed_wandb_id = ckpt.get("wandb_run_id", None)
            print(f"  [resume] Loaded {resume_from}/fold{fold_idx}/best.pt (prev best: {best_acc:.4f})")
            if resumed_wandb_id:
                print(f"  [resume] Will continue Wandb run: {resumed_wandb_id}")
        else:
            print(f"  [resume] WARNING: Checkpoint not found: {resume_ckpt}, training from scratch")

    # Fast mode: channels_last memory format for CNNs
    fast_mode = cfg.get("fast_mode", False)
    if fast_mode and not is_temporal:
        model = model.to(memory_format=torch.channels_last)
        print("[fast_mode] channels_last memory format enabled")

    n_params = sum(p.numel() for p in model.parameters())
    loss_desc = "CE"
    if triplet_lambda > 0:
        loss_desc += f" + Triplet(λ={triplet_lambda})"
    if supcon_lambda > 0:
        loss_desc += f" + SupCon(λ={supcon_lambda})"
    if cfg.get("loss_fn", "ce") == "focal":
        loss_desc += f" [Focal γ={cfg.get('focal_gamma', 2.0)}]"
    if cfg.get("mixup_alpha", 0) > 0:
        loss_desc += f" + MixUp(α={cfg['mixup_alpha']})"
    if cfg.get("cutmix_alpha", 0) > 0:
        loss_desc += f" + CutMix(α={cfg['cutmix_alpha']})"
    print(f"  Loss: {loss_desc}, AMP: {amp_enabled}")
    print(f"  Model: {'temporal' if is_temporal else cfg.get('model_name', '?')}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn = cfg.get("loss_fn", "ce")
    if loss_fn == "focal":
        from src.losses.focal import FocalLoss
        criterion = FocalLoss(
            gamma=cfg.get("focal_gamma", 2.0),
            label_smoothing=cfg.get("label_smoothing", 0.1),
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
    grad_clip = cfg.get("grad_clip", 0)

    # SupCon auxiliary loss
    supcon_loss_fn = None
    if supcon_lambda > 0:
        from src.losses.supcon import SupConLoss
        supcon_loss_fn = SupConLoss(temperature=cfg.get("supcon_temperature", 0.07))
        print(f"  SupCon: temperature={cfg.get('supcon_temperature', 0.07)}")

    # Confusion-weighted triplet
    cw_path = cfg.get("confusion_weights_path", "")
    confusion_weights = None
    if cw_path and triplet_lambda > 0:
        cw_data = np.load(cw_path)
        confusion_weights = torch.from_numpy(cw_data["weights"].astype(np.float32)).to(device)
        print(f"  Confusion weights: {cw_path} shape={confusion_weights.shape}")

    # SWA
    use_swa = cfg.get("swa", False)
    swa_start = cfg.get("swa_start_epoch", 20)
    swa_model, swa_scheduler = None, None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.get("swa_lr", 1e-4))

    best_acc, patience_cnt = 0.0, 0
    best_train_acc, best_val_top5 = 0.0, 0.0
    best_val_loss = float("inf")
    patience = cfg.get("patience", cfg["epochs"])
    epoch_history = []

    # Smoothed early stopping: use moving average of val_top1 for patience decisions
    es_window_size = cfg.get("es_window", 1)
    es_window = deque(maxlen=es_window_size)
    best_smooth = 0.0

    # UMAP (conditional import)
    umap_gif = cfg.get("umap_gif", False)
    umap_freq = cfg.get("umap_every", 10)
    umap_log_freq = cfg.get("umap_log_every", 50)
    umap_frames_mixed, umap_frames_letters, umap_frames_words = [], [], []
    umap_prev_embs = {}
    if umap_gif:
        from src.umap_viz import log_umap_to_wandb, save_and_log_umap_gifs

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_t0 = _time.time()

        use_channels_last = fast_mode and not is_temporal
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip, triplet_lambda, triplet_margin,
            supcon_lambda=supcon_lambda, supcon_loss_fn=supcon_loss_fn,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            is_temporal=is_temporal, channels_last=use_channels_last,
            mixup_alpha=cfg.get("mixup_alpha", 0),
            cutmix_alpha=cfg.get("cutmix_alpha", 0),
            mix_prob=cfg.get("mix_prob", 0.5),
            label_smoothing=cfg.get("label_smoothing", 0.1),
            confusion_weights=confusion_weights)
        val = evaluate(model, val_loader, criterion, device, cfg["num_classes"],
                       is_temporal=is_temporal, channels_last=use_channels_last)

        # Scheduler step
        scheduler_type = cfg.get("scheduler", "cosine")
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif scheduler_type == "plateau":
            if epoch <= cfg.get("warmup_epochs", 0):
                wf = 0.01 + 0.99 * epoch / cfg["warmup_epochs"]
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["lr"] * wf
            else:
                scheduler.step(val["top1"])
        else:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = _time.time() - epoch_t0

        # Smoothed val_top1 for early stopping
        es_window.append(val["top1"])
        val_top1_smooth = sum(es_window) / len(es_window)

        # Best checkpoint: always based on RAW val_top1
        if val["top1"] > best_acc:
            best_acc = val["top1"]
            best_train_acc = train_metrics["acc"]
            best_val_top5 = val["top5"]
            best_val_loss = val["loss"]
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {"model": raw_model.state_dict(), "epoch": epoch,
                 "best_top1": best_acc, "config": cfg, "fold": fold_idx,
                 "wandb_run_id": wandb.run.id},
                fold_dir / "best.pt",
            )
            print(f"  -> New best: top1={best_acc:.4f}  top5={best_val_top5:.4f}")

        # Patience: based on smoothed val_top1 (reduces noise-induced early stops)
        if val_top1_smooth > best_smooth:
            best_smooth = val_top1_smooth
            patience_cnt = 0
        else:
            patience_cnt += 1

        # Log AFTER best/patience update so values are current
        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_trip_loss": train_metrics["trip_loss"],
            "train_supcon_loss": train_metrics.get("supcon_loss", 0),
            "train_acc": train_metrics["acc"],
            "val_loss": val["loss"], "val_acc": val["top1"], "val_top5_acc": val["top5"],
            "val_top1_smooth": val_top1_smooth,
            "best_val_top1": best_acc,
            "patience_cnt": patience_cnt,
            "lr": lr_now,
            "epoch_time_s": round(elapsed, 1),
        })

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_ce_loss": round(train_metrics["ce_loss"], 6),
            "train_trip_loss": round(train_metrics["trip_loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "val_loss": round(val["loss"], 6),
            "val_top1": round(val["top1"], 6),
            "val_top1_smooth": round(val_top1_smooth, 6),
            "best_val_top1": round(best_acc, 6),
            "patience_cnt": patience_cnt,
            "val_top5": round(val["top5"], 6),
            "lr": round(lr_now, 8),
            "epoch_time_s": round(elapsed, 1),
        })

        smooth_str = f"  smooth={val_top1_smooth:.4f}" if es_window_size > 1 else ""
        print(
            f"[fold{fold_idx}] Epoch {epoch}/{cfg['epochs']}  "
            f"train={train_metrics['acc']:.4f}  val_top1={val['top1']:.4f}{smooth_str}  "
            f"val_top5={val['top5']:.4f}  lr={lr_now:.6f}  "
            f"ce={train_metrics['ce_loss']:.4f}  trip={train_metrics['trip_loss']:.4f}  "
            f"time={elapsed:.0f}s"
        )

        # UMAP
        if umap_gif and (epoch == 1 or epoch % umap_freq == 0 or epoch == cfg["epochs"]):
            should_log = (epoch == 1 or epoch % umap_log_freq == 0 or epoch == cfg["epochs"])
            log_umap_to_wandb(model, val_loader, device, epoch,
                              frames_mixed=umap_frames_mixed,
                              frames_letters=umap_frames_letters,
                              frames_words=umap_frames_words,
                              prev_embs=umap_prev_embs,
                              feature_fn=_ce_feature_fn,
                              log_wandb=should_log)

        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch} (smooth={val_top1_smooth:.4f}, best_smooth={best_smooth:.4f})")
            break

    # UMAP GIFs
    if umap_gif:
        save_and_log_umap_gifs(umap_frames_mixed, umap_frames_letters,
                               umap_frames_words, fold_dir)

    # Per-class accuracy table from BEST model (not last epoch)
    best_ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(best_ckpt["model"])
    use_channels_last = cfg.get("fast_mode", False) and not is_temporal
    best_val = evaluate(model, val_loader, criterion, device, cfg["num_classes"],
                        is_temporal=is_temporal, channels_last=use_channels_last)
    print(f"  Best model re-eval: top1={best_val['top1']:.4f}  top5={best_val['top5']:.4f}")

    table = wandb.Table(columns=["class_id", "class_name", "val_accuracy"])
    for c in range(cfg["num_classes"]):
        cname = id_to_name.get(c, str(c))
        table.add_data(c, cname, best_val["per_class"].get(c, 0.0))
    wandb.log({"per_class_accuracy": table})
    wandb.finish()

    # Save fold results
    fold_results = {
        "fold": fold_idx,
        "best_epoch": epoch - patience_cnt,
        "total_epochs": epoch,
        "best_train_acc": round(best_train_acc, 6),
        "best_val_top1": round(best_acc, 6),
        "best_val_top5": round(best_val_top5, 6),
        "best_val_loss": round(best_val_loss, 6),
        "epoch_history": epoch_history,
    }
    with open(fold_dir / "results.json", "w") as f:
        json.dump(fold_results, f, indent=2)

    print(f"Fold {fold_idx} done. train={best_train_acc:.4f}  val_top1={best_acc:.4f}  val_top5={best_val_top5:.4f}")
    return {"train_acc": best_train_acc, "val_top1": best_acc, "val_top5": best_val_top5}


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified sign language recognition training")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=None, help="Override train_fold from YAML")
    parser.add_argument("--device", type=str, default=None, help="Override device from YAML")
    parser.add_argument("--resume-from", type=str, default=None, help="Experiment name to load best.pt weights from (warm start)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.fold is not None:
        cfg["train_fold"] = args.fold
        cfg["train_all_folds"] = False
    if args.device is not None:
        cfg["device"] = args.device

    # Validate config for reproducibility issues
    validate_config(cfg)

    # Architecture-specific sanity checks
    model_name = cfg.get("model_name", "")
    dpr = cfg.get("drop_path_rate", 0)
    if model_name.startswith("mobileone") and dpr > 0:
        print(f"WARNING: {model_name} with drop_path_rate={dpr} may fail to train. "
              f"MobileOne reparameterizable blocks do not tolerate stochastic depth. "
              f"Original experiments used drop_path_rate=0.0")

    # Fast mode: TF32 matmul
    if cfg.get("fast_mode", False):
        torch.set_float32_matmul_precision('medium')
        print("[fast_mode] TF32 matmul precision enabled")

    device = torch.device(cfg["device"])

    exp_name = cfg.get("experiment_name", "")
    if not exp_name:
        import re
        stem = Path(args.config).stem
        exp_name = re.sub(r"^exp\d+_", "", stem)
        print(f"Auto-generated experiment_name: {exp_name}")

    exp_dir = EXPERIMENTS_DIR / exp_name
    folds = range(cfg["n_folds"]) if cfg.get("train_all_folds") else [cfg.get("train_fold", 0)]

    # Check for completed experiment
    if exp_dir.is_dir() and (exp_dir / "experiment_summary.json").exists():
        if args.fold is not None:
            fold_dir = exp_dir / f"fold{args.fold}"
            if (fold_dir / "best.pt").exists():
                print(f"Fold {args.fold} already completed. Skipping.")
                sys.exit(0)
            print(f"Resuming experiment '{exp_name}' for fold {args.fold}")
        else:
            print(f"WARNING: Experiment '{exp_name}' already completed at {exp_dir}")
            print("Skipping training. Delete the directory to retrain.")
            sys.exit(0)
    elif exp_dir.is_dir() and args.fold is not None:
        fold_dir = exp_dir / f"fold{args.fold}"
        if (fold_dir / "best.pt").exists():
            print(f"Fold {args.fold} already completed. Skipping.")
            sys.exit(0)

    exp_dir.mkdir(parents=True, exist_ok=True)
    if not (exp_dir / "config.yaml").exists():
        shutil.copy2(PROJECT_ROOT / args.config, exp_dir / "config.yaml")

    id_to_name = load_class_mapping()
    if id_to_name:
        print(f"Loaded class mapping: {len(id_to_name)} classes")

    results = {}
    for fold_idx in folds:
        fold_dir = exp_dir / f"fold{fold_idx}"
        if (fold_dir / "best.pt").exists():
            print(f"Fold {fold_idx} already completed, skipping.")
            results_path = fold_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    fr = json.load(f)
                results[fold_idx] = {
                    "train_acc": fr["best_train_acc"],
                    "val_top1": fr["best_val_top1"],
                    "val_top5": fr["best_val_top5"],
                }
            continue
        results[fold_idx] = train_fold(cfg, fold_idx, exp_dir, device, exp_name, id_to_name, resume_from=args.resume_from)

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  {'Fold':<6} {'Train':>8} {'Val Top1':>10} {'Val Top5':>10}")
    print(f"  {'-' * 34}")
    for f, r in results.items():
        print(f"  {f:<6} {r['train_acc']:>8.4f} {r['val_top1']:>10.4f} {r['val_top5']:>10.4f}")
    print(f"  {'-' * 34}")
    mean_train = np.mean([r["train_acc"] for r in results.values()])
    mean_top1 = np.mean([r["val_top1"] for r in results.values()])
    mean_top5 = np.mean([r["val_top5"] for r in results.values()])
    print(f"  {'Mean':<6} {mean_train:>8.4f} {mean_top1:>10.4f} {mean_top5:>10.4f}")
    print(f"{'=' * 60}")

    # Experiment summary with reproducibility metadata
    experiment_summary = {
        "experiment_name": exp_name,
        "config_file": args.config,
        "script": "codes/final/train.py",
        "config": cfg,
        "fold_results": {str(k): v for k, v in results.items()},
        "mean_train_acc": round(float(mean_train), 6),
        "mean_val_top1": round(float(mean_top1), 6),
        "mean_val_top5": round(float(mean_top5), 6),
        "reproducibility": {
            "deterministic": cfg.get("deterministic", True),
            "amp": cfg.get("amp", False),
            "fast_mode": cfg.get("fast_mode", False),
            "script_version": "1.0.0",
            "git_commit": get_git_hash(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }
    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"Experiment summary saved to: {exp_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
