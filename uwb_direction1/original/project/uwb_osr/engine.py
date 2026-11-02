"""Training and open-set evaluation for a single OSR test case."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MultiViewOSRNet
from .losses import tmc_loss, supcon_loss, ood_mixup_loss, pos_jitter_loss
from .metrics import auroc, oscr, best_threshold_acc_f1


def get_device(cfg):
    use = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if use == "cuda":
        torch.backends.cudnn.benchmark = True
    return use


def _sample_kept_views(cfg):
    """R1 view dropout: pick which trained-view slots a batch keeps.

    Returns None to keep all views (the default / when disabled), else a sorted
    list of kept slots of size 1..n_views-1 (a random sensor-failure subset).
    """
    V = cfg.n_views
    if cfg.view_dropout <= 0 or V < 2:
        return None
    if torch.rand(()).item() >= cfg.view_dropout:
        return None                                  # this batch keeps all views
    k = int(torch.randint(1, V, ()).item())          # keep 1..V-1 views
    return torch.randperm(V)[:k].sort().values.tolist()


class _GradReverse(torch.autograd.Function):
    """N4c: identity forward, -lamb * grad backward (DANN gradient reversal)."""
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def _dann_lambda(cfg, epoch):
    """DANN ramp 0 -> lambda_subj_adv: 2/(1+e^{-10p}) - 1 over training."""
    p = (epoch + 1) / max(1, cfg.epochs)
    return cfg.lambda_subj_adv * (2.0 / (1.0 + float(np.exp(-10.0 * p))) - 1.0)


def _proto_episode_loss(f_views, y):
    """N4b: in-batch prototypical episode, averaged over views.

    Per view, every class with >=2 samples in the batch donates a random half
    as SUPPORT (prototype = mean feature) and the rest as QUERY, classified
    among the in-batch prototypes by -squared distance. This is the deployment
    condition of few-shot registration written as a training objective: a mean
    of a few samples must classify samples it has never seen. Returns 0 when
    the batch has <2 eligible classes."""
    B, V, d = f_views.shape
    losses = []
    for v in range(V):
        f = f_views[:, v]
        classes = [c for c in y.unique().tolist() if int((y == c).sum()) >= 2]
        if len(classes) < 2:
            continue
        protos, q_feat, q_lab = [], [], []
        for j, c in enumerate(classes):
            idx = torch.where(y == c)[0]
            perm = idx[torch.randperm(len(idx), device=y.device)]
            k = max(1, len(perm) // 2)
            protos.append(f[perm[:k]].mean(0))
            q_feat.append(f[perm[k:]])
            q_lab.append(torch.full((len(perm) - k,), j, device=y.device,
                                    dtype=torch.long))
        q_feat = torch.cat(q_feat)
        if len(q_feat) == 0:
            continue
        logits = -torch.cdist(q_feat, torch.stack(protos)) ** 2
        losses.append(F.cross_entropy(logits, torch.cat(q_lab)))
    if not losses:
        return f_views.sum() * 0.0
    return torch.stack(losses).mean()


def train_one_case(cfg, train_ds, n_known, device, verbose=True, desc="Stage-1",
                   mae_state=None, n_subj=0):
    model = MultiViewOSRNet(cfg, n_known).to(device)
    if mae_state is not None:                       # B2: Stage-0 encoder init
        model.encoder.load_state_dict(mae_state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    subj_head = None                                 # N4c: built lazily (dim of
    use_adv = cfg.lambda_subj_adv > 0 and n_subj > 1  # f_views known at 1st batch)
    pin = device == "cuda"
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True, pin_memory=pin)
    model.train()
    pbar = tqdm(range(cfg.epochs), desc=desc, disable=not verbose,
                dynamic_ncols=True)
    for epoch in pbar:
        anneal = min(1.0, (epoch + 1) / max(1, cfg.edl_kl_anneal_epochs))
        lamb_adv = _dann_lambda(cfg, epoch) if use_adv else 0.0
        running = 0.0
        for batch in loader:
            x, y = batch[0], batch[1]
            s = batch[2] if use_adv and len(batch) > 2 else None
            x = x.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)
            keep = _sample_kept_views(cfg)           # R1: None unless view_dropout>0
            out = (model(x) if keep is None
                   else model(x[:, keep], view_indices=keep))
            loss = F.cross_entropy(out["logits"], y)
            # N4b: episodic auxiliary loss on the same features the few-shot
            # prototypes will live in.
            if cfg.lambda_episodic > 0:
                loss = loss + cfg.lambda_episodic * _proto_episode_loss(
                    out["f_views"], y)
            # N4c: subject-adversarial head on the per-view features. The head
            # itself trains at full strength to identify the subject; the
            # reversal sends -lamb * grad into the encoder, scrubbing subject
            # identity from the very features the few-shot prototypes live in.
            if s is not None:
                B, V, d = out["f_views"].shape
                if subj_head is None:
                    subj_head = torch.nn.Sequential(
                        torch.nn.Linear(d, 128), torch.nn.ReLU(),
                        torch.nn.Linear(128, n_subj)).to(device)
                    opt.add_param_group({"params": subj_head.parameters()})
                fv = _GradReverse.apply(out["f_views"].reshape(B * V, d), lamb_adv)
                s_dev = s.to(device, non_blocking=pin).repeat_interleave(V)
                loss = loss + F.cross_entropy(subj_head(fv), s_dev)
            if cfg.osr_head == "evidential":
                loss = loss + tmc_loss(out, y, n_known, anneal)
                if cfg.lambda_ood > 0:
                    loss = loss + cfg.lambda_ood * ood_mixup_loss(
                        model.evid_head, out["f_views"], y, n_known,
                        cfg.ood_mix_low, cfg.ood_mix_high)
                if cfg.lambda_pos > 0:
                    loss = loss + cfg.lambda_pos * pos_jitter_loss(
                        model.evid_head, out["f_views"], y, n_known, cfg.pos_jitter)
            if cfg.use_supcon:
                B, V, d = out["f_views"].shape
                feats = out["f_views"].reshape(B * V, d)
                labels = y.repeat_interleave(V)
                loss = loss + cfg.lambda_supcon * supcon_loss(
                    feats, labels, cfg.supcon_temp)
            # KD: on a view-dropped batch, distill the kept-subset student toward
            # the full-view self-teacher (same weights, no_grad). Only the closed
            # logits (and optionally the evidential uncertainty) differ with the
            # view set, so those are the meaningful distillation targets.
            if cfg.kd_distill > 0 and keep is not None:
                with torch.no_grad():
                    t_out = model(x)             # full-view teacher
                T = cfg.kd_temp
                kd = F.kl_div(F.log_softmax(out["logits"] / T, -1),
                              F.softmax(t_out["logits"] / T, -1),
                              reduction="batchmean") * (T * T)
                loss = loss + cfg.kd_distill * kd
                if cfg.kd_uncert > 0:
                    loss = loss + cfg.kd_uncert * F.mse_loss(
                        out["uncertainty"], t_out["uncertainty"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        pbar.set_postfix(loss=f"{running/len(loader):.4f}")
    return model


@torch.no_grad()
def _collect(model, ds, cfg, device):
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=cfg.num_workers)
    probs, unc, logits, labels = [], [], [], []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(out["prob"].cpu())
        unc.append(out["uncertainty"].cpu())
        logits.append(out["logits"].cpu())
        labels.append(y)
    return (torch.cat(probs).numpy(), torch.cat(unc).numpy(),
            torch.cat(logits).numpy(), torch.cat(labels).numpy())


def evaluate_osr(model, known_ds, unknown_ds, cfg, n_known, device):
    pk, uk, lk, yk = _collect(model, known_ds, cfg, device)
    pu, uu, lu, _ = _collect(model, unknown_ds, cfg, device)

    # closed-set accuracy on known-test (no rejection)
    pred_k = pk.argmax(1)
    closed_acc = float((pred_k == yk).mean())

    results = {"closed_acc": closed_acc}
    for name, conf_k, conf_u, pred, pred_u in [
        ("evidential", 1.0 - uk, 1.0 - uu, pred_k, pu.argmax(1)),
        ("msp", _softmax(lk).max(1), _softmax(lu).max(1), lk.argmax(1),
         lu.argmax(1)),
    ]:
        is_known = np.concatenate([np.ones(len(conf_k)), np.zeros(len(conf_u))])
        scores = np.concatenate([conf_k, conf_u])
        au = auroc(scores, is_known)
        oc = oscr(conf_k, (pred == yk).astype(float), conf_u)
        acc, f1, thr = best_threshold_acc_f1(pred, conf_k, yk, conf_u, n_known,
                                             pred_u=pred_u)
        results[name] = dict(auroc=au, oscr=oc, acc=acc, f1=f1, thr=float(thr))
    return results


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
