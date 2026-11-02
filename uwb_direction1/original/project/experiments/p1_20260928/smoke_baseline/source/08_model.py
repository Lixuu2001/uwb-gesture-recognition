"""Model: shared dual-path encoder + cross-view fusion + OSR heads.

Pooling is a switch (cfg.pool):
  * "mean"    -> per-path mean pool, feature code = 2*D   (64, lightweight)
  * "flatten" -> per-path token flatten, feature code = D*(T+M)  (8928, Eom)

Pipeline:
  X (B,V,T,M)
    -> SharedDualPathEncoder -> f_views (B,V,feat), temporal+spatial tokens
    -> CrossViewFusion       -> c_fused (B,feat)        [closed-set readout]
    -> ClosedSetHead         -> logits (B,K)
    -> EvidentialHead(f_v)   -> alpha_v (B,V,K) -> Dempster-Shafer -> prob,u
The deep-inspector OSR head lives in inspector.py and consumes f_views.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1) Encoder
# ---------------------------------------------------------------------------
class WideHeadMSA(nn.Module):
    """Eom Eq.7-9 MSA: each head projects D-dim tokens to D_key dims (instead of
    PyTorch's D/heads split), heads are concatenated to heads*D_key and Phi^O
    projects back to D. With D=32, D_key=32, N_h=4 this is the paper's exact
    attention width; token dim and feature-code size are unchanged."""

    def __init__(self, dim, heads, d_key, dropout):
        super().__init__()
        self.h, self.dk = heads, d_key
        inner = heads * d_key
        self.qkv = nn.Linear(dim, 3 * inner)
        self.out = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                          # x: (B, L, D)
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)   # (B, h, L, dk)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.dk ** 0.5
        attn = self.drop(attn.softmax(-1))
        o = (attn @ v).transpose(1, 2).reshape(B, L, -1)    # concat heads
        return self.out(o)


class WideHeadBlock(nn.Module):
    """Pre-LN ViT block (LN -> MSA -> add, LN -> MLP -> add) around WideHeadMSA;
    mirrors nn.TransformerEncoderLayer(norm_first=True, activation=gelu)."""

    def __init__(self, dim, heads, d_key, mlp_ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = WideHeadMSA(dim, heads, d_key, dropout)
        self.d1 = nn.Dropout(dropout)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.d1(self.attn(self.n1(x)))
        return x + self.mlp(self.n2(x))


def _make_block(dim, heads, mlp_ratio, dropout, d_key=0):
    """One pre-LN transformer block mapping (B,L,D)->(B,L,D): WideHeadBlock
    when d_key>0 (Eom Eq.7-9 wide heads), else nn.TransformerEncoderLayer."""
    if d_key and d_key > 0:
        return WideHeadBlock(dim, heads, d_key, mlp_ratio, dropout)
    return nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True)


class BottleneckFusionLayer(nn.Module):
    """F2 X-MV mid-fusion block (MBT-style, Nagrani et al. NeurIPS'21).

    Per view, tokens attend over [own tokens ; shared bottleneck]; the per-view
    updated bottleneck copies are averaged afterwards, so the NB bottleneck
    tokens are the ONLY cross-view channel. Views exchange a compressed summary
    instead of full pairwise attention, which regularises the exchange and
    keeps per-view token streams (hence per-view feature codes) intact.
    """

    def __init__(self, dim, heads, mlp_ratio, dropout, d_key=0):
        super().__init__()
        self.block = _make_block(dim, heads, mlp_ratio, dropout, d_key)

    def forward(self, toks, btl):              # toks (B,V,L,D), btl (B,NB,D)
        B, V, L, D = toks.shape
        outs, btls = [], []
        for v in range(V):                     # V<=3, loop cost negligible
            seq = self.block(torch.cat([toks[:, v], btl], 1))
            outs.append(seq[:, :L])
            btls.append(seq[:, L:])
        return torch.stack(outs, 1), torch.stack(btls, 1).mean(1)


class TransformerPath(nn.Module):
    """Sequence of `seq_len` tokens of dim `in_dim` -> token embeddings (B,L,D).

    xmv_layers=0 (default): the original per-view encoder, byte-for-byte.
    xmv_layers=K>0 (F2): the last K of the `depth` blocks become cross-view
    BottleneckFusionLayers (total depth unchanged, so capacity stays at the
    E3-validated point). Tokens first specialise per view, then exchange
    through the bottleneck (MBT mid-fusion). View identity is added on the
    tokens (view_emb_tok) before the exchange so fusion is not
    permutation-blind across radars.
    """

    def __init__(self, seq_len, in_dim, dim, depth, heads, mlp_ratio, dropout,
                 d_key=0, xmv_layers=0, xmv_bottleneck=4, n_views=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.xmv = min(max(xmv_layers, 0), depth)
        if self.xmv == 0:                          # original construction,
            if d_key and d_key > 0:                # kept verbatim (RNG-equal)
                self.encoder = nn.Sequential(*[
                    WideHeadBlock(dim, heads, d_key, mlp_ratio, dropout)
                    for _ in range(depth)])
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                    dropout=dropout, activation="gelu", batch_first=True,
                    norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            return
        indep = depth - self.xmv
        if d_key and d_key > 0:
            self.encoder = nn.Sequential(*[
                WideHeadBlock(dim, heads, d_key, mlp_ratio, dropout)
                for _ in range(indep)])
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=indep)
        self.view_emb_tok = nn.Parameter(torch.randn(n_views, 1, dim) * 0.02)
        self.btl = nn.Parameter(torch.randn(1, xmv_bottleneck, dim) * 0.02)
        self.fusion_layers = nn.ModuleList(
            BottleneckFusionLayer(dim, heads, mlp_ratio, dropout, d_key)
            for _ in range(self.xmv))

    def forward(self, x, V=1, view_indices=None):  # x: (B*V, seq_len, in_dim)
        h = self.encoder(self.proj(x) + self.pos)  # (B*V, seq_len, D)
        if self.xmv == 0:
            return h
        BV, L, D = h.shape
        B = BV // V
        toks = h.reshape(B, V, L, D)
        slots = list(range(V)) if view_indices is None else view_indices
        toks = toks + self.view_emb_tok[slots].unsqueeze(0)    # view identity
        btl = self.btl.expand(B, -1, -1)
        # V'==1 still runs the blocks (consistent depth for any view subset);
        # the bottleneck then simply carries no cross-view information.
        for lyr in self.fusion_layers:
            toks, btl = lyr(toks, btl)
        return toks.reshape(BV, L, D)


# ============================================================================
#   POOL SWITCH  (cfg.pool ; CLI --pool {mean, flatten})
#   This is the ONLY place the encoder branches on pooling mode. It controls
#   the feature-code dimension feeding the closed-set head and inspector:
#       mean    -> per-path D-dim mean    -> feat = 2*D    (= 64)
#       flatten -> per-path D*L-dim flat -> feat = D*(T+M) (= 8928, Eom)
# ============================================================================
def reduce_tokens(tok, pool, axis):
    """Collapse the token axis. tok has the token axis at position `axis`.
    pool='mean' -> mean over that axis; pool='flatten' -> flatten it into D."""
    if pool == "mean":
        return tok.mean(axis)
    return tok.flatten(axis)                        # merges token axis into feature


class SharedDualPathEncoder(nn.Module):
    """Eom dual-path Transformer, weights shared across views (Siamese)."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        # M_in == cfg.M unless the coherent Doppler channel is concatenated on
        # the range axis (cfg.doppler=concat), which doubles what the encoder
        # sees while the cache keeps its physical 189 bins.
        self.T, self.M = cfg.T, cfg.M_in
        self.pool = cfg.pool
        d_key = getattr(cfg, "d_key", 0)
        xmv_l = getattr(cfg, "xmv_layers", 0)
        xmv_b = getattr(cfg, "xmv_bottleneck", 4)
        self.temporal = TransformerPath(cfg.T, cfg.M_in, cfg.dim, cfg.depth,
                                        cfg.heads, cfg.mlp_ratio, cfg.dropout,
                                        d_key=d_key, xmv_layers=xmv_l,
                                        xmv_bottleneck=xmv_b,
                                        n_views=cfg.n_views)
        self.spatial = TransformerPath(cfg.M_in, cfg.T, cfg.dim, cfg.depth,
                                       cfg.heads, cfg.mlp_ratio, cfg.dropout,
                                       d_key=d_key, xmv_layers=xmv_l,
                                       xmv_bottleneck=xmv_b,
                                       n_views=cfg.n_views)
        self.view_emb = nn.Parameter(torch.randn(cfg.n_views, self.feat_dim) * 0.02)

    @property
    def feat_dim(self):
        return self.dim * (self.T + self.M) if self.pool == "flatten" else 2 * self.dim

    def forward(self, X, view_indices=None):        # X: (B,V',T,M)
        # view_indices (E2 missing-view eval): which trained-view slots the input
        # corresponds to. None -> use slots [0..V-1]. Otherwise the slots are
        # used to pick the right rows of view_emb (so a "right radar only" test
        # uses view_emb[2], not view_emb[0]).
        B, V, T, M = X.shape
        xb = X.reshape(B * V, T, M)
        t_tok = self.temporal(xb, V=V, view_indices=view_indices)         # (B*V',T,D)
        s_tok = self.spatial(xb.transpose(1, 2), V=V,
                             view_indices=view_indices)                   # (B*V',M,D)
        f = torch.cat([reduce_tokens(t_tok, self.pool, 1),
                       reduce_tokens(s_tok, self.pool, 1)], -1)   # (B*V', feat)
        f = f.reshape(B, V, -1)
        f = f + (self.view_emb if view_indices is None
                 else self.view_emb[view_indices])
        t_tok = t_tok.reshape(B, V, T, -1)
        s_tok = s_tok.reshape(B, V, M, -1)
        return f, t_tok, s_tok


# ---------------------------------------------------------------------------
# 2) Cross-view fusion (Method A: time-axis aligned)
# ---------------------------------------------------------------------------
def physical_bias(X):
    """Per-sample VxV physical priors (B,V,V,3): energy-correlation, peak-frame
    delay, range-direction consistency. Injected as additive attention bias."""
    B, V, T, M = X.shape
    E = (X ** 2).sum(-1)                                    # (B,V,T)
    En = (E - E.mean(-1, keepdim=True))
    En = En / (En.norm(dim=-1, keepdim=True) + 1e-6)
    rho = torch.einsum("bvt,bwt->bvw", En, En) / T
    tpk = E.argmax(-1).float()
    dt = (tpk[:, :, None] - tpk[:, None, :]) / T
    mpk = X.abs().argmax(-1).float()
    sdir = torch.sign(mpk[:, :, -1] - mpk[:, :, 0])
    c = sdir[:, :, None] * sdir[:, None, :]
    return torch.stack([rho, dt, c], dim=-1)


class CrossViewFusion(nn.Module):
    """Per slow-time index, attend across the V views (factorised)."""

    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.use_bias = cfg.use_physical_bias
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.dim)
        if self.use_bias:
            self.w_rho = nn.Parameter(torch.zeros(1))
            self.w_dir = nn.Parameter(torch.zeros(1))
            self.delay_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(),
                                           nn.Linear(16, 1))

    def forward(self, t_tokens, X=None):           # t_tokens: (B,V,T,D)
        B, V, T, D = t_tokens.shape
        if V == 1:                                 # single view: nothing to fuse
            return t_tokens
        x = t_tokens.permute(0, 2, 1, 3).reshape(B * T, V, D)
        attn_mask = None
        if self.use_bias and X is not None:
            phys = physical_bias(X)
            rho, dt, c = phys[..., 0], phys[..., 1], phys[..., 2]
            bias = (self.w_rho * rho + self.w_dir * c
                    + self.delay_mlp(dt.unsqueeze(-1)).squeeze(-1)).tanh() * 5.0
            bias = bias.unsqueeze(1).expand(B, T, V, V).reshape(B * T, V, V)
            attn_mask = bias.repeat_interleave(self.heads, dim=0)
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        out = self.norm(x + out)
        return out.reshape(B, T, V, D).permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# 3) Heads
# ---------------------------------------------------------------------------
class ClosedSetHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2048, 256), nn.GELU(),
            nn.Linear(256, n_classes))

    def forward(self, c):
        return self.net(c)


class EvidentialHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_classes))

    def forward(self, f):
        return F.softplus(self.net(f))


# ---------------------------------------------------------------------------
# 4) Dempster-Shafer combination (Trusted Multi-View Classification)
# ---------------------------------------------------------------------------
def ds_combine_two(alpha1, alpha2, K):
    S1, S2 = alpha1.sum(-1, keepdim=True), alpha2.sum(-1, keepdim=True)
    b1, b2 = (alpha1 - 1) / S1, (alpha2 - 1) / S2
    u1, u2 = K / S1, K / S2
    C = (b1.sum(-1, keepdim=True) * b2.sum(-1, keepdim=True)
         - (b1 * b2).sum(-1, keepdim=True))
    denom = (1.0 - C).clamp(min=1e-7)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / denom
    u = (u1 * u2) / denom
    return b * (K / u.clamp(min=1e-7)) + 1.0


def ds_combine(alpha_views, K):                    # (B,V,K) -> (B,K)
    a = alpha_views[:, 0]
    for v in range(1, alpha_views.size(1)):
        a = ds_combine_two(a, alpha_views[:, v], K)
    return a


# ---------------------------------------------------------------------------
# 5) Top model
# ---------------------------------------------------------------------------
class MultiViewOSRNet(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        self.cfg = cfg
        self.K = n_classes
        self.encoder = SharedDualPathEncoder(cfg)
        self.feat_dim = self.encoder.feat_dim
        self.use_fusion = cfg.use_fusion
        if cfg.use_fusion:
            self.fusion = CrossViewFusion(cfg)
        self.closed_head = ClosedSetHead(self.feat_dim, n_classes)
        self.evid_head = EvidentialHead(self.feat_dim, n_classes)

    def forward(self, X, view_indices=None):
        pool = self.cfg.pool
        # f_views are per-radar feature codes used by the inspector heads.
        # c_fused below is the closed-set readout feature; in multi-view mode it
        # can include cross-view temporal attention before the classifier.
        # view_indices (E2) lets eval forward an arbitrary subset of the trained
        # views (any V' <= n_views), with view_emb sliced to match.
        f_views, t_tok, s_tok = self.encoder(X, view_indices=view_indices)  # (B,V',feat),(B,V',T,D),(B,V',M,D)
        t_src = self.fusion(t_tok, X) if self.use_fusion else t_tok
        c_t = reduce_tokens(t_src, pool, 2).mean(1)        # (B, Dt)
        c_s = reduce_tokens(s_tok, pool, 2).mean(1)        # (B, Ds)
        c_fused = torch.cat([c_t, c_s], -1)                # (B, feat)
        logits = self.closed_head(c_fused)

        # F1: per-view feature codes read out AFTER the cross-view temporal
        # attention (spatial path has no post-encoder cross-view stage).
        # Identical to f_views when fusion is off or V'==1.
        if getattr(self.cfg, "inspector_feats", "pre") == "fused":
            f_fused = torch.cat([reduce_tokens(t_src, pool, 2),
                                 reduce_tokens(s_tok, pool, 2)], -1)
            f_fused = f_fused + (self.encoder.view_emb if view_indices is None
                                 else self.encoder.view_emb[view_indices])
        else:
            f_fused = f_views

        V = f_views.size(1)
        evid = torch.stack([self.evid_head(f_views[:, v]) for v in range(V)], 1)
        alpha_v = evid + 1.0
        alpha_c = ds_combine(alpha_v, self.K)
        S = alpha_c.sum(-1, keepdim=True)
        return dict(logits=logits, f_views=f_views, f_views_fused=f_fused,
                    c_fused=c_fused, alpha_v=alpha_v, alpha_c=alpha_c,
                    prob=alpha_c / S, uncertainty=(self.K / S).squeeze(-1))
