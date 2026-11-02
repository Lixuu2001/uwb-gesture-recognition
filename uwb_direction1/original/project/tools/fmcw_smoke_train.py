# -*- coding: utf-8 -*-
"""Smoke test: can a small causal per-frame model learn frame-level gesture
segmentation on the FMCW frame-labeled dataset (user3_e1 sample)?

NOT a performance claim: single user, within-user split (700/300 recordings).
Purpose: verify (a) labels are learnable per-frame, (b) event decoding works on
concatenated streams, (c) per-frame inference is real-time capable.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

BASE = r"D:\uwb_gestures\fmcw_frame_labeled"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
torch.manual_seed(0)

z = np.load(os.path.join(BASE, "user3_e1.npz"), allow_pickle=True)
X = z["inputs"]  # (1000, 100, 3, 32, 64) float32
LK = [k for k in z.keys() if k != "inputs"]
lab_raw = np.asarray(z[LK[0]])
print("label key:", LK[0], lab_raw.shape, lab_raw.dtype)

# map labels to ints, background = 0
vals, counts = np.unique(lab_raw, return_counts=True)
bg_val = vals[np.argmax(counts)]
classes = [bg_val] + [v for v in vals if v != bg_val]
cls2id = {v: i for i, v in enumerate(classes)}
Y = np.vectorize(cls2id.get)(lab_raw).astype(np.int64)  # (1000, 100)
print("classes:", {str(v): i for v, i in cls2id.items()})

# ---- features: per-frame range-Doppler magnitude, mean over antennas ----
# X frame: (3, 32, 64) = antennas x chirps x samples
def rd_features(x):
    """x: (N, 100, 3, 32, 64) -> (N, 100, 32, 32) log-magnitude RD, antenna-mean"""
    x = x - x.mean(axis=-1, keepdims=True)          # remove DC per chirp
    r = np.fft.rfft(x, axis=-1)[..., 1:33]          # range bins 1..32 (drop DC bin)
    d = np.fft.fft(r, axis=-2)                      # Doppler over 32 chirps
    d = np.fft.fftshift(d, axes=-2)
    m = np.abs(d).mean(axis=2)                      # mean over antennas -> (N,100,32,32)
    return np.log1p(m).astype(np.float32)

t0 = time.time()
F = np.empty((X.shape[0], 100, 32, 32), dtype=np.float32)
B = 50
for i in range(0, X.shape[0], B):
    F[i:i+B] = rd_features(X[i:i+B])
print(f"RD features: {F.shape} in {time.time()-t0:.1f}s")

# causal per-frame normalization (running stats would be ideal; here use global
# train-set stats which do not leak future *within* a stream)
idx = rng.permutation(X.shape[0])
tr, te = idx[:700], idx[700:]
mu, sd = F[tr].mean(), F[tr].std()
F = (F - mu) / (sd + 1e-6)

class FrameEnc(nn.Module):
    def __init__(self, emb=96):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),   # 16x16
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),  # 8x8
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # 4x4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, emb), nn.ReLU())
    def forward(self, x):  # (B*T, 1, 32, 32)
        return self.c(x)

class CausalSeg(nn.Module):
    def __init__(self, emb=96, hid=128, ncls=6):
        super().__init__()
        self.enc = FrameEnc(emb)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, ncls)
    def forward(self, x):  # (B, T, 32, 32)
        Bs, T = x.shape[:2]
        e = self.enc(x.reshape(Bs * T, 1, 32, 32)).reshape(Bs, T, -1)
        h, _ = self.gru(e)
        return self.head(h)  # (B, T, ncls)

model = CausalSeg(ncls=len(classes)).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
# inverse-frequency class weights (background dominates 92:1.6)
cnt = np.bincount(Y[tr].reshape(-1), minlength=len(classes)).astype(np.float64)
w = (cnt.sum() / (len(classes) * cnt)) ** 0.75
w = w / w.mean()
print("class weights:", np.round(w, 3))
lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEV))

Ft = torch.from_numpy(F)
Yt = torch.from_numpy(Y)
EPOCHS = 30
BS = 32
for ep in range(EPOCHS):
    model.train()
    perm = rng.permutation(len(tr))
    tot = 0.0
    for i in range(0, len(tr), BS):
        b = tr[perm[i:i+BS]]
        xb = Ft[b].to(DEV)
        yb = Yt[b].to(DEV)
        opt.zero_grad()
        out = model(xb)
        loss = lossf(out.reshape(-1, out.shape[-1]), yb.reshape(-1))
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    # quick eval each 3 epochs
    if ep % 3 == 2 or ep == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(te), BS):
                b = te[i:i+BS]
                preds.append(model(Ft[b].to(DEV)).argmax(-1).cpu())
            P = torch.cat(preds).numpy()
        Yte = Y[te]
        acc = (P == Yte).mean()
        # macro frame F1
        f1s = []
        for c in range(len(classes)):
            tp = ((P == c) & (Yte == c)).sum()
            fp = ((P == c) & (Yte != c)).sum()
            fn = ((P != c) & (Yte == c)).sum()
            f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
        print(f"ep{ep+1} loss={tot/len(tr):.4f} frame_acc={acc:.4f} "
              f"macroF1={np.mean(f1s):.4f} perclass={[f'{v:.3f}' for v in f1s]}")

# ---- event-level eval on a concatenated pseudo-stream of test recordings ----
def to_events(seq, min_len=3):
    ev = []
    s = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            if seq[s] != 0 and (i - s) >= min_len:
                ev.append((int(seq[s]), s, i - 1))
            s = i
    return ev

model.eval()
stream_ids = te[:100]
with torch.no_grad():
    logits = []
    for i in range(0, len(stream_ids), BS):
        b = stream_ids[i:i+BS]
        logits.append(model(Ft[b].to(DEV)).cpu())
    L = torch.cat(logits)                       # (100, 100, C)
Pstream = L.argmax(-1).numpy().reshape(-1)      # concatenated prediction
Ystream = Y[stream_ids].reshape(-1)
pe, ge = to_events(Pstream), to_events(Ystream)

def match(pe, ge, thr=0.5):
    used = set()
    tp = 0
    on_err, off_err = [], []
    for c, s, e in pe:
        best, bi = 0.0, None
        for j, (c2, s2, e2) in enumerate(ge):
            if j in used or c2 != c:
                continue
            inter = max(0, min(e, e2) - max(s, s2) + 1)
            union = max(e, e2) - min(s, s2) + 1
            iou = inter / union
            if iou > best:
                best, bi = iou, j
        if best >= thr:
            used.add(bi)
            tp += 1
            on_err.append(abs(s - ge[bi][1]))
            off_err.append(abs(e - ge[bi][2]))
    prec = tp / max(1, len(pe))
    rec = tp / max(1, len(ge))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1, prec, rec, np.mean(on_err) if on_err else -1, np.mean(off_err) if off_err else -1

for thr in (0.1, 0.25, 0.5):
    f1, p, r, oe, fe = match(pe, ge, thr)
    print(f"event F1@{thr}: {f1:.3f} (P={p:.3f} R={r:.3f}) onsetMAE={oe:.1f}f offsetMAE={fe:.1f}f")
print(f"events: pred={len(pe)} true={len(ge)}  stream={len(Pstream)} frames "
      f"({len(Pstream)/33/60:.1f} min)")

# ---- per-frame inference speed (streaming: encoder+GRU step per frame) ----
model_cpu = CausalSeg(ncls=len(classes))
model_cpu.load_state_dict(model.state_dict())
model_cpu.eval()
x1 = torch.randn(1, 1, 32, 32)
h = torch.zeros(1, 1, 128)
with torch.no_grad():
    t0 = time.time()
    for _ in range(330):
        e = model_cpu.enc(x1).unsqueeze(1)
        o, h = model_cpu.gru(e, h)
        _ = model_cpu.head(o)
    dt = (time.time() - t0) / 330 * 1000
print(f"CPU per-frame streaming inference: {dt:.2f} ms/frame (budget 30.3 ms @33Hz)")
