# -*- coding: utf-8 -*-
"""Inflate a variant member and analyze its label durations. Usage: analyze_variant.py NAME EXPECTED_CRC"""
import zlib, sys, os
import numpy as np

NAME = sys.argv[1]
EXP_CRC = int(sys.argv[2], 16) if len(sys.argv) > 2 else None
BASE = r"D:\uwb_gestures\fmcw_frame_labeled"
SRC = os.path.join(BASE, NAME + ".deflate")
DST = os.path.join(BASE, NAME + ".npz")

if not os.path.exists(DST):
    d = zlib.decompressobj(-15)
    crc = 0
    with open(SRC, "rb") as fin, open(DST, "wb") as fout:
        while True:
            chunk = fin.read(1 << 24)
            if not chunk:
                break
            out = d.decompress(chunk)
            crc = zlib.crc32(out, crc)
            fout.write(out)
        out = d.flush()
        crc = zlib.crc32(out, crc)
        fout.write(out)
    print("inflated:", os.path.getsize(DST) / 1e9, "GB", "crc=%08x" % crc,
          "MATCH=%s" % (EXP_CRC is None or crc == EXP_CRC))
    if EXP_CRC is not None and crc != EXP_CRC:
        os.remove(DST)
        sys.exit(1)

z = np.load(DST, allow_pickle=True)
for k in z.keys():
    print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")
lab = np.asarray(z["targets"])
vals, counts = np.unique(lab, return_counts=True)
print("labels:", {int(v): int(c) for v, c in zip(vals, counts)})

def runs(seq):
    out = []
    s = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((seq[s], s, i - 1))
            s = i
    return out

bg = vals[np.argmax(counts)]
print("assumed background:", int(bg))
gl, on, off, nseg = [], [], [], []
nr = {}
for r in range(lab.shape[0]):
    rr = runs(lab[r])
    nr[len(rr)] = nr.get(len(rr), 0) + 1
    g = [x for x in rr if x[0] != bg]
    nseg.append(len(g))
    for v, s, e in g:
        gl.append(e - s + 1)
        on.append(s)
        off.append(e)
gl = np.array(gl)
print("runs/rec hist:", dict(sorted(nr.items())))
print("segments/rec hist:", dict(sorted(zip(*np.unique(nseg, return_counts=True)))))
if len(gl):
    print(f"gesture len (frames): n={len(gl)} mean={gl.mean():.1f} median={np.median(gl):.0f} "
          f"p10={np.percentile(gl,10):.0f} p90={np.percentile(gl,90):.0f} min={gl.min()} max={gl.max()}")
    print(f"  ms: mean={gl.mean()/33*1000:.0f} p10={np.percentile(gl,10)/33*1000:.0f} "
          f"p90={np.percentile(gl,90)/33*1000:.0f} min={gl.min()/33*1000:.0f} max={gl.max()/33*1000:.0f}")
    on = np.array(on)
    print(f"onset: mean={on.mean():.1f} min={on.min()} max={on.max()}")
    # unique length histogram (to detect fixed-width labeling)
    lv, lc = np.unique(gl, return_counts=True)
    print("length histogram:", {int(a): int(b) for a, b in zip(lv, lc)})
print("\nexamples:")
for r in range(3):
    print(f"  rec{r}:", [(int(v), s, e) for v, s, e in runs(lab[r])])
