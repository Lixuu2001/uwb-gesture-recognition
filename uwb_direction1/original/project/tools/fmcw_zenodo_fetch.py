# -*- coding: utf-8 -*-
"""Selective downloader for the Frame-Labeled 60 GHz FMCW Radar Gesture Dataset.

The Zenodo record is one 36.5 GB zip; this tool fetches single members via HTTP
range requests so you never download the whole archive. Verified 2026-08-28
(user3_e1 / user10_e1_slow / user10_e1_fast, all CRC32-checked).

Usage (any python 3.8+, stdlib only):
    python tools/fmcw_zenodo_fetch.py list
    python tools/fmcw_zenodo_fetch.py fetch fulldataset/user3_e1.npz fmcw_frame_labeled

Members are ~650-750 MB compressed (2.4-2.5 GB npz). 16 parallel connections;
resume-safe (re-run continues incomplete chunks).
"""
import io
import os
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/records/15178095/files/radar_dataset.zip"
NTHREADS = 16


def fetch_range(start, end, retries=50):
    """Fetch bytes [start, end] inclusive."""
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"range fetch failed: {start}-{end}")


def total_size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def central_directory():
    """Parse the (ZIP64) central directory. Returns list of dicts."""
    size = total_size()
    tail = fetch_range(size - 65536, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found")
    # ZIP64 EOCD locator sits right before EOCD
    j = tail.rfind(b"PK\x06\x07", 0, i)
    if j >= 0:
        z64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        z64 = fetch_range(z64_off, z64_off + 55)
        assert z64[:4] == b"PK\x06\x06"
        cd_size, cd_off = struct.unpack("<QQ", z64[40:56])
    else:
        cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    cd = fetch_range(cd_off, cd_off + cd_size - 1)
    entries = []
    p = 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, crc, csize, usize, nlen, elen, clen, lho) = (
            struct.unpack("<H", cd[p + 10:p + 12])[0],
            struct.unpack("<I", cd[p + 16:p + 20])[0],
            struct.unpack("<I", cd[p + 20:p + 24])[0],
            struct.unpack("<I", cd[p + 24:p + 28])[0],
            struct.unpack("<H", cd[p + 28:p + 30])[0],
            struct.unpack("<H", cd[p + 30:p + 32])[0],
            struct.unpack("<H", cd[p + 32:p + 34])[0],
            struct.unpack("<I", cd[p + 42:p + 46])[0],
        )
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        # ZIP64 extra field overrides 0xFFFFFFFF placeholders, field order fixed
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                body = extra[q + 4:q + 4 + hsz]
                b = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", body[b:b + 8])[0]
                    b += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", body[b:b + 8])[0]
                    b += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", body[b:b + 8])[0]
                    b += 8
            q += 4 + hsz
        entries.append(dict(name=name, method=method, crc=crc,
                            csize=csize, usize=usize, lho=lho))
        p += 46 + nlen + elen + clen
    return entries


def fetch_member(name, outdir):
    ent = next((e for e in central_directory() if e["name"] == name), None)
    if ent is None:
        raise SystemExit(f"member not found: {name}")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(name).replace(".npz", "")
    dst = os.path.join(outdir, base + ".npz")
    if os.path.exists(dst):
        print("already exists:", dst)
        return dst
    # local header gives the true data offset (its extra field length differs
    # from the central one)
    lh = fetch_range(ent["lho"], ent["lho"] + 29)
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_start = ent["lho"] + 30 + nlen + elen
    csz = ent["csize"]
    chunk = (csz + NTHREADS - 1) // NTHREADS
    partdir = os.path.join(outdir, "parts_" + base)
    os.makedirs(partdir, exist_ok=True)

    def get_chunk(i):
        cs = data_start + i * chunk
        ce = min(cs + chunk - 1, data_start + csz - 1)
        want = ce - cs + 1
        p = os.path.join(partdir, f"part_{i:02d}")
        have = os.path.getsize(p) if os.path.exists(p) else 0
        while have < want:
            blob = fetch_range(cs + have, ce)
            with open(p, "ab") as f:
                f.write(blob)
            have += len(blob)
        return want

    t0 = time.time()
    with ThreadPoolExecutor(NTHREADS) as ex:
        list(ex.map(get_chunk, range(NTHREADS)))
    print(f"downloaded {csz/1e6:.0f} MB in {time.time()-t0:.0f}s; inflating...")
    d = zlib.decompressobj(-15) if ent["method"] == 8 else None
    crc = 0
    with open(dst, "wb") as fout:
        for i in range(NTHREADS):
            with open(os.path.join(partdir, f"part_{i:02d}"), "rb") as f:
                while True:
                    blk = f.read(1 << 24)
                    if not blk:
                        break
                    out = d.decompress(blk) if d else blk
                    crc = zlib.crc32(out, crc)
                    fout.write(out)
        if d:
            out = d.flush()
            crc = zlib.crc32(out, crc)
            fout.write(out)
    if crc != ent["crc"]:
        os.remove(dst)
        raise SystemExit(f"CRC MISMATCH {crc:08x} != {ent['crc']:08x} - removed output; "
                         f"delete {partdir} and retry")
    for i in range(NTHREADS):
        os.remove(os.path.join(partdir, f"part_{i:02d}"))
    os.rmdir(partdir)
    print(f"OK {dst} ({ent['usize']/1e9:.2f} GB), crc32 verified {crc:08x}")
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "fetch"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "list":
        for e in central_directory():
            print(f"{e['name']:45s} {e['usize']/1e6:9.1f} MB raw "
                  f"{e['csize']/1e6:8.1f} MB compressed  crc={e['crc']:08x}")
    else:
        fetch_member(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "fmcw_frame_labeled")
