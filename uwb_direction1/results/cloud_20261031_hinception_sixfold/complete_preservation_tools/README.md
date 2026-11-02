# Source-sixfold complete preservation preparation

This post-completion tool does not alter the frozen training/audit/controller
code, settings, source protocol, scores or current jobs. It has no model/array
imports, fitting, scoring, Drive or Git operations. Tests use synthetic byte
fixtures, not measured UWB results or trained modelsets.

The production builder first refuses active/incomplete controller state before
hashing checkpoints or creating a destination. It then requires the original
1465 pinned dependencies/readiness marker, all five successful train/audit
terminations, every full audit/state/prediction hash and global score boundary.
The prior held1 private complete archive and all its members/payload hashes must
also verify. These checks repeat after copying and before sealing the new archive.

After the controller finishes all five audits and aggregate:

```
python complete_modelset.py check
python complete_modelset.py build --out /workspace/uwb-hinception-sixfold-complete-20261031
```

Use an absent destination. No automatic retry/resume. The complete archive embeds
all five new initial/first/best/last/final states, selected/last native checkpoints,
all train/eval IDs/logs/predictions/audits/supervisor receipts, plus the entire
already verified held1 complete payload, pinned source/dependency text, source-ID
manifest and fixed comparator prediction/audit evidence. Model and actual sample
ID values remain private. Original raw/maps and runtime binaries are excluded;
restore those separately from their existing private/pinned records. This does
not claim whole-environment/job resume or independent/novel validation.

After a successful local build, an agent must check the owner-only Drive folder,
exact-name-search before uploading, and upload at most three chunk pipelines.
For each chunk check file metadata/owner-only permissions/parent/size/MIME before
`fetch(download_raw_file=true, include_base64=false)`, immediately materialize
and check size/SHA; never print authenticated URLs. Persist upload IDs before
fetch and inspect exact-name/receipts before any retry.

Write READBACK_INPUT.json in manifest order with each chunk's `name` and downloaded
local `path`, then run:

```
python complete_modelset.py verify --out /workspace/uwb-hinception-sixfold-complete-20261031
```

The verifier checks chunk sizes/SHA/order, assembled archive SHA, every regular
member's size/SHA, exact member set, duplicate/unsafe names and embedded/external
complete-scope agreement. A partial first-update manifest cannot produce a
complete-modelset proof. Only after that verification publish a persistent Drive
index/private receipt and final scientific summary. The local build alone is
not a private remote backup; no complete archive or readback is claimed now.

Do not call this while jobs run; the expected refusal is evidence of the guard,
not a scientific failure. Existing first-two-fold insurance remains valid but
is not a substitute for the finished six-model archive.
