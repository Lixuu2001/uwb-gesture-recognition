# Local CNN source-sixfold confirmation registered, not started by this record

The original local baseline's fixed source1criterion passes; the full-fusion and MultiRocket gates fail. Continue the eligible known baseline without rebranding either failed candidate. Code and exact protocol: [research/localcnn_source6_20261029](../../research/localcnn_source6_20261029/).

Reuse exact audited held1local checkpoint/predictions (no retraining), train only held users2–6 with val=held%6+1 and four training users from1–6. Same350220-parameter model/code hash, seed20261029,24epochs,batch32,3threads/channels_last,AdamW.001/decay.0005,unweightedCE and original cache/preprocessing. No source7–8labels or configuration grid.

Role/disjointness, validation-only tie selection and matching constructor/seed versus the reused initial model pass. All five new predictions freeze before aggregate scoring. Existing sixfold MiniRocket controls remain fixed. Descriptive gate:mean difference>=1pp,>=4/6wins,minuser difference>=-2pp; no independent/generalization/eightfold/innovation claim. Further research requires separately frozen protocols.

Observed source1local24epoch time6428.193s gives a five-fold estimate38569.160s with1.2margin. This passes12hwall/36CPU-hour allocation. Independent CPU/wall supervisor retains incomplete receipts and partial artifacts, without scientific-negative claims or automatic retries.

Final upstream artifacts are already privately backed up and authenticated-readback verified (149payload files,22chunks,156309271bytes). Originaldata/source789files unchanged. At this registration no new fitting directory exists. Actual launch must be verified separately; TimeMIL remains prepared/unlaunched while this confirmation has priority.
