# Private continuation and completion

Check STATUS, live processes, STATE, train/audit termination and checkpoint
files before starting anything. Never duplicate controller/run01/slots.
controller.py training→audit is sequential with independent CPU/wall limits.
It ends at complete_audited_unscored_requires_private_backup_and_finalization;
it does not claim complete private preservation or score the held user.

Initial insurance while training (only if initial/first exist):
`python backup.py initial_first --out /workspace/uwb-mlstmfcn-pilot-20261101/initial_first_backup`.
Its manifest scope is initial_first_only_not_latest_or_complete. It is not
latest weights, complete training logs or whole-job resume.

After training AND complete numerical audit AND controller success:
`python backup.py complete --out /workspace/uwb-mlstmfcn-pilot-20261101/complete_backup`.
It contains produced initial/first/best/last/final/custom and native models,
exact IDs/logs/predictions/full audit and code/dependency text/private original
author files. Raw/maps/runtime restore separately from pinned private indexes.
Original author code remains private, not redistributed to Git.

Upload each manifest part at most3parallel pipelines into owner-private
folder12AGMwT99yXCTYBQocQaX901Br-IelmOf. Metadata first; stored non-native
MIME may be inferred from arbitrary compressed chunk bytes. Use rawfetch
download_raw_file=true/include_base64=false; immediately materialize and
check size/SHA. Reject native/shortcut MIME, changed owner/parent/sharing.
Retain uploaded IDs/errors; search exact name before any unknown-ID retry.
Do not print authenticated URLs or change permissions.

Write ordered READBACK_INPUT.json rows{name,path} from materialized chunks,
then `python backup.py verify --out [same directory]`. This verifies archive
and every safe unique regular member, with explicit complete scope binding.
Only after complete scope succeeds, write DRIVE_BACKUP_RECEIPT.json with
state=private_readback_verified,complete_modelset_verified=true,archive_sha256
matching manifest/proof; persist fileIDs/size/SHA in CLOUD_DOWNLOAD_INDEX.json.
Then `python finalize.py` scores once; no refit/tuning. Publish real metadata,
report and memories from latest remote parent/force=false after exact readback.
Pass: separately preregister source6 confirmation. Fail: close exact method
configuration and next literature candidate. Do not use user7–8 labels.
