# Local CNN source-sixfold confirmation actually running

Preregistration commit1b8ffb10393682478718f8dee15f3c8c94528084 preceded launch. Current actual worker43382/supervisor43381 are alive and command/state verified at 2026-10-29T16:16:54.454778+00:00, training held2/validation3 (train1,4,5,6), epoch1. No old job was restarted and held1 remains reused.

Independent first-update check passes:67model tensors changed from the initial model, all floating tensors finite, every optimizer state atstep1,32unique training-only IDs and no current-held/validation IDs. Checkpoint SHA256:`a9bbf3b5df24b21064614cb7a5ad53cdcafa034819a498dd7e4e1638ab181cb7`; protocol SHA256:`b3633e607fc27365e6e7ab487ac6d0161b12cd112124bdf04a203236ca53d1df`. Evidence: [FIRST_UPDATE_VERIFIED.json](FIRST_UPDATE_VERIFIED.json), [LAUNCH.json](LAUNCH.json). Fitted tensors and IDs stay private outsideGit.

The queue trains only five missing folds2–6,24epochs each, with the unchanged seed/model/cache/optimizer/validation selection. All five predictions must freeze before aggregate scoring. No new fold accuracy scores have been observed. No7–8labels.12hwall/36CPU-hour budget with independent supervisor; incomplete on engineering/budget failure, no automatic statistical retry.

Observe /workspace/uwb-localcnn-sixfold-20261029/run01/STATE.json and launch.log plus checkpoints. Do not duplicate a living worker or launch TimeMIL alongside this confirmation. After completion, verify all checkpoint selection, roles, functional validation/diagnostic predictions and aggregate metrics before reporting gain; preserve new states privately. Existing upstream final models/control states already have the verified22chunk/149payload private recovery package.

The previously completed source1pilot is local96.8333%,full96.0000%; matched MultiRocket94.4167%versusMini85.0000%, with both full-fusion/MultiRocket gates failed. Only the separately eligible local baseline is being expanded. This is source-sixfold exploratory baseline confirmation, not old-eightfold improvement, independent validation or a research-contribution claim. TimeMIL's separate64epoch configuration remains frozen/prepared/unlaunched.
