# Completed full-scale MultiRocket source pilot: gain over MiniRocket, continuation gate fails

Unedited maintained aeon1.6.0 corrected multivariate transform,49,728features/seed20261029, train3–6/val2/diagnostic1, all567range/view channels and90frames. Both transforms use the same preregistered balanced alpha1 float64 dual-ridge readout. No parameter/seed/grid selection.

| Method | Held-source-user macro recall |
|---|---:|
| Full MultiRocket |94.4167%|
| Matched MiniRocket with direct ridge readout |85.0000%|
| Audited local CNN reference |96.8333%|
| Audited full CNN reference |96.0000%|

MultiRocket-minus-matchedMiniRocket=+9.4167pp, a genuine known-baseline gain. MultiRocket-minus-localCNN=-2.4167pp, failing the registered requirement to be within0.5pp of the stronger CNN. Joint continuation gate=false. Close this exact configuration without feature/alpha/seed or other rescue grids; this is not a statement that all MultiRocket settings or time-series transforms fail.

The CNN pilot's original MiniRocket score85.3333% used its previously frozen bounded LSQR readout. The new paired85.0000% uses the fixed direct readout prescribed before results. They are distinct declared controls; use each only in its own comparison. No outcome-based solver selection occurred here.

Fit wall160.186s/CPU159.629s; full supervised process wall164.660s/CPU164.091s including startup, within1800s. Independent result audit passes: all16artifact hashes, training-only scaling/weights, ridge objective residuals1.20e-14 and2.48e-13, all2200validation/diagnostic logits per arm, independent macro metrics/gate. Extraction replay covers only six fixed records per arm, not all6900. No refit/resampling was performed by that audit.

See [SUMMARY.json](SUMMARY.json), [COMPLETE_RESULT_AUDIT.json](COMPLETE_RESULT_AUDIT.json), [TERMINATION.json](TERMINATION.json) and [CONTINUATION_COMPLETE.json](CONTINUATION_COMPLETE.json). Entire registered CNN→audit→MultiRocket→audit queue completed successfully and exited; zombie processes are not running training. No users7–8labels used. These repeatedly developed source-user results are exploratory, not independent validation/eightfold results or originality evidence. No real TimeMIL fit has started.

All final fitted transforms/readouts/IDs/logs/model states are privately preserved with chunk/archive/member hash verification; shared recovery index is ../cloud_20261029_mvglf_pilot/FINAL_QUEUE_DRIVE_BACKUP_RECEIPT.json. Raw radar tensors and large regenerable maps/features remain in their separate original/private cloud storage, outsideGit.
