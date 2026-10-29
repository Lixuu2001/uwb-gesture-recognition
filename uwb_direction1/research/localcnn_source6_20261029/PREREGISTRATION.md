# Fixed local CNN baseline: source-sixfold confirmation

This study follows the already frozen positive local-baseline expansion criterion. It is a confirmation of a useful known baseline, not a new fusion mechanism, original contribution or independent validation. The full24epoch fusion configuration and fixed MultiRocket configuration failed their own gates and are not expanded or tuned here. TimeMIL remains separately preregistered and unlaunched; finish this confirmation before considering its explicit launch.

## Evidence authorizing expansion

The completed source-user1pilot (train3–6,val2,held1) scored96.8333% local versus96.0000% full and85.3333% original MiniRocket. The local>=MiniRocket+1pp criterion passes.93independent checks replayed both CNNs' selected validation and diagnostic outputs and verified artifact/role/optimizer state. Full-minus-local=-0.8333pp fails the full-fusion gate; that exact24epoch full configuration is closed. It is not a result about the untested100epoch author setting.

The separately completed full49,728-feature MultiRocket pilot scored94.4167% versus85.0000% matched direct-ridge MiniRocket, but was2.4167pp below the local CNN and failed the registered noninferiority condition. That exact configuration is closed, despite its meaningful gain over MiniRocket. Its verifier checked all16artifact hashes, readout moments/weights/objective,2200logits per arm, independent metrics, and six fixed feature-extraction records per arm. The original85.3333% MiniRocket used bounded LSQR; the new paired85.0000% used the preregistered direct readout. Do not mix these controls or select a preferred result.

Both studies are completed without errors. The controller exited after its entire registered queue and audits. Final CNN/MultiRocket/control artifacts were privately backed up as149members,22owner-only chunks,156309271compressedbytes; all chunks, remote reassembly and every member SHA/size verified. The archive excludes raw radar arrays and large regenerable prepared/features arrays; original data remain in their separately verified private package.

## Frozen protocol and reuse

Use only the existing6900quality-retained IDs from users1–6. In each fold, held userh, validationh%6+1, remaining four users train. No7–8labels. Role counts are explicit in PROTOCOL.json and were checked using metadata only. As ordinary crossvalidation, a source user appears in other folds' training/validation; its labels never enter fitting or checkpoint selection for its own held fold. All source users were previously developed, so this is descriptive/exploratory, with no independence/significance claim.

Reuse the exact audited held1local checkpoint selected at epoch23 and its frozen predictions; do not rerun it. Train only five missing folds2–6,24epochs each,120new epochs total. Same model.py SHA,350220parameters, seed20261029, batch32,3threads/channels_last, AdamW lr0.001/weight_decay0.0005, unweighted CE, no augmentation/scheduler/smoothing, same per-epoch seeded order. Use the existing source-only6900x3x64x128cache with its fixed SHA; preprocessing remains signed maps, bilinear resize and per-record/view RMS. No new preprocessing or architecture choice.

Choose each checkpoint by current validation-user present-class macro recall, lower CE, earlier epoch. Save complete optimizer/RNG, initial/first-update/best/last checkpoints, IDs/splits, epoch logs and unscored predictions. Freeze all five new held prediction arrays before aggregate scoring. Existing sixfold9996-feature MiniRocket transforms/readouts/predictions are fixed comparators, with no refitting/reselection. This does not claim to beat every strong method.

Report per-user local/MiniRocket scores and paired differences, equal-user mean, worst-user score, and class recalls if needed. A descriptive continuation condition is mean difference>=1pp,>=4/6positive users and no user's loss>2pp. This licenses further study only, not independent generalization, eightfold improvement or originality. No seed/model/epoch/readout grid follows a failure. Old eightfold93.4417% is not directly comparable to this four-train-user source protocol.

## Resource and engineering bounds

Observed local24epoch time6428.193seconds yields a five-new-fold estimate38569.160seconds with1.2margin; all new training folds contain<=4700records. The12hour wall/36CPU-hour,3thread allocation covers this estimate. The independent supervisor caps wall43100seconds plus termination grace, process CPU129400soft/129500hard within129600. Existing tested timeout/terminal-receipt helper is SHA-bound. Engineering/budget errors retain partial artifacts and incomplete state, never a scientific-negative claim.

Metadata-role/disjointness, validation-only tie selection, constructor/seed identity with the reused initial model, and syntax checks pass. No source2–6new CNN fitting or diagnostic scoring occurred before registration. The runner is one-shot and refuses existing fitting output; it does not restart automatically. Retain and account for actual consumed budgets before any explicit engineering recovery, with no new statistical trial or resampling.

Run only after verifying frozen hashes and observing no conflicting active job:

```bash
python /workspace/uwb-localcnn-sixfold-20261029/launch.py --run
```

The eventual full sixfold results need independent checkpoint/functional/metric verification before reporting gains. Preserve new models privately, reference the already verified held1archive, and keep raw arrays/tensors/IDs outsideGit. The user's research-contribution goal remains unresolved; published-method accuracy work continues under their latest instruction.
