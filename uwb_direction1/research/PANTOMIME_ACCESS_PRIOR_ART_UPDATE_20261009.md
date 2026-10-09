# Pantomime access and cross-sensor prior-art update — 2026-10-09

## Rights and archive metadata

The Pantomime dataset license gate is partly clarified by the dataset owner's institutional research portal: Aalto's record “Pantomime dataset and source” explicitly lists **CC-BY-4.0** under Dataset Licences and links to DOI 10.5281/zenodo.3772659. The later Zenodo record 10.5281/zenodo.4459969 describes the dataset and source archives, lists 107.6 MB of dataset ZIP and 937.5 kB source ZIP, but its rendered Rights → License field is blank. Treat the Aalto record as evidence that the dataset is offered under CC-BY-4.0, with attribution; do not infer a license for the separate source-code archive from that dataset license. The exact release/DOI relationship still needs confirmation before citing the Zenodo archive as the same licensed version.

The web preview of the source archive exposes a top-level file listing including `LICENSE` (1.2 kB) and `README.md` (7.6 kB); the source license text itself was not opened. The dataset archive preview request returned HTTP 429 and did not expose its member manifest. No archive was downloaded, extracted, or read; no samples or labels were accessed.

## Additional direct prior art

- Sukianto et al. (2026), IEEE Transactions on Radar Systems, DOI 10.1109/TRS.2026.3654128, is an author-institution record for cross-device **FMCW-to-FMCW** hand-gesture adaptation. It describes signal-level range/Doppler harmonization, radar-specific augmentation, a source-pretrained transformer, fine-tuning on a small labeled target set, distillation, and unsupervised federated refinement. This rules out broad claims of new cross-device FMCW harmonization or small-label radar adaptation. It does not evaluate IR-UWB/X4 against Pantomime point clouds.
- Tian et al. (2025), IEEE AIIoT, DOI 10.1109/AIIoT65859.2025.11105355, describes self-training cross-domain **FMCW air-writing** recognition with a small labeled target set. This further rules out generic self-training or FMCW cross-domain air-writing as a contribution. It is not the UWB-Gestures↔Pantomime dynamic swipe/circle endpoint.
- Xue et al. (2026), DOI 10.3389/fnins.2026.1864448, directly covers one/dual/tri-view recognition on the same UWB-Gestures benchmark; see the preceding branch report for detail.
- Gurbuz et al. (2020), DOI 10.1117/12.2559155, directly covers cross-frequency transfer involving X4 IR-UWB and FMCW radars for activity/gait recognition; see the preceding branch report for detail.

A bounded search found no exact published UWB-Gestures/X4 CIR → Pantomime mmWave point-cloud hand-gesture-family transfer match. This is **not** an exhaustive novelty clearance. The two-family metadata map (swipe/circular) remains only an endpoint design, not a method or contribution. Generic cross-sensor transfer, self-training, and modality adaptation are not novelty claims.

## Cloud execution gate

This session's current cloud policy is enforced and restricted: package-manager presets only, no configured capabilities, secrets, or outbound identities; Zenodo is not in the executor's allowed host list. The original UWB restore and 789-file source snapshot are present; process inspection found no active experiment (only old defunct Python entries). No CUDA device. No external archive/sample labels or project labels/arrays were read; no fit or training was started.

## Decision

Data reuse has an institutional CC-BY-4.0 indication, but release linkage and the data archive manifest remain unverified, and the executor cannot fetch Zenodo. Do not start transfer training. Next gate: confirm the exact CC-BY dataset release/record mapping and manifest through an allowed route; complete targeted direct-prior-art review of that exact endpoint; only then freeze direction, participant-disjoint folds, target-label budget (including zero-shot), family aggregation, and a representation before opening labels or fitting. Overall research goal remains unvalidated.

## Primary sources

- Aalto dataset portal: https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/
- Zenodo dataset/source record: https://zenodo.org/records/4459969
- Sukianto et al.: https://doi.org/10.1109/TRS.2026.3654128
- Tian et al.: https://doi.org/10.1109/AIIoT65859.2025.11105355
- Xue et al.: https://doi.org/10.3389/fnins.2026.1864448
- Gurbuz et al.: https://doi.org/10.1117/12.2559155
