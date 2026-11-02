# Pantomime rights gate clarified; transfer remains blocked — 2026-10-09

## Work performed

- Re-read the branch status, continuation, project memory, and latest cross-sensor report.
- Rechecked managed-cloud readiness and local process/data presence.
- Reviewed the official Aalto research portal and Zenodo Pantomime records, source-archive preview, and primary institutional/publisher records for recent cross-domain radar gesture work.
- Did not fetch the Pantomime archives or read their contents/data, labels, or any UWB labels/arrays.

## Findings

Aalto's institutional Pantomime dataset record explicitly lists CC-BY-4.0 for the dataset. The newer Zenodo dataset-and-source page leaves its visible license field blank; the record/version relationship and dataset-archive manifest remain to verify. The source ZIP preview exposes a separate LICENSE file, but its text was not opened. Thus the dataset has positive licensing evidence, while the exact archive linkage/manifest and source-code license remain unresolved.

Additional prior art closes off generic FMCW cross-domain/self-training claims: Sukianto et al. (2026) cover signal harmonization and few-label cross-device FMCW gesture transfer; Tian et al. (2025) cover self-training cross-domain FMCW air-writing. Together with Xue et al. (2026) on same-benchmark UWB multiview fusion and Gurbuz et al. (2020) on X4-to-FMCW cross-frequency transfer, these do not establish the exact X4 CIR → Pantomime point-cloud family-transfer task, but that narrow task is still only an unevaluated endpoint and has not cleared exhaustive prior-art review.

Current executor is CPU-only with enforced restricted egress, no configured external capabilities, secrets, or outbound identities; Zenodo is not allowed for shell access. UWB data and the 789-file source snapshot are present, and no active training process was found. No model was fit and no training was started.

## Decision

No candidate clears both method novelty and execution feasibility. Do not rerun the already-preempted same-dataset multi-view experiments or generic transfer/self-training grids. Continue with a targeted exact-endpoint prior-art review and, through an allowed route, confirm the licensed release/version and dataset manifest. Only after those gates pass should a participant-disjoint protocol and representation be frozen. No contribution or score is claimed; the user's goal remains unvalidated.

## Sources

- Aalto Pantomime dataset portal: https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/
- Zenodo Pantomime record: https://zenodo.org/records/4459969
- Sukianto et al., DOI 10.1109/TRS.2026.3654128
- Tian et al., DOI 10.1109/AIIoT65859.2025.11105355
- Xue et al., DOI 10.3389/fnins.2026.1864448
- Gurbuz et al., DOI 10.1117/12.2559155
