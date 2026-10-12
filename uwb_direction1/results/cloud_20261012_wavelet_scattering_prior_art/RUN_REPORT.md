# Wavelet-scattering same-corpus prior-art gate

Date: 2026-10-12 UTC

## Finding

A targeted search located Al-Jumaily and Khushaba, “Deep Hand Gesture Recognition: A Wavelet Scattering Alternative to Convolutional Networks,” 2023 IEEE Statistical Signal Processing Workshop, DOI 10.1109/SSP53291.2023.10208011. The IEEE publisher record and the authors' university repository abstract both state that the method compares wavelet-scattering features plus lightweight conventional classifiers against DCNNs on a public UWB impulse-radar gesture database with eight participants and twelve gestures, reporting 95% average accuracy across subjects.

This is the same UWB-Gestures corpus used here, so generic wavelet-scattering features or “lightweight scattering instead of CNN” are already prior art for this data. The accessible primary-source material was the publisher record and author-affiliated repository abstract; the paper is subscription-only and its full method, exact preprocessing, split construction, per-user results, and compute measurements were not accessible in this review. The reported 95% therefore cannot be treated as comparable to the project's eight-fold held-user macro score.

## Decision

- Close generic wavelet-scattering replacement of a CNN as a novelty claim for this corpus.
- Do not start a WST grid or claim the published result has been reproduced.
- A carefully frozen WST implementation could still be a named literature baseline, but that alone would not satisfy the project's original-method goal. Exact reproduction also requires the paper's method details and a protocol-compatible input.
- No training, fitting, project-label access, tensor-value access, or checkpoint access occurred in this review.

## Cloud preflight

- Restored dataset directory: present at `/workspace/uwb-restored-20260930`; it was not downloaded again.
- Process inspection: no active UWB training process; only historical defunct Python entries.
- Runtime: Python 3.12.14, PyTorch 2.14.0+cpu, five CPUs, no CUDA.
- Workspace free space: 746,266,624 bytes. The separately screened 36.5 GB FMCW archive cannot be staged here.
- Network policy: restricted/enforced. No policy bypass or external dataset access was attempted.

## Sources reviewed

- IEEE publisher record: https://ieeexplore.ieee.org/abstract/document/10208011
- Author-affiliated ECU repository record and abstract: https://ro.ecu.edu.au/ecuworks2022-2026/3131/
- Dataset descriptor: Ahmed et al., *UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors*, Scientific Data (2021), DOI 10.1038/s41597-021-00876-0, https://www.nature.com/articles/s41597-021-00876-0

## Reproducibility

This is a bounded prior-art review, not an experiment. It did not read the project labels or array values, create a model, tune parameters, or alter the restored data. The literature search is not exhaustive, and no claim is made that every possible UWB signal-specific contribution has been excluded.
