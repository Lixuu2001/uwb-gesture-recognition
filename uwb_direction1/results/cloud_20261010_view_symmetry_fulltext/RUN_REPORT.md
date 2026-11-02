# Full-text evidence on left/right view asymmetry and mirror-symmetry gate

**Observed:** 2026-10-10T22:11:27Z  
**Branch parent:** b26ac088b028c1fab2bbf79577717008e2b25d6a

## Literature finding

I read the publisher-hosted full text of Xue, Yang, and Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” *Frontiers in Neuroscience* (2026), DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448), [full text](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full). The article uses the same UWB-Gestures corpus and three XeThru X4 viewpoints (left, top, right). It already reports single-, dual-, and tri-view fusion under a 6-subject/2-subject split; that prior art preempts generic multi-view fusion and a view-subset study as a method claim.

The full text also discusses the left/right performance asymmetry and attributes it as potentially arising from asymmetric gesture execution, view-dependent scattering, and self-occlusion. Those plausible explanations mean the published data do not support treating left/right views as exact physical mirrors or assuming a label-equivariant sensor swap. This is a caution against inferring an augmentation map from the view names alone. It does not prove that every carefully defined symmetry is impossible; it leaves the exact mapping unverified.

Prior-art leads for physical FMCW/multistatic radar gesture augmentation were already documented, but their full texts remain unreviewed. Accordingly, generic mirror/physical augmentation is still not novelty-cleared. The present evidence does not justify an experiment.

## Cloud preflight

The managed executor is available in this turn. A fresh process scan found no external UWB training process. PyTorch is 2.14.0+cpu, CUDA is unavailable, and five CPUs are reported. No X4/optical acquisition device nodes are present. There are historical model-weight files from prior probes, but no active training run; none were loaded or changed. The restored dataset directory is present. The local source checkout exposes 31 project files; the remote branch continues to hold the 789-file original snapshot and was not modified.

No project labels or radar-array values were read. No fit, training, tuning, checkpoint selection, or data restoration ran. The user's research target remains unvalidated.

## Disposition

Do not launch generic mirror augmentation, sensor permutation, or held-view training. Continue screening only distinct, literature-supported candidates that can be evaluated from the restored clutter-removed maps. A physical-invariance claim still requires acquisition/calibration and synchronized motion-truth capabilities absent from this executor. Do not repeat blocked IEEE/DOI/UESTC queries unless a new lawful full-text route appears.
