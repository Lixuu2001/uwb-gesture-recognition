# Temporal-reversal label-transform candidate: prior-art gate

**Date:** 2026-10-28 11:13 UTC  
**Decision:** do not train or claim novelty for this candidate.

## Candidate screened

For the existing clutter-removed range-time maps, reverse the 90-frame slow-time sequence and map the label to the gesture with the opposite ordered motion: LR↔RL, UD↔DU, each diagonal swipe to its endpoint-reversed partner, and CW↔CCW. The inward push and no-gesture labels are self-mapped. Training would apply this transform only to source-user clips, with all 8 users held out in turn and model selection confined to the existing source-user validation protocol. It would be implementable from the current real-valued maps; no raw complex echoes or synchronized hand-motion truth are required for the transform itself.

## Literature finding and narrow distinction

The core operation is already established outside radar: Molchanov et al., *Hand Gesture Recognition with 3D Convolutional Neural Networks* (CVPR Workshops 2015), explicitly reverse frame order and switch a left-swipe label to right-swipe. This is the same mathematical label-covariant temporal transform, so applying it to radar maps or to this dataset’s inverse-direction pairs is a modality/dataset transfer of a known augmentation, not a sufficiently distinct method contribution. A primary-source search result also identifies Regani et al., *Time Reversal Based Robust Gesture Recognition Using WiFi* (ICASSP 2020), as radar-adjacent wireless gesture time-reversal work. The UWB-Gestures dataset paper defines the ordered directional and CW/CCW class pairs, but does not by itself establish a new learning mechanism.

Bounded source search was conducted on 2026-10-28. The official CVF PDF endpoint rejected direct retrieval (403), and the ICASSP PDF endpoint timed out; indexed primary-source excerpts exposed the relevant operation. This is enough to reject generic novelty, but not a complete systematic review. Sources:

- Molchanov et al. (2015), CVF primary source: https://openaccess.thecvf.com/content_cvpr_workshops_2015/W15/papers/Molchanov_Hand_Gesture_Recognition_2015_CVPR_paper.pdf
- Ahmed et al. (2021), UWB-Gestures dataset descriptor: https://www.nature.com/articles/s41597-021-00876-0
- Regani et al. (2020), ICASSP primary-source PDF: https://dihana.cps.unizar.es/proceedings/ICASSP/2020/pdfs/0008304.pdf

## Decision and next step

No arrays, label values, or prediction outputs were read. No training, fitting, tuning, or data download occurred. Do not run an eight-fold experiment to rescue a candidate whose central operation is preempted. This screen does not show that temporal reversal is ineffective; it says that an improvement on this corpus would not, by itself, establish a novel method.

The next eligible step remains a bounded, method-first screen for a materially different, physically justified mechanism that is directly observable in the current clutter-removed input. If such a mechanism cannot be identified and distinguished from prior art, preserve the completed reproducibility audit as an engineering/reproducibility result without representing it as the requested method contribution. No candidate currently authorizes training.
