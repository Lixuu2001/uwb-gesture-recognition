# Range-resolved slow-time spectrum novelty and observability gate

Date: 2026-10-12 17:26 UTC  
Branch head at review: `b05a9623f47b2ac52c11ce55f3403838651a02af`  
Review type: bounded primary-source review of one candidate; no project array values, labels, checkpoints, fit, or training were accessed.

## Candidate screened

Candidate: form a slow-time spectrum independently at each range bin of the available clutter-removed time-range maps, then classify the range-resolved envelope spectra using one radar view (or compare the same representation across the three fixed views).

This is computable from the stored real-valued tensor in principle. It is not equivalent to recovering coherent Doppler from pulse-level complex returns. The cache inventory records only (X_{clutterremoved}), shape (9593, 3, 90, 189), float32; the dataset descriptor says 90 slow-time rows represent 4.5 seconds (20 frames/s). Therefore a temporal spectrum of the stored envelope has a 10 Hz Nyquist limit, and its bins describe amplitude-envelope modulation in this sampled representation. Without the complex pulse sequence, it cannot substantiate phase-derived micro-Doppler or resolve components above that limit.

## Primary-source review

- Ritchie & Jones, “Micro-Doppler Gesture Recognition using Doppler, Time and Range Based Features,” IEEE RadarConf 2019, DOI [10.1109/RADAR.2019.8835782](https://doi.org/10.1109/RADAR.2019.8835782). The accepted manuscript was read. It explicitly generates a spectrogram per range bin and retains range as a gesture-classification dimension. Its data are 24 GHz FMCW, so this is adjacent rather than an exact UWB reproduction; nevertheless, a generic “range-resolved Doppler/time-frequency gesture representation” is not a cleared contribution.
- Ahmed et al., “UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors,” Scientific Data 2021, DOI [10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0). The primary data descriptor identifies 20 frames/s and 90×189 time/range maps over 4.5 s, with three independent monostatic X4 views. This bounds what the stored temporal sampling can identify; it supplies no phase or independent motion ground truth for the present cache.
- Xue, Yang & Zhu, “Neuromorphic-inspired multi-view global-local fusion for IR-UWB radar dynamic gesture recognition,” Frontiers in Neuroscience 2026, DOI [10.3389/fnins.2026.1864448](https://doi.org/10.3389/fnins.2026.1864448). Its full primary article was checked: it uses motion-enhanced TR maps, slow-time MTI, three views and local/global temporal branches on the same corpus. This further preempts presenting generic temporal/frequency processing or view fusion as the contribution.
- Ren et al., “Algorithm for Gesture Recognition Using an IR-UWB Radar Sensor,” 2016, DOI [10.4236/jcc.2016.43015](https://doi.org/10.4236/jcc.2016.43015), already covers hand direction, distance change and hand-surface/energy variation. The repository's archived range-frequency and temporal-descriptor screens are also relevant; this would repeat a nearby descriptor screen under a new name.

## Decision

**Not cleared; no classifier or feature experiment started.** This is a known representation family, and the version computable from the cache is an envelope-modulation spectrum, not a physically measured phase-Doppler observable. Changing FMCW to this UWB corpus or applying another readout would not by itself establish a materially new mechanism. A future UWB-specific method would need a distinct physical hypothesis and primary-source distinction, not just a new spectral transform or model.

The cached files are present in `/workspace/uwb-restored-20260930`; no restore or Drive access was needed. No training process was active in the fresh process check. Current compute is CPU-only, five cores, no CUDA; the root filesystem has about 712 MiB free. No array values, labels or checkpoints were read. The original research contribution remains unvalidated.

## Gate for further work

Continue with the existing cache only if a narrowly stated signal mechanism both (1) has a defensible distinction from the cited and archived work, and (2) is identifiable from real clutter-removed 90×189 maps with a frozen user-disjoint protocol. A claim requiring coherent phase, per-view empty-scene background, calibrated transfer, synchronized inter-radar phase/timing, independent hand-motion truth, or frequencies above the envelope sampling limit requires those measured inputs. The cloud has no acquisition hardware; no further model/descriptor grid is justified until that gate changes.
