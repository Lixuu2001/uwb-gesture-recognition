# IR-UWB event-coded SNN patent prior-art note

Date: 2026-10-11 (UTC)

## New primary-source finding

A patent-family page for CN112130118A/B, *SNN-based ultra-wideband radar signal processing system and processing method*, lists a priority date of 2020-08-19, A-publication date 2020-12-25, and B-publication date 2023-11-17. The disclosed pipeline receives IR-UWB reflected pulses, applies time-event coding, feeds the coded sequence to an SNN, and describes gesture recognition as an example application. The document also describes supervised and unsupervised/WTA variants. [Patent-family record and text](https://patents.google.com/patent/CN112130118A/en)

This is a prior-art indicator for the broad combination “IR-UWB event coding + SNN for gestures.” It reinforces the 2022 IEEE paper's abstract-level description of time-event/level-crossing encoding and SNN classification. The IEEE record reports results for five/six gestures, but its full paper and exact relationship to the UWB-Gestures release remain unverified. [IEEE paper record](https://ieeexplore.ieee.org/document/9870013/)

## Scope and limits

This note is not a patentability, infringement, or freedom-to-operate opinion. It does not establish that the patent or the 2022 paper used the same public UWB-Gestures samples, three-view setup, or eight-held-user protocol. It does establish that “use event-coded SNN on IR-UWB gestures” is not a sufficiently distinct proposal by itself.

The current project cache contains processed clutter-removed range-time maps, not the original reflected pulse stream needed to validate sensor-native event capture. Turning these maps into threshold events would be a different, software-derived representation and would still face the cited gesture-specific event/SNN prior art. No such conversion was run.

## Decision and execution

Do not start an event-coded/SNN grid or claim novelty for that generic combination. Any future variant must specify a mechanism that differs concretely from the disclosed time-event pulse encoding/SNN pipeline and the 2022 gesture paper, then pass a protocol-compliant feasibility gate. No data, labels, arrays, or checkpoints were read; no archive was fetched and no training or fitting occurred. A fresh process check at 2026-10-11 20:11 UTC found no active training job.

The latest cloud executor remains CPU-only/restricted with limited workspace space, and this dataset lacks raw complex/pulse-level captures and calibrated motion truth. Details of the prior-art review are bounded to the patent text and publisher record; no legal conclusion is made.
