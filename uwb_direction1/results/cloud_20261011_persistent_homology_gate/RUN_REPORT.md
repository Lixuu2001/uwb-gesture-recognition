# Bounded persistent-homology prior-art screen

Date: 2026-10-11  
Branch: `codex/uwb-direction1-cloud-20260930`

## Question and scope

Screened whether persistent-homology/topological summaries of the cached range-time gesture representation are a distinct, testable candidate. Search queries covered radar gesture recognition with persistent homology/TDA, IR-UWB phase-space/topological gesture representations, radar micro-Doppler topology, and radar gesture event segmentation. This is a bounded public-source screen, not an exhaustive review.

## Evidence found

- A primary-source radar paper applies persistent homology to 60 GHz OTFS delay-Doppler responses for human-presence detection. It is radar-specific topological signal processing, but not hand-gesture classification or IR-UWB. [Paper](https://www.mdpi.com/1424-8220/23/4/2224)
- A recent SAR study uses topological data analysis to examine simulated/real SAR image embeddings; it is not dynamic hand gestures. [Paper](https://doi.org/10.1080/26941899.2026.2654220)
- A time-series classification paper uses TDA outside radar, establishing that “apply topology to a time series” is generic prior art. [Paper record](https://www.jstage.jst.go.jp/article/tjsai/32/3/32_D-G72/_article/-char/en)
- The exact UWB-Gestures source describes the same IR-UWB gesture corpus and its representations; it does not establish a topological method. The same-corpus 2026 multi-view paper already establishes three-view modeling and fusion, so combining generic topology with those views is not by itself a cleared contribution. [Dataset descriptor](https://www.nature.com/articles/s41597-021-00876-0), [same-corpus multi-view paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full)

The search did not surface an exact IR-UWB hand-gesture persistent-homology classifier in the bounded results. That absence is not proof of novelty. The proposed use would currently be a generic technique transfer, with filtration/point-cloud choices and a signal-specific mechanism still unspecified. **Novelty gate remains closed; no classifier or parameter grid is justified.**

## Runtime and data safety

- Restored data path `/workspace/uwb-restored-20260930` is present; no restore/download was needed.
- Process inspection found no active training process; only three defunct historical Python entries (PIDs 598, 607, 4719). Python 3.12.14 has PyTorch 2.14.0+cpu; CUDA is unavailable.
- Historical `*.lock` / queue lock files remain under prior experiment directories. They were not removed or treated as evidence of a live job.
- No data labels, radar-array values, or checkpoint contents were read. No model was fitted; no training, checkpoint mutation, or source-snapshot change occurred.

## Decision

Do not run a TDA classifier probe in this turn. Generic persistent homology plus radar gestures/multi-view is not yet a defensible original method. Revisit only if a specific radar-observable mechanism and a method-level distinction from the above prior art are identified, with a frozen train-only diagnostic and cost protocol. Otherwise, the current project still needs a genuinely distinct method or new calibrated/annotated data; no valid research contribution has been established.
