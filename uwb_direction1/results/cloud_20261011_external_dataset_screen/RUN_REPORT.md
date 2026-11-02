# External radar gesture dataset screen

**Date:** 2026-10-11  
**Scope:** bounded public-source search for an independent cohort that could validate X4 IR-UWB gesture findings. This is not an exhaustive dataset census.

## Findings

1. The primary UWB-Gestures data descriptor presents the 12-class, eight-volunteer, three-X4-view collection as the public IR-UWB dynamic hand-gesture dataset. The restored cache in this project is derived from that same cohort. A separately hosted Figshare item, `UWB_HGR_Dataset`, describes itself as a preprocessed form of the original and repeats the same 12 classes, eight volunteers, 800 samples per class, and 90×183×3 shape. It is therefore not an independent test cohort. This is based on dataset descriptions, not a file-level hash audit.  
   Sources: [Scientific Data descriptor](https://www.nature.com/articles/s41597-021-00876-0), [Figshare item 27915876](https://figshare.com/articles/dataset/UWB_HGR_Dataset/27915876).

2. The KIT/RADAR4KIT collection is a genuinely separate gesture cohort: its institutional record documents 2,479 FC-FMCW recordings from 25 people, seven gestures and zero-class data, captured with an Infineon BGT60TR13C; it is released under CC BY 4.0. The record establishes a different sensor family and cohort, but the public metadata reviewed here does not establish a label-by-label mapping to the X4 classes or a signal representation directly compatible with the project's clutter-removed X4 range-time maps. It could support a separately scoped cross-sensor transfer study after a mapping and preprocessing review; it is not protocol-matched external validation of the current X4 result.  
   Source: [KIT institutional dataset record](https://publikationen.bibliothek.kit.edu/1000172790).

3. The public Multi-Radar-Dataset repository documents two IR-UWB streams plus one FMCW stream, four volunteers, and 192 minutes in two indoor scenarios. Its described labels are posture/body-motion and multi-person conditions (stillness, random limb movement, upper-body rocking, turning), not the 12 hand gestures. It is not an external hand-gesture test set.  
   Source: [author repository and dataset description](https://github.com/yangxiuzhu777/Multi-Radar-Dataset).

4. MIRAGE is a separate mmWave point-cloud gesture/interference benchmark using a TI IWR1843 FMCW radar. Its official repository describes point-cloud features and multiperson interference experiments; the signal type and task distribution differ from the three-view X4 impulse-radar corpus. It may support a different cross-task robustness question, but cannot be treated as matched IR-UWB gesture validation.  
   Source: [MIRAGE authors' repository](https://github.com/CEA-LIST/MIRAGE).

## Research decision

This bounded search did not identify an independent, protocol-compatible IR-UWB hand-gesture cohort. That is a scoped search result, not proof that none exists. Do not download or relabel the cited datasets merely to claim external validation. The KIT set is a possible future cross-sensor transfer study only if a defensible shared-label mapping, input conversion, and frozen train/test protocol are established first.

No UWB project labels, radar-array values, or checkpoint contents were accessed. No fitting or training ran. The restored data at `/workspace/uwb-restored-20260930` was already present and was not re-downloaded. Process inspection found no active UWB/training process; the three Python entries were defunct historical processes. The environment is CPU-only and has no acquisition device. The research contribution gate remains unresolved; this dataset screen does not clear or validate a method.

## Reproduction of the screen

Search was limited to public primary dataset records and author repositories for IR-UWB gesture datasets, multi-radar UWB datasets, and separate public hand-gesture cohorts. The above inclusion/exclusion decisions use the dataset descriptions linked beside each finding. No archive was downloaded. Revisit only if a new candidate with matching hand-gesture labels and traceable independent participants is found, or if the research scope is explicitly changed to cross-sensor transfer.
