# Public IR-UWB hand-gesture dataset check — 2026-10-09

## Scope and decision

This is a bounded search for an independently collected public impulse-radio UWB hand-gesture cohort that could provide external validation. It is not an exhaustive literature survey and not an experiment. The targeted searches used the phrases “public IR-UWB radar hand gesture recognition dataset”, “UWB radar gesture dataset hand gestures public raw data”, and “public impulse radio UWB radar gesture recognition dataset participants”.

**Decision:** no independent, publicly available dynamic IR-UWB hand-gesture cohort was identified in this search. The primary public dataset found is UWB-Gestures itself. A later Figshare record named `UWB_HGR_Dataset` explicitly points to the UWB-Gestures paper and describes 12 gestures, 800 samples per gesture, eight volunteers, and 90×183×3-shaped data; its metadata does not establish a new collection or an independent cohort. Treat it as a derivative/repackaging candidate, not external validation. This is a bounded-search result, not proof that no such dataset exists.

OPERAnet is a separate, synchronized multimodal dataset with UWB, Wi-Fi, passive radar and Kinect, but its task is daily activity recognition/localization rather than the same dynamic hand-gesture vocabulary. It is not a direct external test cohort for the current UWB-Gestures classifier without changing the endpoint and obtaining compatible labels.

## Primary sources

1. Ahmed et al., “UWB-gestures, a public dataset of dynamic hand gestures acquired using impulse radar sensors,” *Scientific Data* 8, 102 (2021), [DOI 10.1038/s41597-021-00876-0](https://doi.org/10.1038/s41597-021-00876-0). The publisher describes 9,600 samples, eight volunteers, twelve dynamic gestures, and three independently stored radar views. Its data-record section identifies the Figshare dataset record below.
2. Ahmed & Park, [“A Public Dataset of Dynamic Hand-gestures Acquired using Impulse-radar sensors”](https://doi.org/10.6084/m9.figshare.12652592), Figshare record 12652592, version 3 (2022-04-20). The record is the dataset cited by the 2021 descriptor, not evidence of a new cohort.
3. `UWB_HGR_Dataset`, [Figshare record 27915876](https://figshare.com/articles/dataset/UWB_HGR_Dataset/27915876) (search-result metadata; full record fetch returned 403 in this review). Its listing cites the original UWB-Gestures paper and reports matching participant/class/window counts. The accessible metadata was insufficient to verify file-level provenance, so the derivative classification remains qualified.
4. Bocus et al., “OPERAnet, a multimodal activity recognition dataset acquired from radio frequency and vision-based sensors,” *Scientific Data* 9 (2022), [DOI 10.1038/s41597-022-01573-2](https://doi.org/10.1038/s41597-022-01573-2). The publisher describes UWB alongside Wi-Fi, passive Wi-Fi radar and Kinect across six participants performing daily activities; this is related UWB sensing data, but not a hand-gesture cohort.

## Consequences

- The current eight-person UWB-Gestures results do not gain external-cohort validation from repackaged copies of the same collection.
- OPERAnet may support a different activity-sensing question, but it cannot validate the present gesture-classification endpoint as-is.
- Do not download either large archive: the restored original package is already present, and no independent gesture cohort was established.
- No labels, radar arrays, sample-level scores, or private backups were read. No code/model fitting or training was performed. The method-contribution goal remains unvalidated.

