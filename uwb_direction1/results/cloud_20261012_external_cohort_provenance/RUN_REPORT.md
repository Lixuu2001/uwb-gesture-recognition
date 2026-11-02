# Independent IR-UWB gesture-cohort provenance check

Date: 2026-10-12 18:03 UTC  
Branch head at review: `1763721af10f48163d282b93a8ed9fd1792f9347`  
Review type: bounded source search and primary dataset-metadata inspection. No dataset files were downloaded and no UWB array, label, checkpoint, fit or training was accessed.

## Finding

The apparent 2024 Figshare candidate `UWB_HGR_Dataset` (record 27915876) is explicitly described by its own landing-page metadata as a **preprocessed form of the original UWB-Gestures dataset**. Its card lists 12 gesture categories, eight volunteers, and each sample as 90×183×3. The original 2021 descriptor identifies 9,600 samples from eight volunteers and three independently stored X4 radar views. This is a repack/processed release of the same cohort, not independent external-user validation.

The bounded search for public dynamic hand-gesture datasets specifically captured with impulse-radio UWB found the original UWB-Gestures release and this derivative, plus other works evaluating on the same cohort. It did not identify a verified independent cohort in this search. This is not an exhaustive claim about every data repository. FMCW/mmWave or other IR-UWB human-activity sets are not assumed to provide label-, sensor-, or protocol-equivalent external validation.

## Primary sources

- The Figshare landing page for [UWB_HGR_Dataset, record 27915876](https://figshare.com/articles/dataset/UWB_HGR_Dataset/27915876) says it is a preprocessed form of the original corpus and gives its class, participant, and sample-shape metadata.
- The original [UWB-Gestures data descriptor](https://doi.org/10.1038/s41597-021-00876-0) reports 12 dynamic gestures, 9,600 samples, eight volunteers, and three independent monostatic X4 views.
- The public Figshare record [12652592](https://doi.org/10.6084/m9.figshare.12652592) is the original UWB-Gestures release, not a second cohort.

## Decision and execution

The derivative does not open an external-validation route or add an eligible signal channel. The existing Direction 1 objective remains unvalidated. No network download, local restore, data fit, training, or model search was performed. The existing migration and private Drive folder were left untouched.

The next useful evidence change would be an independently recruited cohort or a measured condition absent from the current cache, with stable sample/user mapping and the exact sensor channels required by a separately literature-cleared mechanism. If only the existing cohort and processed real-valued TR maps remain available, no new training claim follows from this provenance check.
