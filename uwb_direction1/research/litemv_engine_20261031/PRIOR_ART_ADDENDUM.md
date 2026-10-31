# Primary source behind LITEMV's fixed filters

Ismail-Fawaz, Devanne, Weber and Forestier, *Deep Learning For Time Series
Classification Using New Hand-Crafted Convolution Filters*, IEEE BigData2022.
DOI10.1109/BIGDATA55660.2022.10020496.
[Author-hosted PDF](https://maxime-devanne.com/delegation/publis/ismail-fawaz_BigData2022.pdf),
[author project](https://msd-irimas.github.io/pages/HCCF-4-tsc/).

Read PDF abstract/introduction, sectionIII.A–D andIV.A, plus IV.B opening and
the start of CO-FCN details. The accessible PDF has10 pages; this is not a
full-paper/results-table/figure-visual review. The PDF was read through the web
connector, not locally downloaded or SHA-pinned. The associated CF-4-TSC code
was identified but not fetched or implemented in this continuation.

The paper defines alternating-sign trend filters and a squared-parabolic peak
filter, then combines them with learned FCN/Inception features. Its ensemble
variant uses five hybrid models. These patterns support the provenance of
LITEMV's existing fixed-filter code; they do not establish radar motion,
cross-user robustness, originality here, or a UWB accuracy gain.

This is supporting prior art, not a newly chosen configuration, ablation grid
or replacement experiment. Continue the currently frozen OS-CNN audit/gate,
then the prospective LITEMV accuracy workflow if permitted by that gate.
Published precedent is useful for a strong baseline; it is not a reason to
abandon actual accuracy experiments or to rename fixed filters as an invention.
