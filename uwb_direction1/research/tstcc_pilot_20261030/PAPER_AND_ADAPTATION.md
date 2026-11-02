Published candidate: Eldele et al., IJCAI2021:2352–2359,
https://doi.org/10.24963/ijcai.2021/324 . Publisher HTML extraction at
https://ijcai.org/proceedings/2021/324 was read for Methods3.1–3.3,
Implementation4.2 and evaluation5.1–5.3/Table1–3. PDF redirect failed;
equations were not decoded, figures not inspected. No full-paper review claim.

Paper uses weak/strong augmented contexts for cross-view future prediction,
plus contextual contrast. Forty pretraining and downstream epochs, batch128,
Adam.0003/betas.9/.99/decay.0003 match code. HAR code uses max_seg8 while
paper states10; code weak view multiplies Gaussian mean2/sigma1.1 gains per
timestamp without separate jitter, differing from prose. We retain code rules.

Adaptation preserves all567variables/90frames via explicit channels-first
axes. Original source channel0 broadcast is repaired to a shared temporal
permutation of every channel; NumPy2 ragged shuffling uses segment indices.
Gaussian distributions/segment cuts, exclusive upper bound and empty segments
remain. Fixed training views are generated once with original RNG order then
float32 conversion; streaming reduces temporary memory. This is documented
multivariate adaptation, not byte-identical author augmentation or a novel
physical transformation. No correctness/label-invariance claim for radar.

Full HAR internal dimensions and40epoch stages retained. Shape-only changes
567input/12classes/13latent steps. Linear evaluation freezes encoder parameters
but retains original train-mode dropout and BatchNorm running-stat updates.
Last-epoch selection and validation-loss ReduceLROnPlateau match author driver;
full1000validation records are retained instead of validation drop_last=True.
One fixed seed/user-disjoint pilot differs from original five-seed dataset
study. Full method transfer may fail; all source scores remain exploratory.
