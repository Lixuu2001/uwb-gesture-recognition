Next published-method candidate: Eldele et al., Time-Series Representation
Learning via Temporal and Contextual Contrasting, IJCAI2021:2352–2359,
DOI10.24963/ijcai.2021/324. Paper link from author README:
https://www.ijcai.org/proceedings/2021/0324.pdf .

Read scope: complete author README, model, temporal module, attention, loss,
trainer, loader, augmentation, HAR config and main at pinned author commit
15bdfd6eced6aeb6c638839828effe1b4b2ea391. Paper full text/figures have not
been read. This is a runnable-source preparation, not a full paper replication
or UWB accuracy result. Bounded latest memory/STATUS search has no exact TS-TCC
method entry; TSLANet references the same author name and remains a distinct
failed method. This bounded search is not an exhaustive novelty proof.

Unmodified core imports and actual author model_train successfully execute one
full HAR-default synthetic batch128 SSL update on CPU: encoder247500 parameters,
temporal313096, latent128x128x13, finite loss/weights and9encoder+65temporal Adam
states. Keep internal32/64/128 convolution widths, transformer hidden100/depth4/
heads4, six prediction steps,40epochs,batch128, Adam.0003/betas.9/.99/decay.0003.
Explicit shape-only adapter567inputchannels/12classes/features_len13 supports
90frames; original HAR has9/6/18. Source preprocessing/training not invoked.
Two arbitrary random tensors supplied the two views for this smoke only; it
does not test author augmentation, full trajectory/loss parity, resume or cost.
The deprecated implicit LogSoftmax dimension emitted12 warnings and remained
unmodified. einops0.8.1 is reused from the existing pinned private workspace;
its file hashes are indexed. No new package installation or real arrays.

Concrete pre-registration issue to resolve, not an accuracy rejection: original
augmentation NumPy ragged-list permutation fails under current NumPy2. After
analysis-only index-list compatibility, the unchanged ret[i]=pat[0,warp] line
broadcasts channel0 across all channels for4/4 synthetic permuted records.
The loader's minimum-dimension heuristic would also swap90time/567channels;
real adapter must preserve explicit axis order. Paper-vs-code multivariate
augmentation semantics need primary-paper verification before choosing the
fixed real-data implementation. Do not silently fix this or call either
variant physically validated. A documented compatibility/multichannel adapter,
original-step comparison, full checkpoint/RNG resume test and bounded full
40epoch pipeline cost are next concrete work; then remotely freeze exactly one
source-user protocol and run it after previous TS2Vec audited/preserved closure.
Prior-art status alone is not a reason to stop the accuracy experiment loop.

No TS-TCC real-data fitting, source metrics, C/seed search, prediction or label
access occurred. No training launcher or real protocol exists yet. No original
contribution claimed. Root LICENSE fetch returned404; license not verified,
so this preparation publishes our adapter/checks and source hashes/references,
not a redistributed author source snapshot. The local10blobs were restored to
byte-exact UTF8 and Git blobSHA1 after apply_patch added trailing newlines before
any import; trace in SOURCE_INDEX. Requests-package absence and blocked shell
api.github.com CONNECT403 were handled with stdlib/connected GitHub tools;
configured network policy was not bypassed.
