# Full FCN2000/B16 synthetic CPU policy

Pinned hfawaz review FCN Conv128/256/128 kernels8/5/3, BN/ReLU/GAP/12softmax, native Keras3.11.3 Torch backend. All846860/845836 parameters and input90x567 retained. Original paper author Conv2D/univariate preprocessing and Adam epsilon differences are disclosed in source/PAPER_REVIEW.json; this is a transfer baseline, not legacy TF parity or novelty.

One fixed2000epochs/B16/seed20261029/oneCPU policy, unweighted CE/Adam, source ReduceLR before minimum-trainingloss checkpoint. No parameter/seed/epoch/batch sweep or smaller fallback. Cost fixture synthetic only,8train/4eval/3native saves/3fullstate saves. Full scenario4700train/1000validation/1200diagnostic uses294/63/75 batches,588000updates and9400000appearances. No UWB values/labels or new real training authorized by this registration. Prior MLSTM exact250/B128 configuration failed its bounded gate and is closed.

Conservative formula2*(2000*294*maxtrain+126138*maxeval+2001*maxnative+4004*maxfullstate)+120 includes everyepoch validation, final dual inference and worst-case best+last complete checkpoints. Fixed43200s/14GiB gate; resource failure is not an accuracy negative. Independent profile CPU590soft/600hard, wall620; no retry.

Before any timing: latest-parent force=false Git publication, exact all-file readback, matching COST_HASHES marker; all source, runtime1302 files and private/audited predecessor closure bytes verified before candidate import. Pass only authorizes separately registered actual UWB runner/full auditor/accuracy protocol; fail closes this exact resource policy. Source evidence remains exploratory and7-8 labels excluded.
