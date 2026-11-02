Fixed published TS-TCC multivariate adaptation, one seed20261029, full HAR
default internal architecture and batch128/40epochs per phase. Four phases:
SSL40, frozen-parameter linear40, full fine-tune40, same-architecture scratch40.
Both readouts transfer the same final SSL nonlogits state; fresh identical head
and fresh Adam. All four1440update stages total5760updates/737280appearances.
Paper standard linear evaluation and source-domain transfer fine-tuning motivate
both readouts before any real fitting. No post-diagnostic choice added later.

Keep full189rangebins x3views x90frames, record/view RMS, explicit567channels
and90slowtime axis, source users3–6=4700train/2=1000validation/1=1200diagnostic.
Existing quality exclusions/6900IDs/maps hashes fixed; no user7–8 labels.
Batch36x128=4608records each epoch,92dropped by fixed seeded shuffle, same
order each phase; all actual IDs logged. All eight users historically developed,
so this is exploratory, not untouched test validation or an eightfold result.

Original core models/loss/optimizers/step body unchanged; transparent NumPy2
segment-list fix and allchannel time-permutation fix, original Gaussian gains/
jitter/cuts/emptysegments/code maxseg8 retained. Fixed augmentation once for
training records only. Streamed float32 final views and final NumPy RNG exactly
match full corrected-reference augmentation; allchannels checked. No radar
physical-label invariance established and no new architecture/contribution.

Original code-vs-paper differences and CPU/dedicated sampler are explicit in
PAPER_AND_ADAPTATION. Linear keeps original train-mode dropout and BN running
statistics; only encoder parameters freeze. Classification scheduler is default
ReduceLROnPlateau using user2 CE; validation retains all1000instead of dropped
records. Select exactly last40epoch, no validation checkpoint/grid choice. SSL
does not use label or validation model selection. Seed/configuration fixed.

Freeze all3classifiers' validation+diagnostic IDs/logits/predictions before any
held1 labels are indexed. Report present-class macro. Any prospectively named
arm must exceed frozen strongest same-held1 .9866666666666667 by at least.005
(1e-12tolerance) to license separately frozen source confirmation. Otherwise
close this exact config with no augmentation/epoch/seed/width rescue search.
Report all arms and fine-tune/linear-vs-scratch differences, with no originality
or mechanism conclusion from architecture combination or accuracy alone.

All original author single-step and checkpoint-restored next2synthetic steps
passed bitwise for both models/optimizers/scheduler/allRNG in all4phases.
Four pre-data guard tests pass. Synthetic32fullbatch update cost estimate
2785.263668seconds passes43200CPU/wall. Conservative measured first-update max
included; cost/test were concurrent independent one-thread processes, which may
inflate measured latency. Actual child bounded43000/43100CPU and43100wall by
independent supervisor; no automatic retry/resume. No real TS-TCC values or fit
before remote preregistration exact readback. Original789snapshot unchanged.

Full model/temporal/optimizers/scheduler/Torch+NumPy+PythonRNG/identity/epochcursor
initial+first+rollinglast checkpoints and separate10/20/30/40snapshots eachphase;
keep fixed augmentation recipe/hash, all ID logs, validation histories and final
unscored predictions. A tested restore helper exists; no automatic job-resume.
Completed-run audit independently reconstructs all5760updates, all737280ID
appearances, all four initial/first/10/20/30/40/last states, full6900RMS inputs,
fixed4700augmentation views,120validation epochs and6600final logit rows. Audit
7200CPU/wall budget, no gate accepted before successful audit/private backup
with authenticatedraw readback and all archive/member checks. Raw/model/ID
values stay out of Git; private recovery indices and metadata allowed.
