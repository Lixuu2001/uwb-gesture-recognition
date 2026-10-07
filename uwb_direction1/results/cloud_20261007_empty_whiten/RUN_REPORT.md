# Training-empty spectral prewhitening screen failed

Preregistration pushed before fitting: `384f3d2380955dc5d2e6d8903e0c53ecbee9d2a4`. Protocol SHA256 `64616dd50b95506892a7a83c28631c9797ebad9ddd57f9b0c776a536c2967a27`.

| Arm | Mean source-user macro accuracy |
|---|---:|
|ridge|93.063131%|
|raw_blocks|93.092172%|
|empty_whiten|92.940657%|
|shuffled|92.891414%|

- Primary vs ridge: -0.122475pp, 2/6 strict wins, worst -0.818182pp; passed=False.
- Primary vs raw_blocks: -0.151515pp, 2/6 strict wins, worst -0.666667pp; passed=False.
- Primary vs shuffled: +0.049242pp, 2/6 strict wins, worst -0.250000pp; passed=False.

Joint effect gate fails. Close the fixed PSD/shrinkage/FFT/block/seed configuration; no eightfold extension, parameter grid or secondary-control promotion. This does not rule out all clutter suppression or source-background modeling. Existing Huber and user-deletion failures were not rerun.

Six reused development users3–8, four training users/next validation/held diagnostic, identical quality exclusions and frozen128dim epoch24 source CNN features. Each fold estimates a per-radar fast-time spectral template from400 training-only empty-class11 clips, equally weighting its four source users. Each clip/view is divided by its own RF RMS. A boxcar512-point zero-padded FFT computes undoubled positive-frequency periodogram ordinates; average over90frames, clips, then users. Gain is inverse square root of PSD plus0.1 times full-spectrum mean PSD, normalized to unit mean-square gain over the full mirrored spectrum. This is an established regularized prewhitener under a circular/stationary approximation, not a novel filter or a proven noise model.

Every query uses the training template without class/user adaptation. The filtered512-point inverse FFT is cropped to its first189 range samples. Log1p RMS in9slow-time×21range blocks per radar gives567 features, concatenated with128 frozen CNN features. Raw-block control uses exactly the same descriptor dimensions with no filter. Shuffled control retains DC/Nyquist and permutes interior positive-frequency gains independently by radar with fixed seed[20261007,held,view], preserving full gain distribution and white-noise gain but breaking frequency alignment. Original ridge is unchanged. Fast-time FFT frequency is reciprocal range-bin index, not slow-time Doppler.

Each descriptor arm fits training-only equal-user/equal-present-class weighted ridge with its own training mean/scale/intercept. Select among lambda.001/.01/.1 using only next validation user's macro accuracy, then onehot MSE, then larger lambda. Save all candidate validation scores. Held labels enter only evaluation after fitted parameters and predictions are persisted. Global label arrays are loaded for source/validation indexing and are not technically sealed; no held-label fitting/selection is performed. Descriptor arrays are deterministically regenerated from raw SHA-pinned data/templates/code, with full array SHA saved instead of duplicating large arrays. Existing baseline has a common−.5 cached score offset, checked with unchanged argmax.

Source empty gestures can contain RF system response, body returns and informative pulse structure, so they cannot be equated to pure nuisance noise. Filtering may discard useful spectrum, amplify weak bands, distort finite-window boundaries or alter the pulse envelope. No SNR optimality, preserved range resolution or target-domain covariance guarantee is claimed. The frozen CNN is unchanged; only the appended descriptors see the filtered input. Same-size raw features and permuted gains prevent attributing generic feature enlargement to learned noise structure.

A post-hoc training-template diagnostic (TEMPLATE_DIAGNOSTIC.json) gives mean spectral flatness0.014350 before filtering,0.154946 with aligned gains and0.007450 with shuffled gains. These are the same fitted source empty periodograms, so the change verifies an in-sample spectral intervention, not noise ground truth or out-of-domain suppression. It changes no gate or model selection.

Independent audit recomputed periodograms with SciPy's full two-sided API and filtered signals using full complex FFT/IFFT plus explicit block slicing. It checked 127800 clip/arm/fold appearances and every saved descriptor array SHA; maximum descriptor difference 6.6613381e-16. It independently rebuilt 54 regression fits and validation selections; maximum score difference 6.1173289e-13, all argmax identical. Exact source/empty IDs, gain permutations, truth alignment, 63 artifact SHA and gate arithmetic passed. Synthetic identity-filter and Parseval checks passed. Actual run 185.015668s versus synthetic estimate434.616655s under900s budget. No new CNN training or background queue.

Literature: [Luo etal2015, optimal receive filter design in coloured noise](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2014.0421) provides motivation and cautions about envelope distortion; abstract/introduction/section2 prose read, image equations and later sections not fully reviewed. Our simpler spectral filter is not a replication. [SciPy periodogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html) supports scaling verification. Review details and limitations are in research/EMPTY_WHITEN_REVIEW.md. Established filtering plus block features is not an original contribution.

Reproduce after restoring the private prior artifacts, using fresh output directories:

```sh
python uwb_direction1/tools/test_empty_whiten.py
python uwb_direction1/tools/run_empty_whiten.py --protocol uwb_direction1/research/EMPTY_WHITEN_PROTOCOL.json --out /workspace/empty-whiten-replay-cost --cost-only
python uwb_direction1/tools/run_empty_whiten.py --protocol uwb_direction1/research/EMPTY_WHITEN_PROTOCOL.json --out /workspace/empty-whiten-replay --cost /workspace/empty-whiten-replay-cost/COST.json
python uwb_direction1/tools/audit_empty_whiten.py /workspace/empty-whiten-replay
```

Private artifacts include source template PSD/gains/IDs, models, all candidate validation predictions, final query scores, numerical checks, configs and logs. Prior energy-area/semantic-code Drive receipts preserve encoder/feature/baseline recovery chains; migration preserves raw arrays. No large arrays in Git, original789source snapshot unchanged. Final Drive receipt is authoritative; archived report may precede receipt. Eightfold94.897727% ridge is a different protocol and is not directly compared to these means. Goal remains unmet; automatic continuation remains enabled.

私有备份已逐份下载复验：7块、12916333字节、95成员SHA全部一致，元数据shared=false且仅owner；未更改共享权限。
