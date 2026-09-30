"""P0 diagnostic for method B: may the three radars be treated as one time axis?

Row k of the Left matrix and row k of the Right matrix share a cache index. That
is a storage fact, not a physical one. Before any time-aligned target is built,
this audit asks what the data itself says about a per-frame correspondence, and
reports the zero-energy population that the process loss must skip.
"""
import argparse
import json

import numpy as np

import protocol as P
import motion_targets as MT
from protocol import DIAG, dump

PAIRS = ((0, 1), (0, 2), (1, 2))
MAX_LAG = 6


def normalised_cross_correlation(a, b, max_lag=MAX_LAG):
    """a, b: (N, T) non-negative envelopes. Returns (N, 2*max_lag+1) correlations.

    Each lag is scored on its own overlap window, so a long shift is not
    rewarded merely for comparing fewer frames against a different mean.
    """
    n, t = a.shape
    out = np.zeros((n, 2 * max_lag + 1), 'float64')
    for j, lag in enumerate(range(-max_lag, max_lag + 1)):
        left = a[:, max(0, lag):t + min(0, lag)]
        right = b[:, max(0, -lag):t - max(0, lag)]
        left = left - left.mean(1, keepdims=True)
        right = right - right.mean(1, keepdims=True)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        out[:, j] = (left * right).sum(1) / np.maximum(denominator, 1e-12)
    return out


def peak_statistics(correlation, max_lag=MAX_LAG):
    lags = np.arange(-max_lag, max_lag + 1)
    best = correlation.argmax(1)
    peak = correlation[np.arange(len(correlation)), best]
    masked = correlation.copy()
    for offset in (-1, 0, 1):                       # a neighbouring bin is the same peak
        index = np.clip(best + offset, 0, correlation.shape[1] - 1)
        masked[np.arange(len(correlation)), index] = -np.inf
    second = masked.max(1)
    return lags[best], peak, peak - second


def mismatched_partner(ids, y, subj, seed):
    """Pair each clip with a different clip of the same class and user.

    This is the null the true-pair correlation has to beat. If aligning Left of
    one repetition with Right of another repetition scores the same, then the
    envelopes carry no per-instance temporal correspondence to distil, however
    the lag histogram happens to look.
    """
    rng = np.random.default_rng(seed)
    partner = ids.copy()
    for cls in np.unique(y[ids]):
        for user in np.unique(subj[ids]):
            where = np.where((y[ids] == cls) & (subj[ids] == user))[0]
            if len(where) < 2:
                continue
            perm = rng.permutation(len(where))
            for _ in range(8):
                if not (perm == np.arange(len(where))).any():
                    break
                perm = rng.permutation(len(where))
            partner[where] = ids[where[perm]]
    return partner


def run(folds, energy_kind='phase'):
    y, subj = P.labels()
    energy, concentration = MT.load_energy(energy_kind)
    report = {}
    for fold in folds:
        tr, va, te = P.fold_split(fold, y, subj)
        fold_report = {}
        for a, b in PAIRS:
            correlation = normalised_cross_correlation(energy[tr, a], energy[tr, b])
            lag, peak, margin = peak_statistics(correlation)
            by_class = {int(c): [float(np.mean(lag[y[tr] == c])), float(np.std(lag[y[tr] == c]))]
                        for c in np.unique(y[tr])}
            by_user = {int(u): [float(np.mean(lag[subj[tr] == u])), float(np.std(lag[subj[tr] == u]))]
                       for u in np.unique(subj[tr])}
            fold_report[f'{P.VIEW_NAMES[a]}-{P.VIEW_NAMES[b]}'] = {
                'lag_histogram': {int(k): int(v) for k, v in
                                  zip(*np.unique(lag, return_counts=True))},
                'lag_mean': float(lag.mean()), 'lag_std': float(lag.std()),
                'fraction_zero_lag': float((lag == 0).mean()),
                'fraction_within_one_frame': float((np.abs(lag) <= 1).mean()),
                'peak_correlation_mean': float(peak.mean()),
                'peak_correlation_median': float(np.median(peak)),
                'fraction_peak_below_0.2': float((peak < 0.2).mean()),
                'second_peak_margin_mean': float(margin.mean()),
                'fraction_ambiguous_margin_below_0.05': float((margin < 0.05).mean()),
                'lag_mean_std_by_class': by_class, 'lag_mean_std_by_user': by_user,
                'class_lag_spread': float(np.std([v[0] for v in by_class.values()])),
                'user_lag_spread': float(np.std([v[0] for v in by_user.values()]))}
            partner = mismatched_partner(tr, y, subj, P.AUXILIARY_SEED + fold)
            null = normalised_cross_correlation(energy[tr, a], energy[partner, b])
            null_lag, null_peak, null_margin = peak_statistics(null)
            entry = fold_report[f'{P.VIEW_NAMES[a]}-{P.VIEW_NAMES[b]}']
            entry['null_same_class_other_instance'] = {
                'peak_correlation_mean': float(null_peak.mean()),
                'fraction_zero_lag': float((null_lag == 0).mean()),
                'zero_lag_correlation_mean': float(null[:, MAX_LAG].mean())}
            entry['true_minus_null_zero_lag_correlation'] = float(
                correlation[:, MAX_LAG].mean() - null[:, MAX_LAG].mean())
            entry['true_minus_null_peak_correlation'] = float(peak.mean() - null_peak.mean())
            entry['zero_lag_correlation_mean'] = float(correlation[:, MAX_LAG].mean())
            # Where the population curve peaks answers the offset question; the
            # per-sample argmax only says how noisy one clip pair is.
            curve = correlation.mean(0)
            entry['mean_correlation_curve'] = curve.round(5).tolist()
            entry['population_peak_lag'] = int(np.arange(-MAX_LAG, MAX_LAG + 1)[curve.argmax()])
            entry['null_same_class_other_instance']['mean_correlation_curve'] = null.mean(0).round(5).tolist()
            paired = correlation[:, MAX_LAG] - null[:, MAX_LAG]
            entry['paired_zero_lag_advantage'] = {
                'fraction_true_above_null': float((paired > 0).mean()),
                'by_user': {int(u): float(paired[subj[tr] == u].mean())
                            for u in np.unique(subj[tr])}}
        activity, progress, seg_ok, sample_ok, meta = MT.build_fold_targets(
            fold, tr, mode='multi', device='cpu')
        invalid = ~sample_ok.numpy()
        fold_report['targets'] = {
            'scale_percentile95': meta['scale_percentile95'],
            'invalid_fraction_train': float(invalid[tr].mean()),
            'invalid_fraction_all_quality': float(invalid[np.concatenate([tr, va, te])].mean()),
            'invalid_by_user': {int(u): float(invalid[tr][subj[tr] == u].mean())
                                for u in np.unique(subj[tr])},
            'invalid_by_class': {int(c): float(invalid[tr][y[tr] == c].mean())
                                 for c in np.unique(y[tr])},
            'segments_all_valid': bool(seg_ok.all()),
            'activity_entropy_mean': float((-(activity.clamp_min(1e-12).log()
                                              * activity).sum(-1))[sample_ok].mean()),
            'max_activity_entropy': float(np.log(P.SEGMENTS)),
            'peak_segment_histogram': {int(k): int(v) for k, v in zip(
                *np.unique(activity[sample_ok].argmax(-1).numpy(), return_counts=True))}}
        report[fold] = fold_report
        print('fold', fold, {k: round(v['fraction_zero_lag'], 3)
                             for k, v in fold_report.items() if k != 'targets'}, flush=True)

    pairs = [f'{P.VIEW_NAMES[a]}-{P.VIEW_NAMES[b]}' for a, b in PAIRS]
    across = {pair: {k: float(np.mean([report[f][pair][k] for f in folds]))
                     for k in ('lag_mean', 'lag_std', 'fraction_zero_lag',
                               'fraction_within_one_frame', 'peak_correlation_mean',
                               'fraction_peak_below_0.2', 'second_peak_margin_mean',
                               'fraction_ambiguous_margin_below_0.05',
                               'class_lag_spread', 'user_lag_spread',
                               'zero_lag_correlation_mean',
                               'true_minus_null_zero_lag_correlation',
                               'true_minus_null_peak_correlation')}
              for pair in pairs}
    across['targets'] = {k: float(np.mean([report[f]['targets'][k] for f in folds]))
                         for k in ('invalid_fraction_train', 'invalid_fraction_all_quality',
                                   'activity_entropy_mean')}
    out = {'per_fold': report, 'across_folds': across, 'max_lag': MAX_LAG,
           'energy_kind': energy_kind,
           'acquisition_documentation': 'no trigger, frame-rate or inter-radar timestamp '
                                        'record was found in the local dataset copy; the '
                                        'bundled MATLAB script slices the same 90-row block '
                                        'from all three radars, which fixes the sample unit '
                                        'but does not establish per-frame synchronisation',
           'decision_rule': 'a stable near-zero lag with a clear single peak permits the '
                            'shared time axis the 15-segment targets assume; a broad, '
                            'multi-peak or action-dependent lag does not, and B then falls '
                            'back to whole-clip statistics'}
    dump(DIAG / 'timing_audit.json', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    args = ap.parse_args()
    result = run(args.folds)
    print(json.dumps(result['across_folds'], indent=2))
    print('TIMING_AUDIT_OK')
