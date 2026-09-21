#!/usr/bin/env python
"""Analyse the per-sample CMA-ES sweep log written by adaptation/tf_mpol.py.

Each row is one test sample's optimum: the fitted quantile-function parameters
(a, s, d) for f(q) = (1-d)*sigmoid(a*(q - s)) + d -- s is the transition
quantile directly -- the achieved metric (best_<metric>), the model's baseline
(pred_<metric>) and the gain (delta_<metric> = best - pred). The optimised
metric (pesq, ssnr, ...) is auto-detected from the column names.

It answers one question: do the optimal fit parameters correlate with the
sample's difficulty (baseline metric)?  e.g. do low-quality samples prefer a
different warp shape?

Usage:
    python analyze_sweep.py [sweep_params.csv] [-o sweep_stats.png]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PARAMS = ['a', 's', 'd']

# Warp-domain clamps for the deployed map; must match cfg.regress {min, max}.
# d's min is negative: a clamped d<0 hard-gates the noisiest bins to 0.
CLAMP = {'a': (0.1, 30.0), 'd': (-0.1, 0.5), 's': (0.0, 1.0)}


def detect_metric(df):
    """Find the optimised metric from a 'best_<metric>' column."""
    for col in df.columns:
        if col.startswith('best_'):
            return col[len('best_'):]
    raise ValueError("no 'best_<metric>' column found in CSV")


def correlate(x, y):
    """Pearson and Spearman with p-values, NaN-safe."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), len(x)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp, len(x)


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''


def ols_fit(x, y):
    """Ordinary least-squares line fit + R^2, NaN-safe.

    Returns (slope, icpt, r2, n) or (nan, nan, nan, n). Plain least squares: the
    b/a leverage outliers that once motivated a robust (Theil-Sen) fit are gone
    (dropping m + the fixed-a sweep), so OLS reproduces the same line while being
    the standard, no-explanation-needed choice. It is also the right *predictor*
    here: the same noisy blind feature is used at fit and inference, so OLS's
    attenuated slope is wanted -- no errors-in-variables (TLS) correction.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan'), float('nan'), len(x)
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return slope, icpt, r2, len(x)


def poly_fit(x, y, deg):
    """Polynomial least-squares fit + R^2, NaN-safe. deg=1 reduces to ols_fit.

    For the transition quantile s the blind-feature map is monotone but curved
    (spearman > pearson; s ramps up steeply at low act_frac), so a degree>1 fit
    can cut the low-feature bias -- exactly the high-s, high-leverage regime that
    also gates d. Returns np.polyfit coeffs (highest power first) so the curve is
    np.polyval(coef, x). Keep deg low: the curvature sits where the blind feature
    is noisiest, so a flexible fit trades bias for variance there.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < deg + 2 or x.std() == 0 or y.std() == 0:
        return np.full(deg + 1, np.nan), float('nan'), len(x)
    coef = np.polyfit(x, y, deg)
    pred = np.polyval(coef, x)
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2, len(x)


def ols_fit_w(x, y, w):
    """Weighted least-squares line fit + weighted R^2, NaN-safe.

    Down-weights samples where the parameter has no leverage. For the floor d the
    weight is the suppressed mass (~ s, the fraction of the normalized rank below
    the transition): where few bins reach the floor PESQ is flat in d, so the
    oracle d is noise there -- and mispredicting it is harmless. Weighting by
    leverage is thus both the right estimator (ignore the noise) and the right
    decision objective (the PESQ cost of a d-error is itself ~ leverage).
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan'), float('nan'), len(x)
    W = w / w.sum()
    xm, ym = (W * x).sum(), (W * y).sum()
    var = (W * (x - xm) ** 2).sum()
    if var == 0:
        return float('nan'), float('nan'), float('nan'), len(x)
    slope = (W * (x - xm) * (y - ym)).sum() / var
    icpt = ym - slope * xm
    pred = slope * x + icpt
    r2 = 1.0 - (w * (y - pred) ** 2).sum() / (w * (y - ym) ** 2).sum()
    return slope, icpt, r2, len(x)


def lad_fit_w(x, y, w, iters=60, eps=1e-6):
    """Weighted least-absolute-deviations (median) line fit + weighted R^2.

    Robust hedge for the floor d against the one-sided pile-up of censored d=0
    samples: OLS fits the (mass-weighted) conditional *mean*, which a boundary
    pile-up drags; LAD fits the conditional *median*, which shrugs it off, with no
    symmetric-Gaussian assumption. Solved by IRLS (reweight each point by
    w/|resid|), seeded from OLS. Returns (slope, icpt, r2_l2, n) where r2_l2 is the
    *L2* weighted R^2 of the LAD line -- same yardstick as ols_fit_w, so the gap to
    the OLS R^2 is exactly the variance cost of going robust. Close coefficients +
    R^2 => OLS leaves nothing on the table.
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan'), float('nan'), len(x)
    slope, icpt = np.polyfit(x, y, 1)  # OLS seed
    for _ in range(iters):
        rw = w / np.maximum(np.abs(y - (slope * x + icpt)), eps)
        W = rw / rw.sum()
        xm, ym = (W * x).sum(), (W * y).sum()
        var = (W * (x - xm) ** 2).sum()
        if var == 0:
            break
        ns = (W * (x - xm) * (y - ym)).sum() / var
        ni = ym - ns * xm
        if abs(ns - slope) < 1e-10 and abs(ni - icpt) < 1e-10:
            slope, icpt = ns, ni
            break
        slope, icpt = ns, ni
    pred = slope * x + icpt
    ym_w = (w / w.sum() * y).sum()
    r2 = 1.0 - (w * (y - pred) ** 2).sum() / (w * (y - ym_w) ** 2).sum()
    return slope, icpt, r2, len(x)


def leverage_split(df, feature, param='d', by='s'):
    """param<-feature correlation split by leverage (suppressed mass ~ s).

    The floor d only shapes the output where mass sits below the transition, so
    its oracle value -- and its feature-predictability -- is heteroscedastic:
    strong where `by` is high, pure noise where `by` is low. A single pooled R^2
    averages the two and understates the recoverable structure (the low-`by` half
    is both unpredictable AND irrelevant).
    """
    s = df[by].to_numpy()
    med = float(np.median(s))
    x, y = df[feature].to_numpy(), df[param].to_numpy()
    print(f'  leverage split of {param}<-{feature}  ({by} median={med:.2f}):')
    for name, m in [(f'low {by}  (little mass at floor)', s < med),
                    (f'high {by} (much mass at floor)', s >= med)]:
        pr, pp, sr, sp, nn = correlate(x[m], y[m])
        print(f'    {name:<30} pearson={pr:>+7.3f}{stars(pp):<3} '
              f'std({param})={y[m].std():.3f}  n={nn}')


def multi_ols(X, y):
    """OLS fit y ~ X (one or more feature columns) + intercept, NaN-safe.

    Returns (coefs, icpt, r2, n); coefs aligns with X's columns. Multivariate
    counterpart of ols_fit for the joint snr_db+act_frac regression.
    """
    X = np.atleast_2d(X)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) < X.shape[1] + 2 or y.std() == 0:
        return np.full(X.shape[1], np.nan), float('nan'), float('nan'), len(y)
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef[:-1], coef[-1], r2, len(y)


def detect_ablation_metric(df):
    """Return the metric of a param_ablation log (has a 'gap_<metric>' col), else None."""
    for col in df.columns:
        if col.startswith('gap_'):
            return col[len('gap_'):]
    return None


def detect_ablation_param(df):
    """Which warp param the 1-D oracle searched, read off the '<p>_oracle' column."""
    for p in ('a', 's', 'd'):
        if f'{p}_oracle' in df.columns:
            return p
    raise KeyError('no <a|s|d>_oracle column: not a param_ablation log')


def analyze_ablation(df, out):
    """Summarise the regressed-vs-1-D-oracle ablation of one warp param.

    The per-sample gap = oracle_p - fixed_p upper-bounds what *any* blind
    predictor of p could add on top of the other two params' deployed regression.
    A small gap => p is effectively universal (fix it). A large gap => look for a
    p-feature; the gap-vs-feature correlations say which feature that might be.
    """
    metric = detect_ablation_metric(df)
    p = detect_ablation_param(df)
    FIX, ORA, GAP = f'fixed_{metric}', f'oracle_{metric}', f'gap_{metric}'
    P_ORA, P_FIX = f'{p}_oracle', f'{p}_fixed'
    df = df[np.isfinite(df[FIX]) & np.isfinite(df[ORA])].copy()
    # a is a multiplicative steepness clamped to [0.1, 30], so log(a) is its
    # natural regression scale: a blind a-predictor deploys as
    # a = exp(slope*feature + icpt), keeping a > 0 without a clamp and making the
    # slope a per-unit *ratio* on steepness. mean log(a) => geometric-mean a.
    # s and d are additive (and d may be <= 0), so they stay on the linear scale.
    log_col = f'log_{P_ORA}' if p == 'a' else ''
    if log_col:
        df[log_col] = np.log(df[P_ORA].where(df[P_ORA] > 0))
    n = len(df)
    print(f'Loaded {n} ablation samples from gap_{metric}  '
          f'(metric: {metric}, searched param: {p})\n')
    print(f'  mean regressed-{p} {metric}:  {df[FIX].mean():+.4f}')
    print(f'  mean oracle-{p} {metric}:     {df[ORA].mean():+.4f}')
    print(f'  mean gap (oracle-fixed):  {df[GAP].mean():+.4f}'
          f'   (median {df[GAP].median():+.4f}, '
          f'95th pct {df[GAP].quantile(0.95):+.4f})')
    print(f'  samples with gap > 0.01:  {(df[GAP] > 0.01).mean():.1%}')
    print(f'  oracle {p}: mean {df[P_ORA].mean():+.3f}  '
          f'std {df[P_ORA].std():.3f}  '
          f'(regressed {p}: mean {df[P_FIX].mean():+.3f}, '
          f'std {df[P_FIX].std():.3f})')
    if log_col:
        la = df[log_col].to_numpy()
        la = la[np.isfinite(la)]
        print(f'  oracle log({p}): mean {la.mean():+.3f}  std {la.std():.3f}  '
              f'(geo-mean {p} = {np.exp(la.mean()):.3f})')

    feats = [c for c in ('snr_db', 'act_frac') if c in df.columns]
    print(f'\nWhat predicts the gap / the oracle {p}:')
    print(f'  {"y":<12} {"x":<10} {"pearson":>9} {"":<3} {"spearman":>9} {"":<3} '
          f'{"OLS-slope":>9} {"OLS-icpt":>9}  n')
    rows = [GAP, P_ORA] + ([log_col] if log_col else [])
    for y in rows:
        for xc in feats:
            pr, pp, sr, sp, m = correlate(df[y].to_numpy(), df[xc].to_numpy())
            slope, icpt, _, _ = ols_fit(df[xc].to_numpy(), df[y].to_numpy())
            print(f'  {y:<12} {xc:<10} {pr:>+9.3f} {stars(pp):<3} '
                  f'{sr:>+9.3f} {stars(sp):<3} {slope:>+9.4f} {icpt:>+9.4f}  {m}')
    print('\n  (* p<.05  ** p<.01  *** p<.001)')
    if log_col:
        print(f'  ({log_col} deploys as {p} = exp(slope*feature + icpt))')
    else:
        print(f'  ({P_ORA} deploys as {p} = slope*feature + icpt)')

    # Scatter: gap, oracle-p (and log for a) vs each blind feature + OLS line.
    fig, axes = plt.subplots(len(rows), len(feats),
                             figsize=(5 * len(feats), 3 * len(rows)),
                             squeeze=False)
    for i, y in enumerate(rows):
        for j, xc in enumerate(feats):
            ax = axes[i][j]
            xx, yy = df[xc].to_numpy(), df[y].to_numpy()
            ax.scatter(xx, yy, s=12, alpha=0.5)
            slope, icpt, _, _ = ols_fit(xx, yy)
            mfin = np.isfinite(xx) & np.isfinite(yy)
            if np.isfinite(slope) and mfin.sum() >= 2:
                xs = np.linspace(xx[mfin].min(), xx[mfin].max(), 50)
                ax.plot(xs, slope * xs + icpt, 'r-', lw=1.5)
            pr, pp, sr, sp, _ = correlate(yy, xx)
            ax.set_xlabel(xc)
            ax.set_ylabel(y)
            ax.set_title(f'r={pr:+.2f}{stars(pp)}  rho={sr:+.2f}{stars(sp)}',
                         fontsize=10)
    fig.suptitle(f'{p} headroom: gap & oracle-{p} vs blind features', y=1.0)
    fig.tight_layout()
    abl_out = out.rsplit('.', 1)[0] + '_ablation.png'
    fig.savefig(abl_out, dpi=120, bbox_inches='tight')
    print(f'\nSaved figure to {abl_out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?', default='sweep_params.csv')
    ap.add_argument('-o', '--out', default='sweep_stats.png')
    ap.add_argument('--feature', default='act_frac',
                    help='blind feature driving the deployed regression map')
    ap.add_argument('--degree', type=int, default=2,
                    help='polynomial degree for the s deployed fit (1 = linear); '
                         'the quadratic is reported next to the linear line')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Back-compat: older logs stored the sigmoid location as b (f = sigmoid(a*q-b));
    # recover the transition quantile s = b/a so they read under the s schema.
    if 'b' in df.columns and 's' not in df.columns:
        df['s'] = df['b'] / df['a'].replace(0, np.nan)
    if detect_ablation_metric(df) is not None:
        analyze_ablation(df, args.out)
        return
    metric = detect_metric(df)
    BEST, PRED, DELTA = f'best_{metric}', f'pred_{metric}', f'delta_{metric}'
    # pred = baseline (sample difficulty); delta = gain, summarised below.
    TARGETS = [PRED]
    # Drop rows where the metric failed on the baseline (best is guarded >= pred).
    df = df[np.isfinite(df[PRED]) & np.isfinite(df[BEST])]
    n = len(df)
    print(f'Loaded {n} samples from {args.csv}  (metric: {metric})\n')

    print('Summary (mean +/- std  [min, max]):')
    for col in PARAMS + [PRED, BEST, DELTA]:
        v = df[col].to_numpy()
        print(f'  {col:<13} {v.mean():+.3f} +/- {v.std():.3f}'
              f'   [{v.min():+.3f}, {v.max():+.3f}]')
    frac_gain = (df[DELTA] > 0).mean()
    print(f'\n  samples with positive gain: {frac_gain:.1%}')
    print(f'  mean gain:                  {df[DELTA].mean():+.4f}')

    print(f'\nCorrelations vs baseline ({PRED}):')
    print(f'  {"var":<13} {"target":<13} {"pearson":>9} {"":<3} '
          f'{"spearman":>9} {"":<3}  n')
    rows = PARAMS + [PRED, BEST]
    for var in rows:
        for tgt in TARGETS:
            if var == tgt:
                continue
            pr, pp, sr, sp, m = correlate(df[var].to_numpy(), df[tgt].to_numpy())
            print(f'  {var:<13} {tgt:<13} {pr:>+9.3f} {stars(pp):<3} '
                  f'{sr:>+9.3f} {stars(sp):<3}  {m}')
    print('\n  (* p<.05  ** p<.01  *** p<.001)')

    # Blind feature -> parameter predictability. These columns are written by
    # tf_mpol.py alongside the oracle params; if present, report how well each
    # blind feature predicts the params it is meant to drive (snr_db -> d,
    # act_frac -> the transition quantile s).
    features = [c for c in ('snr_db', 'act_frac', 'tail_snr') if c in df.columns]

    if features:
        dff = df.copy()
        print(f'\nBlind feature -> parameter predictability  (n={len(dff)}):')
        print(f'  {"feature":<10} {"param":<6} {"pearson":>9} {"":<3} '
              f'{"spearman":>9} {"":<3} {"OLS-slope":>9} {"OLS-icpt":>9}  n')
        for feat in features:
            for par in PARAMS:
                x, y = dff[feat].to_numpy(), dff[par].to_numpy()
                pr, pp, sr, sp, mm = correlate(x, y)
                slope, icpt, _, _ = ols_fit(x, y)
                print(f'  {feat:<10} {par:<6} {pr:>+9.3f} {stars(pp):<3} '
                      f'{sr:>+9.3f} {stars(sp):<3} {slope:>+9.4f} '
                      f'{icpt:>+9.4f}  {mm}')
        print('  (OLS: param ~= slope*feature + icpt)')

        # Multiple regression: snr_db and act_frac jointly -> each param. The
        # question is whether the deployed single-feature map leaves predictability
        # on the table: compare the joint OLS R^2 to the best single-feature OLS
        # R^2 (same method, so the delta is purely what the 2nd feature adds). The
        # per-feature coefficients are partial slopes (effect holding the other
        # feature fixed).
        mfeats = [c for c in ('snr_db', 'act_frac') if c in dff.columns]
        if len(mfeats) >= 2:
            print(f'\nMultiple regression  '
                  f'(param ~= {" + ".join(f"b_{f}*{f}" for f in mfeats)} + icpt, OLS):')
            hdr = ' '.join(f'{"b_" + f:>11}' for f in mfeats)
            print(f'  {"param":<6} {hdr} {"icpt":>11} {"R2":>7} {"R2_best1":>9}  n')
            for par in PARAMS:
                y = dff[par].to_numpy()
                coefs, icpt, r2, nn = multi_ols(dff[mfeats].to_numpy(), y)
                best1 = max(multi_ols(dff[[f]].to_numpy(), y)[2] for f in mfeats)
                cstr = ' '.join(f'{c:>+11.4f}' for c in coefs)
                print(f'  {par:<6} {cstr} {icpt:>+11.4f} {r2:>+7.3f} '
                      f'{best1:>+9.3f}  {nn}')
            print('  (R2_best1 = best single-feature OLS R2; R2 - R2_best1 = '
                  'gain from the 2nd feature)')

        # Deployed regression (cfg.regress): the single-feature map actually used
        # at test time. d and the transition quantile s are fit on the feature;
        # steepness a is NOT regressed -- it is fixed at a_fixed (the a_ablation
        # showed per-sample a is a weak second-order lever). Reported for each
        # candidate driver (args.feature first -- the deployed one -- then the
        # other) so their R^2 can be compared head to head before deciding which
        # feature to wire in.
        full = df.copy()
        subsets = {'d': full, 's': full}
        a_fixed = float(full['a'].median())
        drivers = [args.feature] + [f for f in ('act_frac', 'snr_db')
                                    if f in df.columns and f != args.feature]
        for driver in [d for d in drivers if d in df.columns]:
            print(f'\nDeployed regression  (param ~= slope*{driver} + icpt, '
                  f'OLS; a fixed at median):')
            print(f'  {"param":<6} {"slope":>10} {"icpt":>10} {"R2":>7}  n')
            coeffs = {}
            poly_coeffs = {}
            for par in ('d', 's'):
                sub = subsets[par]
                slope, icpt, r2, nn = ols_fit(sub[driver].to_numpy(),
                                              sub[par].to_numpy())
                if par == 'd':
                    # Deploy the mass-weighted fit: weight by suppressed mass
                    # (~ s, clipped to [0,1]) so the low-leverage samples where d
                    # is noise don't drag the line. Pooled R2 understates d.
                    lev = np.clip(sub['s'].to_numpy(), 0.0, 1.0)
                    ws, wi, wr2, _ = ols_fit_w(sub[driver].to_numpy(),
                                               sub[par].to_numpy(), lev)
                    ls, li, lr2, _ = lad_fit_w(sub[driver].to_numpy(),
                                               sub[par].to_numpy(), lev)
                    coeffs[par] = (ws, wi)
                    print(f'  {par:<6} {slope:>+10.4f} {icpt:>+10.4f} {r2:>+7.3f}  {nn}'
                          f'   [mass-wtd OLS: {ws:+.4f} {wi:+.4f} R2={wr2:+.3f} <- deployed]')
                    print(f'  {"":<6} {"":>10} {"":>10} {"":>7}  '
                          f'   [mass-wtd LAD: {ls:+.4f} {li:+.4f} R2={lr2:+.3f} '
                          f'(L2 R2; robust median-fit hedge for the d=0 pile-up)]')
                else:
                    coeffs[par] = (slope, icpt)
                    line = (f'  {par:<6} {slope:>+10.4f} {icpt:>+10.4f} '
                            f'{r2:>+7.3f}  {nn}')
                    # s is monotone-but-curved in the feature; report the deg>1
                    # fit's R2 gain next to the linear line. The linear map is what
                    # ships until the deploy code carries a polynomial s.
                    if par == 's' and args.degree > 1:
                        pc, pr2, _ = poly_fit(sub[driver].to_numpy(),
                                              sub[par].to_numpy(), args.degree)
                        poly_coeffs[par] = pc
                        cs = ' '.join(f'{c:+.4f}' for c in pc)
                        line += (f'   [deg{args.degree}: R2={pr2:+.3f} '
                                 f'(dR2={pr2 - r2:+.3f})  [{cs}]]')
                    print(line)
            leverage_split(full, driver, 'd')
            print(f'  a_fixed = {a_fixed:.3f} (median a; s is the knee quantile)')
            print('\n  Copy into config/adaptation/tf_mpol.yaml:')
            print('  # d fit is mass-weighted (weight ~ s); s is plain OLS')
            print('  regress:')
            print(f'    a_fixed: {a_fixed:.3f}')
            for par in ('d', 's'):
                slope, icpt = coeffs[par]
                lo, hi = CLAMP[par]
                print(f'    {par}: {{slope: {slope:.3f}, icpt: {icpt:.3f}, '
                      f'min: {lo}, max: {hi}}}')
                if par in poly_coeffs:
                    cs = ', '.join(f'{c:.4f}' for c in poly_coeffs[par])
                    print(f'    # {par} deg{args.degree} alt (needs a poly deploy '
                          f'map): poly: [{cs}]  # highest power first')
            lo, hi = CLAMP['a']
            print(f'    a: {{min: {lo}, max: {hi}}}')

        # Collapse test: refit each deployed param on the clean-derived true_*
        # feature next to the blind fit (same subset). Run per domain: if the blind
        # slope/icpt scatter across domains but the true-feature fit is stable (and
        # the EARS-D outlier rejoins the others), the cross-domain coefficient
        # spread is feature miscalibration, not a real param->feature relationship.
        feat_pairs = [(b, t) for b, t in (('act_frac', 'true_act_frac'),
                                          ('snr_db', 'true_snr_db'),
                                          ('tail_snr', 'true_tail_snr'))
                      if b in df.columns and t in df.columns]
        if feat_pairs:
            print('\nBlind vs. true feature -> param  (OLS; collapse test):')
            print(f'  {"param":<6} {"feature":<9} {"bl_slope":>9} {"bl_icpt":>9} '
                  f'{"bl_R2":>6}   {"tr_slope":>9} {"tr_icpt":>9} {"tr_R2":>6}  n')
            true_coeffs = {}
            for par in ('d', 's'):
                sub = subsets[par]
                for bfeat, tfeat in feat_pairs:
                    bs, bi, br2, _ = ols_fit(sub[bfeat].to_numpy(),
                                             sub[par].to_numpy())
                    ts, ti, tr2, nn = ols_fit(sub[tfeat].to_numpy(),
                                              sub[par].to_numpy())
                    true_coeffs[(par, tfeat)] = (ts, ti)
                    print(f'  {par:<6} {bfeat:<9} {bs:>+9.4f} {bi:>+9.4f} '
                          f'{br2:>+6.2f}   {ts:>+9.4f} {ti:>+9.4f} {tr2:>+6.2f}  {nn}')
            print('  (bl_* = blind feature fit, tr_* = true feature fit)')

            # Feature-ceiling config: the deployed map fit on the *true* activity.
            # Paste into config/adaptation/tf_mpol_exp.yaml; running regress mode
            # with this gives the PESQ a perfect activity estimator could reach,
            # bounding the headroom from improving the blind feature.
            if ('d', 'true_act_frac') in true_coeffs:
                print('\n  Feature-ceiling run -- copy into '
                      'config/adaptation/tf_mpol_exp.yaml (regress mode):')
                print('  regress:')
                print(f'    a_fixed: {a_fixed:.3f}')
                print('    feature: true_act_frac')
                for par in ('d', 's'):
                    ts, ti = true_coeffs[(par, 'true_act_frac')]
                    lo, hi = CLAMP[par]
                    print(f'    {par}: {{slope: {ts:.3f}, icpt: {ti:.3f}, '
                          f'min: {lo}, max: {hi}}}')
                lo, hi = CLAMP['a']
                print(f'    a: {{min: {lo}, max: {hi}}}')

        # Scatter grid: each blind feature (cols) vs each param (rows) + robust
        # fit line. s is clipped to a readable range so the knee outliers don't
        # flatten every other panel.
        rows_f = PARAMS
        figf, axf = plt.subplots(len(rows_f), len(features),
                                 figsize=(5 * len(features), 3 * len(rows_f)),
                                 squeeze=False)
        lev_all = np.clip(dff['s'].to_numpy(), 0.0, 1.0)  # d's leverage ~ mass
        for i, par in enumerate(rows_f):
            for j, feat in enumerate(features):
                ax = axf[i][j]
                x, y = dff[feat].to_numpy(), dff[par].to_numpy()
                pr, pp, sr, sp, _ = correlate(x, y)
                slope, icpt, _, _ = ols_fit(x, y)
                mfin = np.isfinite(x)
                if par == 'd':
                    # Size/shade points by leverage; overlay the mass-weighted
                    # line (deployed) next to the plain OLS line so it is visible
                    # that d tracks the feature exactly where it has leverage.
                    ax.scatter(x, y, s=4 + 40 * lev_all, alpha=0.35,
                               c=lev_all, cmap='viridis')
                    ws, wi, wr2, _ = ols_fit_w(x, y, lev_all)
                    ls, li, lr2, _ = lad_fit_w(x, y, lev_all)
                    if np.isfinite(slope) and mfin.sum() >= 2:
                        xs = np.linspace(x[mfin].min(), x[mfin].max(), 50)
                        ax.plot(xs, slope * xs + icpt, color='0.5', lw=1.3,
                                ls='--', label='OLS')
                        ax.plot(xs, ws * xs + wi, 'r-', lw=1.8,
                                label=f'mass-wtd OLS (R2={wr2:+.2f})')
                        if np.isfinite(ls):
                            ax.plot(xs, ls * xs + li, color='tab:blue', lw=1.6,
                                    ls=':', label=f'mass-wtd LAD (R2={lr2:+.2f})')
                    ax.legend(fontsize=7, loc='best')
                else:
                    ax.scatter(x, y, s=12, alpha=0.5)
                    if np.isfinite(slope) and mfin.sum() >= 2:
                        xs = np.linspace(x[mfin].min(), x[mfin].max(), 50)
                        ax.plot(xs, slope * xs + icpt, 'r-', lw=1.5,
                                label='linear')
                        # s is curved in the feature: overlay the deg>1 fit so the
                        # low-feature ramp the linear line misses is visible.
                        if par == 's' and args.degree > 1:
                            pc, pr2, _ = poly_fit(x, y, args.degree)
                            if np.all(np.isfinite(pc)):
                                ax.plot(xs, np.polyval(pc, xs), color='darkorange',
                                        lw=1.8, label=f'deg{args.degree} '
                                        f'(R2={pr2:+.2f})')
                                ax.legend(fontsize=7, loc='best')
                ax.set_xlabel(feat)
                ax.set_ylabel(par)
                if par == 's':
                    ax.set_ylim(-0.1, 1.5)
                ax.set_title(f'r={pr:+.2f}{stars(pp)}  rho={sr:+.2f}{stars(sp)}',
                             fontsize=10)
        figf.suptitle('Blind features vs. oracle parameters', y=1.0)
        figf.tight_layout()
        feat_out = args.out.rsplit('.', 1)[0] + '_features.png'
        figf.savefig(feat_out, dpi=120, bbox_inches='tight')
        print(f'\nSaved figure to {feat_out}')

    # True vs. estimated blind features: the clean-derived ground truth (true_*,
    # logged by tf_mpol.py) against the deployed blind estimate. The y=x line is
    # perfect calibration; a fitted slope != 1 is a scale error and an offset != 0
    # a bias -- the feature miscalibration that can masquerade as a domain-
    # dependent feature->param relationship. Run per dataset and compare across.
    pairs = [(est, tru) for est, tru in (('snr_db', 'true_snr_db'),
                                         ('act_frac', 'true_act_frac'),
                                         ('tail_snr', 'true_tail_snr'))
             if est in df.columns and tru in df.columns]
    if pairs:
        figt, axt = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5),
                                 squeeze=False)
        for j, (est, tru) in enumerate(pairs):
            ax = axt[0][j]
            x, y = df[est].to_numpy(), df[tru].to_numpy()
            ax.scatter(x, y, s=12, alpha=0.5)
            slope, icpt, r2, _ = ols_fit(x, y)
            pr, pp, sr, sp, _ = correlate(x, y)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 2:
                lo = float(min(x[m].min(), y[m].min()))
                hi = float(max(x[m].max(), y[m].max()))
                ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x')
                if np.isfinite(slope):
                    xs = np.linspace(x[m].min(), x[m].max(), 50)
                    ax.plot(xs, slope * xs + icpt, 'r-', lw=1.5,
                            label=f'fit: {slope:+.2f}x{icpt:+.2f}')
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect('equal')
            ax.set_xlabel(f'estimated ({est})')
            ax.set_ylabel(f'true ({tru})')
            ax.set_title(f'{est}:  r={pr:+.2f}{stars(pp)}  R2={r2:+.2f}  '
                         f'slope={slope:+.2f}', fontsize=10)
            ax.legend(fontsize=8)
        figt.suptitle('True vs. estimated blind features  '
                      '(y=x = perfect calibration)', y=1.02)
        figt.tight_layout()
        true_out = args.out.rsplit('.', 1)[0] + '_truefeat.png'
        figt.savefig(true_out, dpi=120, bbox_inches='tight')
        print(f'\nSaved figure to {true_out}')

    # Figure: params (rows) x targets (cols) scatter + linear fit.
    rows_fig = PARAMS
    fig, axes = plt.subplots(len(rows_fig), len(TARGETS),
                             figsize=(5 * len(TARGETS), 3 * len(rows_fig)),
                             squeeze=False)
    for i, var in enumerate(rows_fig):
        for j, tgt in enumerate(TARGETS):
            ax = axes[i][j]
            x, y = df[tgt].to_numpy(), df[var].to_numpy()
            ax.scatter(x, y, s=12, alpha=0.5)
            pr, pp, sr, sp, _ = correlate(x, y)
            mfin = np.isfinite(x) & np.isfinite(y)
            if mfin.sum() >= 2 and x[mfin].std() > 0:
                coef = np.polyfit(x[mfin], y[mfin], 1)
                xs = np.linspace(x[mfin].min(), x[mfin].max(), 50)
                ax.plot(xs, np.polyval(coef, xs), 'r-', lw=1.5)
            ax.set_xlabel(tgt)
            ax.set_ylabel(var)
            ax.set_title(f'r={pr:+.2f}{stars(pp)}  rho={sr:+.2f}{stars(sp)}',
                         fontsize=10)
    fig.suptitle(f'Optimal fit parameters vs. baseline {metric}', y=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'\nSaved figure to {args.out}')


if __name__ == '__main__':
    main()
