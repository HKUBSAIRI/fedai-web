"""Narrative benchmark: length-adjusted comparison and breakdowns by year.
AI answers run longer than the real ones, and longer answers cover more points. The adjustment compares two methods on the same
reference answer (one fixed effect per answer) and controls for log(answer length / real answer length):
    score ~ method + log(pred_words / ref_words) + C(answer)
Usage: python narr_length.py <scores csv> [<scores csv> ...]   (results/narrative_scores_<period>.csv)"""
import sys, numpy as np, pandas as pd
import statsmodels.formula.api as smf
ORDER = ['N0', 'N1', 'L_ctx', 'L_prev', 'L_named', 'ORACLE']

def common_rows(path):
    S = pd.read_csv(path); S = S[S.final.notna()]
    W = S.pivot_table(index='key', columns='method', values='final')
    keep = W.dropna(subset=ORDER).index                       # rows every method answered (as in narr_report.py)
    S = S[S.key.isin(keep)].copy(); S['lr'] = np.log(S.pred_words / S.ref_words); S['year'] = S.meeting_date.str[:4]
    return S, W.loc[keep]

def adjusted(S, a, b, score='final'):
    """Length-adjusted difference a - b on the same answers, with its p-value (standard errors clustered by answer)."""
    D = S[S.method.isin([a, b])].copy(); D['x'] = (D.method == a).astype(int)
    m = smf.ols(f'{score} ~ x + lr + C(key)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.key)[0]})
    return m.params['x'], m.pvalues['x']

for path in sys.argv[1:]:
    S, W = common_rows(path)
    ai = S[S.method == 'L_prev'].set_index('key')
    print(f'== {path}: {W.shape[0]} answers, AI/real length median {(ai.pred_words / ai.ref_words).median():.2f}')
    for a, b in [('L_prev', 'N0'), ('L_named', 'N0'), ('L_named', 'L_prev')]:
        raw = (W[a] - W[b]).mean()
        parts = ', '.join(f'{s} {adjusted(S, a, b, s)[0]:+.3f}' for s in ['position', 'coverage', 'no_contradiction'])
        d, p = adjusted(S, a, b)
        print(f'  {a} - {b}: raw {raw:+.3f}; length-adjusted {d:+.3f} (p={p:.3f}); parts: {parts}')
    short = ai.index[ai.pred_words <= 1.2 * ai.ref_words]
    print(f'  AI answers at most 1.2x the real length: {len(short)}; L_prev {W.loc[short, "L_prev"].mean():.3f}, N0 {W.loc[short, "N0"].mean():.3f}')
    Y = S.pivot_table(index=['year', 'key'], columns='method', values='final').groupby(level=0)
    T = pd.DataFrame({'n': Y.size(), 'N0': Y['N0'].mean(), 'L_prev': Y['L_prev'].mean(), 'L_named': Y['L_named'].mean()})
    T['L_prev-N0'] = T.L_prev - T.N0; T['named gain'] = T.L_named - T.L_prev
    print('  by year:\n' + T.round(3).to_string())
