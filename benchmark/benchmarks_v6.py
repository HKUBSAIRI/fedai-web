"""Sixth batch: herding toward the committee (Rulke & Tillmann 2023, 'Do FOMC Members Herd?').
B16  own previous dot + a * (previous committee median - own previous dot)
B16b B5 (own previous dot + policy change since the previous SEP) + a * (previous median - own previous dot)
a is estimated per horizon group by least squares on all earlier SEPs (expanding window), bounded to [0, 1]."""
import numpy as np, pandas as pd
from protocol import fitset
P = pd.read_csv('data/predictions_v5.csv', parse_dates=['meeting_date'])
meetings = sorted(P.meeting_date.unique()); ZLB = 0.125
P['gap'] = P.B1a_prev_median - P.B0_prior_own
P['rev'] = P.federal_funds_rate - P.B0_prior_own
P['rev5'] = P.federal_funds_rate - P.B5_prior_plus_policy_change
GROUPS = (['0'], ['1'], ['2', '3'], ['LR'])
for c in ['B16_herd_to_median', 'B16b_herd_plus_policy']: P[c] = np.nan
alpha_log = []
for t in meetings:
    hist = P[P.meeting_date < t]; cur = P[P.meeting_date == t]
    for hs in GROUPS:
        te = cur[cur.h.isin(hs)]; tr = fitset(hist[hist.h.isin(hs)].dropna(subset=['gap', 'rev', 'rev5']))
        tr = tr[tr.gap != 0]
        if len(tr) < 20 or te.empty: continue
        a = float(np.clip((tr.gap * tr.rev).sum() / (tr.gap ** 2).sum(), 0, 1))
        b = float(np.clip((tr.gap * tr.rev5).sum() / (tr.gap ** 2).sum(), 0, 1))
        P.loc[te.index, 'B16_herd_to_median'] = np.maximum(ZLB, te.B0_prior_own + a * te.gap)
        P.loc[te.index, 'B16b_herd_plus_policy'] = np.maximum(ZLB, te.B5_prior_plus_policy_change + b * te.gap)
        alpha_log.append((t, '+'.join(hs), a, b))
A = pd.DataFrame(alpha_log, columns=['meeting_date', 'h', 'alpha', 'alpha_with_policy'])
print(A[A.meeting_date >= '2014'].groupby('h')[['alpha', 'alpha_with_policy']].agg(['mean', 'min', 'max']).round(2))
# too few earlier SEPs to estimate a: fall back to the un-herded version so every method covers the same rows
P['B16_herd_to_median'] = P.B16_herd_to_median.fillna(P.B0_prior_own)
P['B16b_herd_plus_policy'] = P.B16b_herd_plus_policy.fillna(P.B5_prior_plus_policy_change)
P.drop(columns=['gap', 'rev', 'rev5']).to_csv('data/predictions_v6.csv', index=False)
print(P[['B16_herd_to_median', 'B16b_herd_plus_policy']].notna().sum().to_dict())
