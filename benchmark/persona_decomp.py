"""How much of a participant's dot is explained by who he/she is, by biography, and by new data?
Level  = dot - same-SEP committee median (where he/she sits).
Revision = dot - own previous dot for the same year (how he/she changes it). Horizons 1-3, SEPs 2012-2020. In-sample R^2."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
P = pd.read_csv('data/predictions_v6.csv', parse_dates=['meeting_date'])
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
F = pd.read_csv('data/features_external.csv', parse_dates=['meeting_date'])
S = pd.read_csv('data/text_tone.csv', parse_dates=['meeting_date'])
X = P.merge(D[['meeting_date', 'person_id', 'h', 'voter', 'target_mid', 'core_inflation', 'unrate']], on=['meeting_date', 'person_id', 'h'])
X = X.merge(F[['meeting_date', 'person_id', 'infl_exp', 'school', 'district_ugap']], on=['meeting_date', 'person_id'], how='left')
X['infl_exp'] = (X.infl_exp - X.groupby('meeting_date').infl_exp.transform('mean')).fillna(0); X['school'] = X.school.fillna('unknown')
last = S.sort_values('meeting_date')
X['tone'] = [float(last[(last.person_id == p) & (last.meeting_date < t)].tail(3).tone_rel.mean()) if ((last.person_id == p) & (last.meeting_date < t)).any() else 0.0
             for p, t in zip(X.person_id, X.meeting_date)]
X['level'] = X.federal_funds_rate - X.groupby(['meeting_date', 'h']).federal_funds_rate.transform('median')
X['rev'] = X.federal_funds_rate - X.B0_prior_own
meetings = sorted(X.meeting_date.unique()); prevm = {m: (meetings[i - 1] if i else None) for i, m in enumerate(meetings)}
mac = X.groupby('meeting_date')[['target_mid', 'core_inflation', 'unrate']].first()
for c in mac: X['d_' + c] = [mac.loc[m, c] - mac.loc[prevm[m], c] if prevm[m] is not None else 0 for m in X.meeting_date]
X['gap'] = X.B1a_prev_median - X.B0_prior_own
L = X[X.h.isin(['1', '2', '3'])].dropna(subset=['level', 'rev'])
# keep rows where the participant had a dot for the same target year in the previous SEP (not a newcomer / new-year fallback)
own = set(zip(X.person_id, X.meeting_date, X.proj_year))
L = L[[(p, prevm[t], y) in own for p, t, y in zip(L.person_id, L.meeting_date, L.proj_year)]]
spec = {'这个人本身（每人一个固定偏移）': 'C(person_id)',
        '履历与身份（经历的通胀、学派、辖区失业、投票权）': 'infl_exp + C(school) + district_ugap + voter',
        '发言鹰鸽分（之前 3 次会议）': 'tone',
        '新的经济数据（利率、通胀、失业的变化）': 'd_target_mid + d_core_inflation + d_unrate',
        '上次与委员会的差距': 'gap'}
rows = []
for name, rhs in spec.items():
    r_level = smf.ols(f'level ~ C(h) + {rhs}', L).fit().rsquared - smf.ols('level ~ C(h)', L).fit().rsquared
    r_rev = smf.ols(f'rev ~ C(h) + {rhs}', L).fit().rsquared - smf.ols('rev ~ C(h)', L).fit().rsquared
    rows.append((name, r_level, r_rev))
R = pd.DataFrame(rows, columns=['因素', '解释“站位”（比中位数高多少）的比例', '解释“修改”（比上次改多少）的比例'])
print('n =', len(L), '| SEP-meeting effects alone explain revision R2 =', round(smf.ols('rev ~ C(h) + C(meeting_date)', L).fit().rsquared, 3))
print(R.round(3).to_string(index=False)); R.round(3).to_csv('data/persona_decomposition.csv', index=False)
