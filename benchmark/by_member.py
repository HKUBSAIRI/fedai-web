"""Accuracy by tenure (the participant's k-th SEP with a dot) and by participant, test period 2014-2020."""
import numpy as np, pandas as pd
P = pd.read_csv('data/predictions_v6.csv', parse_dates=['meeting_date'])
P['k'] = P.groupby('person_id').meeting_date.rank(method='dense').astype(int)        # k-th SEP of this person (from 2012)
T = P[P.meeting_date >= '2014-01-01'].copy()
M = ['B0_prior_own', 'B5_prior_plus_policy_change', 'B14_combo_plus_text_h0', 'B6_ridge_revision']
T['dev_med'] = (T.federal_funds_rate - T.groupby(['meeting_date', 'h']).federal_funds_rate.transform('median')).abs()
T['bucket'] = pd.cut(T.k, [0, 1, 4, 8, 16, 100], labels=['第1次', '第2-4次', '第5-8次', '第9-16次', '第17次以后'])
tab = T.groupby('bucket', observed=True).apply(lambda g: pd.Series({**{m: np.mean(np.abs(g[m] - g.federal_funds_rate)) for m in M}, 'n': len(g),
                                                                       'people': g.person_id.nunique()}))
tab['n'] = tab.n.astype(int); tab['people'] = tab.people.astype(int)
print(tab.round(3).to_string()); tab.round(3).to_csv('data/by_tenure.csv')
per = T.groupby('participant_name').apply(lambda g: pd.Series({'SEPs': g.meeting_date.nunique(), 'n': len(g),
       **{m: np.mean(np.abs(g[m] - g.federal_funds_rate)) for m in M},
       'avg_abs_dev_from_median': g.dev_med.mean() }))
per = per[per.SEPs >= 6].sort_values('B14_combo_plus_text_h0')
per[['SEPs', 'n']] = per[['SEPs', 'n']].astype(int)
print(per.round(3).to_string()); per.round(3).to_csv('data/by_member.csv')
print('corr(B14 error, member SEP count):', round(np.corrcoef(per.SEPs, per.B14_combo_plus_text_h0)[0, 1], 2), '| corr(B14 error, distance from median):', round(np.corrcoef(per.avg_abs_dev_from_median, per.B14_combo_plus_text_h0)[0, 1], 2))
