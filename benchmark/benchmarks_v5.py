"""Fifth batch: personal features from biographies (inflation experience, PhD school) and regional conditions.
Tests (1) whether they explain where a person sits relative to the committee and (2) whether they improve revision models."""
import numpy as np, pandas as pd, warnings
from protocol import fitset
from sklearn.linear_model import RidgeCV
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
V4 = pd.read_csv('data/predictions_v4.csv', parse_dates=['meeting_date'])
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
for c in V4.columns:
    if c.startswith('B'): D[c] = V4[c]
F = pd.read_csv('data/features_external.csv', parse_dates=['meeting_date'])
F.loc[F.participant_name == 'James Lyon', ['birth_year', 'age', 'infl_exp']] = np.nan; F.loc[F.participant_name == 'James Lyon', 'school'] = 'unknown'
D = D.merge(F[['meeting_date', 'person_id', 'infl_exp', 'age', 'school', 'district_ugap']], on=['meeting_date', 'person_id'], how='left')
D['infl_exp_rel'] = D.infl_exp - D.groupby('meeting_date').infl_exp.transform('mean')
D['infl_exp_rel'] = D.infl_exp_rel.fillna(0); D['age'] = D.age.fillna(D.age.mean())
for s in ['freshwater', 'saltwater', 'other_phd', 'no_phd']: D['sch_' + s] = (D.school == s).astype(int)
D['dev'] = D.federal_funds_rate - D.groupby(['meeting_date', 'h']).federal_funds_rate.transform('median')
# (1) level: deviation from the same-SEP committee median on personal traits (horizons 1-3 where dispersion is largest)
L = D[D.h.isin(['1', '2', '3'])]
m = smf.ols('dev ~ infl_exp_rel + sch_freshwater + sch_saltwater + sch_other_phd + district_ugap + voter + C(h)', data=L).fit(cov_type='cluster', cov_kwds={'groups': L.person_id.astype('category').cat.codes})
print(m.summary().tables[1]); print('R2', round(m.rsquared, 3), 'n', int(m.nobs))
# (2) revision ridge with the new traits
meetings = sorted(D.meeting_date.unique()); ZLB = 0.125
mac = D.groupby('meeting_date')[['target_mid', 'core_inflation', 'unrate']].first()
prevm = {mm: (meetings[i - 1] if i else None) for i, mm in enumerate(meetings)}
for c in ['target_mid', 'core_inflation', 'unrate']:
    D['d_' + c] = [mac.loc[mm, c] - mac.loc[prevm[mm], c] if prevm[mm] is not None else 0.0 for mm in D.meeting_date]
D['gap_med'] = D.B1a_prev_median - D.B0_prior_own; D['gap_now'] = D.target_mid - D.B0_prior_own
D['hn'] = D.h.replace('LR', '4').astype(int)
X = ['d_target_mid', 'd_core_inflation', 'd_unrate', 'gap_med', 'gap_now', 'hn', 'voter', 'target_mid', 'unrate', 'core_inflation',
     'infl_exp_rel', 'sch_freshwater', 'sch_saltwater', 'sch_other_phd', 'district_ugap']
XL = ['infl_exp_rel', 'sch_freshwater', 'sch_saltwater', 'sch_other_phd', 'district_ugap', 'voter']
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[D.meeting_date < t].dropna(subset=X + ['B0_prior_own'])
    if hist.meeting_date.nunique() < 6: continue
    ok = cur.dropna(subset=X + ['B0_prior_own'])
    for hs in (['0'], ['1'], ['2', '3'], ['LR']):
        tr = fitset(hist[hist.h.isin(hs)]); te = ok[ok.h.isin(hs)]
        if len(tr) < 30 or te.empty: continue
        r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(tr[X], tr.federal_funds_rate - tr.B0_prior_own)
        put('B6d_ridge_bio_region', te.index, np.maximum(ZLB, te.B0_prior_own + r.predict(te[X])))
out = V4.copy()
for k, v in preds.items(): out[k] = v

out.to_csv('data/predictions_v5.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
