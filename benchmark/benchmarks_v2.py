"""Second batch: models that adjust the participant's own previous dot (the strongest first-batch benchmark).
All information is pre-meeting; parameters are re-estimated on an expanding window for every SEP."""
import numpy as np, pandas as pd, warnings
from protocol import fitset
from sklearn.linear_model import RidgeCV, LinearRegression
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
V1 = pd.read_csv('data/predictions_v1.csv', parse_dates=['meeting_date'])
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
D = D.join(V1[['B0_prior_own', 'B1a_prev_median', 'B1b_no_change']])
meetings = sorted(D.meeting_date.unique())
mac = D.groupby('meeting_date')[['target_mid', 'core_inflation', 'unrate']].first()
prevm = {m: (meetings[i - 1] if i else None) for i, m in enumerate(meetings)}
for c in ['target_mid', 'core_inflation', 'unrate']:
    D['d_' + c] = [mac.loc[m, c] - mac.loc[prevm[m], c] if prevm[m] is not None else 0.0 for m in D.meeting_date]
D['gap_med'] = D.B1a_prev_median - D.B0_prior_own        # how far he was from the committee last time
D['gap_now'] = D.target_mid - D.B0_prior_own              # his old path vs today's actual rate
D['hn'] = D.h.replace('LR', '4').astype(int)
D['new'] = (D.B0_prior_own == D.B1a_prev_median).astype(int)  # crude flag: fell back to median (newcomer / new year)
D['target'] = D.federal_funds_rate - D.B0_prior_own
ZLB = 0.125
X = ['d_target_mid', 'd_core_inflation', 'd_unrate', 'gap_med', 'gap_now', 'hn', 'voter', 'target_mid', 'unrate', 'core_inflation']
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[(D.meeting_date < t)].dropna(subset=X + ['target'])
    # B5: own previous path shifted by the actual policy change since the previous SEP (not LR)
    shift = np.where(cur.h == 'LR', 0.0, cur.d_target_mid)
    put('B5_prior_plus_policy_change', cur.index, np.maximum(ZLB, cur.B0_prior_own + shift))
    if hist.meeting_date.nunique() < 6: continue
    ok = cur.dropna(subset=X)
    # B6: ridge on the revision (y - own previous dot), one model per horizon group
    for hs in (['0'], ['1'], ['2', '3'], ['LR']):
        tr = fitset(hist[hist.h.isin(hs)]); te = ok[ok.h.isin(hs)]
        if len(tr) < 30 or te.empty: continue
        r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(tr[X], tr.target)
        put('B6_ridge_revision', te.index, np.maximum(ZLB, te.B0_prior_own + r.predict(te[X])))
        a = LinearRegression().fit(tr[['B0_prior_own']], tr.federal_funds_rate)   # B8: AR(1)-type shrink of own dot
        put('B8_ar1_own', te.index, np.maximum(ZLB, a.predict(te[['B0_prior_own']])))
    # B7: gradient boosting on the revision, all horizons pooled
    hf = fitset(hist)
    if len(hf) >= 150 and not ok.empty:
        g = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=0, n_jobs=2)
        g.fit(hf[X], hf.target)
        put('B7_xgb_revision', ok.index, np.maximum(ZLB, ok.B0_prior_own + g.predict(ok[X])))
out = V1.copy()
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v2.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
