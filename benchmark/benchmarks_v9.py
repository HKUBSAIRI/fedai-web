"""Ninth batch: expert forecasts of the median dot, and two more features for B25. Specified before the first run.
  B28  expert two-step: lag + (expected median - previous median) for the same calendar year (LR by LR). The expected median
       is the average of the median responses of the New York Fed's Survey of Primary Dealers and Survey of Market
       Participants to "What are your expectations for the medians of FOMC participants' target federal funds rate
       projections in the SEP?" (fetch_nyfed_survey.py; asked before every SEP since March 2016). Responses are due about a
       week before the meeting, but the results are published about three weeks after it: an expert-forecast baseline, not
       a public-information one. No parameters.
  B28b equal-weight average of B25 and B28.
  B25b B25 + the committee's median revision at the previous SEP for the same year (forecast revisions are smoothed,
       Nordhaus 1987; monotone non-decreasing) + the participant's past share of changed dots at that horizon (shrunk
       toward one half). Same settings as B25.
Rows without an own previous dot for the same year keep the naive forecast; SEPs without a survey (before March 2016)
have no B28 or B28b. Floored at 0.125."""
import warnings, numpy as np, pandas as pd
from xgboost import XGBRegressor
from protocol import fitset
import news
warnings.filterwarnings('ignore')
V8 = pd.read_csv('data/predictions_v8.csv', parse_dates=['meeting_date'])
D = news.panel(V8); ZLB = news.ZLB
D = D.join(V8[[c for c in V8.columns if c[0] == 'B' and c not in D.columns]])
meetings = sorted(D.meeting_date.unique())
D['rev'] = D.ffr - D.lag
for c in news.NEWS + news.GAPS: D[c] = D[c].fillna(0.0)
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v
def final(te, f): return np.maximum(ZLB, np.where(te.has_lag, te.lag + f, te.lag))

# ---------- B28, B28b: expert forecasts of the median dot
E = pd.read_csv('data/nyfed_expected_median.csv', dtype={'target': str})
E = E.groupby(['sep_year', 'sep_month', 'target']).expected_median.mean()           # average of the two surveys
D['expert'] = [E.get((t.year, t.month, 'LR' if py == -1 else str(int(py))), np.nan) for t, py in zip(D.meeting_date, D.proj_year)]
ok = D.expert.notna()
put('B28_expert_two_step', D.index[ok], final(D[ok], (D.expert - D.prev_med)[ok]))
put('B28b_combo_b25_expert', D.index[ok], (D.B25_xgb_l1_news[ok] + preds['B28_expert_two_step'][ok]) / 2)

# ---------- B25b: B25 + committee momentum + the participant's propensity to change
com = D[D.has_lag].groupby(['sep_index', 'proj_year']).rev.median()
D['com_prev_rev'] = [com.get((s - 1, py), 0.0) for s, py in zip(D.sep_index, D.proj_year)]
D['changed'] = (D.rev.abs() > 1e-9).astype(float)
D['own_change_rate'] = 0.5
for _, g in D[D.has_lag].sort_values('meeting_date').groupby(['person_id', 'h']):
    D.loc[g.index, 'own_change_rate'] = ((g.changed.shift().fillna(0).cumsum() + 1) / (np.arange(len(g)) + 2)).values
MONO = {'d_pol': 1, 'd_fwd': 1, 'd_spf': 1, 'd_core_inflation': 1, 'd_unrate': -1, 'd_gdp_growth_y': 1,
        'gap_med': 1, 'gap_now': 1, 'gap_mkt': 1, 'gap_spf': 1, 'hn': 0, 'com_prev_rev': 1, 'own_change_rate': 0}
FX = news.NEWS + news.GAPS + ['hn', 'com_prev_rev', 'own_change_rate']
cache = {}
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[(D.meeting_date < t) & D.has_lag]
    if hist.meeting_date.nunique() < 6: continue
    hf = fitset(hist)
    if len(hf) not in cache:
        cache[len(hf)] = XGBRegressor(objective='reg:absoluteerror', n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                                      colsample_bytree=0.8, min_child_weight=10, monotone_constraints=tuple(MONO[c] for c in FX),
                                      random_state=0, n_jobs=2).fit(hf[FX], hf.rev)
    put('B25b_xgb_l1_news_momentum', cur.index, final(cur, cache[len(hf)].predict(cur[FX])))

out = V8.copy()
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v9.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
