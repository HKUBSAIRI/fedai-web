"""Third batch: market expectations (Treasury forward rates), combinations, and personal features.
Pre-meeting information only; expanding-window estimation."""
import numpy as np, pandas as pd, warnings
from protocol import fitset, may_learn
from sklearn.linear_model import RidgeCV
warnings.filterwarnings('ignore')
V2 = pd.read_csv('data/predictions_v2.csv', parse_dates=['meeting_date'])
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
D = D.join(V2.drop(columns=[c for c in V2.columns if c in D.columns or c == 'proj_year']))
D['proj_year'] = V2.proj_year
meetings = sorted(D.meeting_date.unique())
ZLB = 0.125
# ---------- Treasury curve the day before each meeting (yields are not revised)
MAT = {'DTB3': .25, 'DTB6': .5, 'DGS1': 1, 'DGS2': 2, 'DGS3': 3, 'DGS5': 5, 'DGS10': 10}
Y = None
for s in MAT:
    f = pd.read_csv(f'data/fred_{s}.csv', parse_dates=['observation_date']).rename(columns={'observation_date': 'date', s: s})
    f[s] = pd.to_numeric(f[s], errors='coerce'); Y = f if Y is None else Y.merge(f, on='date', how='outer')
Y = Y.sort_values('date').set_index('date').ffill()
def curve(day):
    row = Y[Y.index < day].iloc[-1]
    return np.array(list(MAT.values())), row[list(MAT)].values.astype(float)
def fwd(tau, m, y):           # instantaneous forward from linear interpolation of y*tau
    yt = m * y; e = 0.05
    a, b = max(tau - e, m[0]), tau + e
    return (np.interp(b, m, yt) - np.interp(a, m, yt)) / (b - a)
mk = {}
for t in meetings:
    m, y = curve(pd.Timestamp(t)); yr = pd.Timestamp(t).year
    for h in range(4):
        tau = (pd.Timestamp(f'{yr + h}-12-31') - pd.Timestamp(t)).days / 365.25
        mk[(t, str(h))] = fwd(max(tau, .1), m, y)
    mk[(t, 'LR')] = (10 * y[-1] - 5 * y[-2]) / 5      # 5y5y forward
D['market'] = [mk[(t, h)] for t, h in zip(D.meeting_date, D.h)]
D['B9_market_forward'] = np.maximum(ZLB, D.market)
# ---------- revision features (same as benchmarks_v2.py)
mac = D.groupby('meeting_date')[['target_mid', 'core_inflation', 'unrate']].first()
prevm = {m: (meetings[i - 1] if i else None) for i, m in enumerate(meetings)}
for c in ['target_mid', 'core_inflation', 'unrate']:
    D['d_' + c] = [mac.loc[m, c] - mac.loc[prevm[m], c] if prevm[m] is not None else 0.0 for m in D.meeting_date]
D['gap_med'] = D.B1a_prev_median - D.B0_prior_own
D['gap_now'] = D.target_mid - D.B0_prior_own
D['hn'] = D.h.replace('LR', '4').astype(int)
# ---------- personal features
_aff = pd.read_csv('data/roles.csv', parse_dates=['meeting_date'])[['meeting_date', 'person_id', 'affiliation']].rename(columns={'affiliation': 'role_aff'})
D['governor'] = (~D[['meeting_date', 'person_id']].merge(_aff, on=['meeting_date', 'person_id'], how='left').role_aff.fillna('').str.contains('Reserve Bank')).astype(int).values
D['tenure'] = D.groupby('person_id').meeting_date.rank(method='dense') - 1          # earlier SEPs of this person
D['dev_med'] = D.federal_funds_rate - D.B1a_prev_median                             # realized deviation (used only from the past)
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v
def src(c): return preds[c] if c in preds else (D[c] if c in D.columns else pd.Series(np.nan, index=D.index))
past_err = {}                                   # per method, per horizon: list of past MAEs, for selection
CANDS = ['B0_prior_own', 'B5_prior_plus_policy_change', 'B6_ridge_revision', 'B8_ar1_own', 'B9b_market_plus_person']
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[D.meeting_date < t]
    # B9b: market forward + person's average past gap to the market (shrunk)
    if len(hist):
        g = (hist.federal_funds_rate - hist.market).groupby([hist.person_id, hist.h]).agg(['sum', 'count'])
        off = (g['sum'] / (g['count'] + 3)).to_dict()
        put('B9b_market_plus_person', cur.index, np.maximum(ZLB, cur.market + [off.get((p, h), 0.0) for p, h in zip(cur.person_id, cur.h)]))
    # personal hawkishness so far: mean past deviation from the previous-SEP median, shrunk
    hk = hist.groupby('person_id').dev_med.agg(['sum', 'count'])
    D.loc[cur.index, 'hawk'] = cur.person_id.map((hk['sum'] / (hk['count'] + 5))).fillna(0).values
    # B6b: ridge on the revision with personal features and the market signal
    X = ['d_target_mid', 'd_core_inflation', 'd_unrate', 'gap_med', 'gap_now', 'hn', 'voter', 'target_mid', 'unrate', 'core_inflation',
         'governor', 'tenure', 'hawk', 'mkt_gap']
    D.loc[cur.index, 'mkt_gap'] = (cur.market - cur.B0_prior_own).values
    hist = D[D.meeting_date < t]; cur = D[D.meeting_date == t]
    if hist.meeting_date.nunique() >= 6:
        tr0 = fitset(hist).dropna(subset=X + ['B0_prior_own']); te0 = cur.dropna(subset=X + ['B0_prior_own'])
        for hs in (['0'], ['1'], ['2', '3'], ['LR']):
            tr = tr0[tr0.h.isin(hs)]; te = te0[te0.h.isin(hs)]
            if len(tr) < 30 or te.empty: continue
            r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(tr[X], tr.federal_funds_rate - tr.B0_prior_own)
            put('B6b_ridge_personal_market', te.index, np.maximum(ZLB, te.B0_prior_own + r.predict(te[X])))
    # B12: equal-weight combination of the strong simple benchmarks
    put('B12_combo_B0_B5_B8', cur.index, cur[['B0_prior_own', 'B5_prior_plus_policy_change', 'B8_ar1_own']].mean(axis=1, skipna=True).values)
    # B13: per horizon, the candidate with the lowest MAE over the previous SEPs (needs >= 4 scored SEPs)
    for h in cur.h.unique():
        te = cur[cur.h == h]
        scores = {c: np.mean(past_err[(c, h)]) for c in CANDS if len(past_err.get((c, h), [])) >= 4}
        best = min(scores, key=scores.get) if scores else 'B5_prior_plus_policy_change'
        put('B13_select_by_horizon', te.index, src(best).loc[te.index].values)
    # record errors of this SEP for future selection (frozen after FEDAI_TRAIN_END)
    for c in (CANDS if may_learn(t) else []):
        s = src(c)
        for h in cur.h.unique():
            te = cur[cur.h == h]; e = (s.loc[te.index] - te.federal_funds_rate).abs().dropna()
            if len(e): past_err.setdefault((c, h), []).append(e.mean())
out = V2.copy()
out['B9_market_forward'] = D.B9_market_forward
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v3.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
