"""Eighth batch: news-based and machine-learning models, a two-step committee model and combinations (11.5 meeting).
The models predict the revision r - lag, where lag is the participant's own previous dot for the same calendar year.
  B22  lag + change of the Treasury forward for the end of year H since the previous SEP (no parameters; LR: lag)
  B22b lag + change of the SPF median bill-rate path for the end of year H since the previous SEP (no parameters; LR: lag)
  B23  two-step: (1) the committee's median revision per horizon from the news (ridge on committee-level data, per horizon
       group); (2) lag + the predicted committee revision
  B24  median (quantile) regression of the revision on the news and gaps, per horizon group; the L1 penalty is chosen on
       the last quarter of the estimation SEPs
  B25  XGBoost with absolute-error loss on the revision, pooled over horizons, monotone in the news and gaps
  B26  ordinal: revision classes -50, -25, 0, +25, +50 bp (rounded, capped), multinomial logit per horizon group; the
       forecast adds the median class (its median past revision)
  B27  equal-weight combination of B0, B5, B8, B22 and B22b (B12 plus the market and survey news)
  B27b combination weighted by inverse past MSE per horizon group; B27c stacking: non-negative least squares weights per
       horizon group, shrunk halfway toward equal weights; B13b per horizon group, the member with the lowest past MAE.
       Pool: POOL below
News: policy rate, Treasury forward and SPF path for year H, core inflation, unemployment, GDP growth (changes since the
previous SEP). Gaps: committee median, policy rate, market and SPF minus the lag. Pre-meeting information only; parameters
re-estimated on an expanding window or fixed after FEDAI_TRAIN_END; combination weights learn only from SEPs that may be
used for estimation. Rows without an own previous dot for the same year keep the naive forecast. Floored at 0.125."""
import warnings, numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV, QuantileRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from protocol import fitset, may_learn
import news
warnings.filterwarnings('ignore')
V7 = pd.read_csv('data/predictions_v7.csv', parse_dates=['meeting_date'])
D = news.panel(V7); ZLB = news.ZLB
D = D.join(V7[[c for c in V7.columns if c[0] == 'B' and c not in D.columns]])
meetings = sorted(D.meeting_date.unique())
GROUPS = (['0'], ['1'], ['2', '3'], ['LR'])
F = news.NEWS + news.GAPS
D['rev'] = D.ffr - D.lag
for c in F: D[c] = D[c].fillna(0.0)
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v
def final(te, f): return np.maximum(ZLB, np.where(te.has_lag, te.lag + f, te.lag))

# ---------- B22, B22b: no parameters
cal = (D.h != 'LR').values
put('B22_prior_plus_market_change', D.index, final(D, np.where(cal, D.d_fwd, 0.0)))
put('B22b_prior_plus_spf_change', D.index, final(D, np.where(cal, D.d_spf, 0.0)))

# ---------- estimated models
def varying(tr, cols): return [c for c in cols if tr[c].std() > 1e-9]
def ridge_committee(tr, cols):                                  # B23 step 1 on committee-level rows
    cols = varying(tr, cols); sc = StandardScaler().fit(tr[cols])
    r = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(sc.transform(tr[cols]), tr.rev)
    return lambda te: r.predict(sc.transform(te[cols]))
def qreg(tr, cols):
    cols = varying(tr, cols); sc = StandardScaler().fit(tr[cols]); seps = sorted(tr.meeting_date.unique())
    cut = seps[int(len(seps) * 0.75)]; a, b = tr[tr.meeting_date < cut], tr[tr.meeting_date >= cut]
    q = lambda al: QuantileRegressor(quantile=.5, alpha=al, solver='highs')
    best = min([0.0, 0.002, 0.01, 0.05], key=lambda al: np.abs(b.rev - q(al).fit(sc.transform(a[cols]), a.rev).predict(sc.transform(b[cols]))).mean())
    m = q(best).fit(sc.transform(tr[cols]), tr.rev)
    return lambda te: m.predict(sc.transform(te[cols]))
def ordinal(tr, cols):
    cols = varying(tr, cols); sc = StandardScaler().fit(tr[cols])
    k = np.clip(np.round(tr.rev / 0.25), -2, 2).astype(int)
    val = tr.rev.groupby(k).median()                             # each class's median past revision
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr[cols]), k)
    def f(te):
        cp = np.cumsum(m.predict_proba(sc.transform(te[cols])), axis=1)
        return np.array([val[m.classes_[np.argmax(c >= 0.5)]] for c in cp])
    return f
MONO = {'d_pol': 1, 'd_fwd': 1, 'd_spf': 1, 'd_core_inflation': 1, 'd_unrate': -1, 'd_gdp_growth_y': 1,
        'gap_med': 1, 'gap_now': 1, 'gap_mkt': 1, 'gap_spf': 1, 'hn': 0}
FX = F + ['hn']
cache = {}
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[(D.meeting_date < t) & D.has_lag]
    if hist.meeting_date.nunique() < 6: continue
    hf = fitset(hist)
    # B23 step 1: committee median revision per SEP and horizon, with the SEP's news
    com = hf.groupby(['meeting_date', 'h']).agg(rev=('rev', 'median'), **{c: (c, 'first') for c in news.NEWS}).reset_index()
    for hs in GROUPS:
        tr, te = hf[hf.h.isin(hs)], cur[cur.h.isin(hs)]
        if len(tr) < 30 or te.empty: continue
        key = (tuple(hs), len(tr))
        if key not in cache: cache[key] = (ridge_committee(com[com.h.isin(hs)], news.NEWS), qreg(tr, F), ordinal(tr, F))
        for name, f in zip(['B23_two_step_committee', 'B24_median_regression', 'B26_ordinal_logit'], cache[key]):
            put(name, te.index, final(te, f(te)))
    key = ('xgb', len(hf))
    if key not in cache:
        cache[key] = XGBRegressor(objective='reg:absoluteerror', n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=10, monotone_constraints=tuple(MONO[c] for c in FX),
                                  random_state=0, n_jobs=2).fit(hf[FX], hf.rev)
    put('B25_xgb_l1_news', cur.index, final(cur, cache[key].predict(cur[FX])))
    print(t.date(), end=' ', flush=True)
print()

# ---------- combinations
def src(c): return preds[c] if c in preds else D[c]
put('B27_combo_news', D.index, pd.concat([src(c) for c in ['B0_prior_own', 'B5_prior_plus_policy_change', 'B8_ar1_own',
                                                             'B22_prior_plus_market_change', 'B22b_prior_plus_spf_change']], axis=1).mean(axis=1).values)
POOL = ['B0_prior_own', 'B5_prior_plus_policy_change', 'B8_ar1_own', 'B16b_herd_plus_policy', 'B17_hetero_ar1', 'B18_hetero_arx',
        'B19_var_sep_variables', 'B19b_var_horizons', 'B20_panel_var', 'B21_panel_varx', 'B22_prior_plus_market_change',
        'B22b_prior_plus_spf_change', 'B23_two_step_committee', 'B24_median_regression', 'B25_xgb_l1_news', 'B26_ordinal_logit']
POOL = [c for c in POOL if c in preds or c in D.columns]
M = pd.concat([src(c) for c in POOL], axis=1, keys=POOL)
past = []                                                       # rows of SEPs whose outcomes may be used
for t in meetings:
    cur = D[D.meeting_date == t]
    if past:
        P = D.loc[np.concatenate(past)]; P = P[P.has_lag]
        for hs in GROUPS:
            te = cur[cur.h.isin(hs)]
            ok = [c for c in POOL if M.loc[te.index, c].notna().all()]
            tr = P[P.h.isin(hs)]; tr = tr[M.loc[tr.index, ok].notna().all(axis=1)]     # past SEPs where every member exists
            if te.empty or not ok or tr.meeting_date.nunique() < 4: continue
            E = M.loc[tr.index, ok].sub(tr.ffr, axis=0)
            w = 1 / (E ** 2).mean(); w = w / w.sum()
            put('B27b_combo_inverse_mse', te.index, M.loc[te.index, ok].values @ w.values)
            b = nnls(M.loc[tr.index, ok].values, tr.ffr.values)[0]
            b = b / b.sum() if b.sum() > 0 else np.full(len(ok), 1 / len(ok))
            put('B27c_combo_stacking', te.index, M.loc[te.index, ok].values @ (0.5 * b + 0.5 / len(ok)))
            best = E.abs().mean().idxmin()
            put('B13b_select_by_horizon', te.index, M.loc[te.index, best].values)
    if may_learn(t): past.append(cur.index.values)

out = V7.copy()
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v8.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
