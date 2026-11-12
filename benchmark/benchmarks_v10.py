"""Tenth batch: the CS-style baselines (2026-11-06). Every setting below was written
down before the first run and is not changed after seeing results for 2016 or later. The choice among the pre-registered
variants (the threshold of B31e, the variant of B32) uses only rolling-origin forecasts for the 12 SEPs of 2013-03 to
2015-12 (each from a model fitted on earlier SEPs), scored on the calendar-year cells whose own previous dot is above the
lower bound (lag >= 0.375). The choice is the same in both protocols.

Notation: lag = own previous dot for the same year. z1 = rd25(change of the SPF bill path for year H since the previous SEP),
z2 = rd25(change of the Treasury forward), z3 = rd25(c / 2), c = change of the committee median for year H at the previous
SEP (median at t-1 minus median at t-2). rd25 rounds a revision to the 25 bp grid, exact midpoints toward zero. S(x, d) =
sign(x) max(|x| - d, 0) is the soft threshold. Longer run: every method in this batch keeps the own previous dot.
Forecasts floored at 0.125; rows without an own previous dot for the same year keep the naive forecast.

  B30a  lag + z1                                                    no parameters
  B30b  lag + median(z1, z2, z3)                                    no parameters
  B30c  B30b, except when the three signals disagree and their mean zbar lies strictly between 0 and +-25 bp: then
        k = round(N |zbar| / 25 bp) of the N participants with an own previous dot revise by 25 bp toward zbar and the
        others keep their dot, ranked by the own share of changed dots at that horizon (shrunk toward 1/2, as in B25b)
        + 1/4 sign(zbar) hawkishness (own past distance to the committee median of the same SEP and year, sum / (n + 5))
  B31   lag + S(d_spf, 0.125)                                       no parameters (beta = 1)
  B31e  lag + beta S(d_spf, delta); beta by censored least absolute deviations (Powell 1984; fitted dot
        max(0.125, lag + beta S)) on the Y1-Y3 cells, pooled; delta in {0, 0.0625, 0.125, 0.25} by rolling origin
  B32   linear margin + monotone trees: XGBoost on the revision (absolute loss, settings of B25) with base margin
        beta S(d_spf, 0.125); features of B25b + voter + hawkishness + hawkishness x sign(d_spf); trees non-decreasing in
        the news, the gaps and the committee's previous revision (non-increasing in the change of unemployment), so they
        cannot undo the margin; 25 seeds, median of the 25 forecasts. Variants: training cells all Y0-Y3 ("all") or only
        lag >= 0.375 ("unc"); beta = 1 ("b1") or beta of B31e at delta = 0.125 ("bhat"); output continuous ("cont") or on
        the grid, lag + rd25(revision) ("grid"). B32 = the grid output of the variant with the lowest rolling-origin MAE
  B33   online combination of B0, B22b, B27, B25b, B30b, B31, B32 and B16 by AdaHedge (de Rooij et al. 2014; no tuning
        parameter). One weight vector, updated after every SEP from its calendar-year cells. Loss a: committee level,
        mean over years of |mean implied revision - change of the committee median| (public on the meeting day);
        loss b: MAE over the SEP's cells. Forecast lag + rd25(weighted mean - lag).
        B33b_frozen: as b, but the weights stop learning after FEDAI_TRAIN_END. The weights are state, updated from
        realised outcomes by a rule without free parameters; they keep learning after 2015 in a and b
  B39   median(B30b, B32, B27)
  B30d  lower-bound rule on top of B39: if the policy rate is at 0.125 and the last meeting with a higher
        rate was at most 365 days ago, Y0 and Y1 = 0.125, Y2 = 0.125 + median(Y2 dot - policy rate) over the
        estimation cells with the rate at the lower bound; Y3 and LR unchanged. Never fires in 2012-2015 or 2016-2019
Outputs: data/predictions_v10.csv; data/v10_selection.csv (rolling-origin scores of the variants); data/v10_validation.csv
(2013-03 to 2015-12: MAE by SEP and stress episode, direction of the implied committee revision); data/v10_weights.csv
(B33 weights by SEP); data/v10_diagnostics.csv (beta with and without the censoring correction, correlation of the SPF
news with revisions on all and on unconstrained cells, rank-1 check of the deviations from the median). The last three use
2012-2015 only. Nothing here looks at 2016 or later; scoring is done by evaluate_unified.py."""
import hashlib, os, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from protocol import fitset, may_learn
import news
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
print('benchmarks_v10.py sha256', hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16])
V9 = pd.read_csv('data/predictions_v9.csv', parse_dates=['meeting_date'])
D = news.panel(V9); ZLB = news.ZLB
D = D.join(V9[[c for c in V9.columns if c[0] == 'B' and c not in D.columns]])
meetings = sorted(D.meeting_date.unique())
SEL_FROM, SEL_END = pd.Timestamp('2013-03-01'), pd.Timestamp('2015-12-31')     # rolling origins for the choices
EST_END = pd.Timestamp('2015-12-31')                                            # diagnostics use 2012-2015 only
MIN_SEPS, DELTAS, UNC = 4, [0.0, 0.0625, 0.125, 0.25], 0.375 - 1e-9   # 4 earlier SEPs with own lags: first origin 2013-03
SEEDS = range(int(os.environ.get('V10_SEEDS', 25)))                      # V10_SEEDS=1 only for a quick smoke test
D['rev'] = D.ffr - D.lag
for c in news.NEWS + news.GAPS: D[c] = D[c].fillna(0.0)
CAL = (D.h != 'LR').values
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = np.asarray(v, float)
def final(te, f): return np.maximum(ZLB, np.where(te.has_lag, te.lag + f, te.lag))
def rd(x, step=0.25):
    x = np.asarray(x, float); return np.sign(x) * np.floor(np.abs(x) / step + 0.5 - 1e-9) * step
def soft(x, d): x = np.asarray(x, float); return np.sign(x) * np.maximum(np.abs(x) - d, 0.0)
def cal0(df, v): return np.where(df.h.values != 'LR', v, 0.0)
def src(c): return preds[c] if c in preds else D[c]

# ---------- committee medians, momentum, propensity to revise, hawkishness (state: realised dots of earlier SEPs only)
med = D.groupby(['sep_index', 'proj_year']).ffr.median()
mget = lambda s, py: med.get((s, py), np.nan)
D['c_prev'] = np.nan_to_num([mget(s - 1, py) - mget(s - 2, py) for s, py in zip(D.sep_index, D.proj_year)])
D['c_now'] = [mget(s, py) - mget(s - 1, py) for s, py in zip(D.sep_index, D.proj_year)]   # realised; losses only
com = D[D.has_lag].groupby(['sep_index', 'proj_year']).rev.median()                        # as in B25b
D['com_prev_rev'] = [com.get((s - 1, py), 0.0) for s, py in zip(D.sep_index, D.proj_year)]
D['changed'] = (D.rev.abs() > 1e-9).astype(float)
D['own_change_rate'] = 0.5
for _, g in D[D.has_lag].sort_values('meeting_date').groupby(['person_id', 'h']):
    D.loc[g.index, 'own_change_rate'] = ((g.changed.shift().fillna(0).cumsum() + 1) / (np.arange(len(g)) + 2)).values
D['dev'] = D.ffr - [mget(s, py) for s, py in zip(D.sep_index, D.proj_year)]
per = D[CAL].groupby(['person_id', 'sep_index']).dev.agg(['sum', 'count']).reset_index().sort_values('sep_index')
per['cs'] = per.groupby('person_id')['sum'].cumsum() - per['sum']; per['cn'] = per.groupby('person_id')['count'].cumsum() - per['count']
hawk = dict(zip(zip(per.person_id, per.sep_index), per.cs / (per.cn + 5)))
D['hawk'] = [hawk.get((p, s), 0.0) for p, s in zip(D.person_id, D.sep_index)]
D['hawk_x'] = D.hawk * np.sign(D.d_spf)

# ---------- B30a, B30b, B30c: no parameters
D['z1'] = cal0(D, rd(D.d_spf)); D['z2'] = cal0(D, rd(D.d_fwd)); D['z3'] = cal0(D, rd(D.c_prev / 2))
Z = D[['z1', 'z2', 'z3']].values
D['c30b'] = np.median(Z, axis=1)
put('B30a_grid_spf', D.index, final(D, D.z1))
put('B30b_grid_median_signals', D.index, final(D, D.c30b))
c30c = D.c30b.copy()
for _, g in D[CAL & D.has_lag].groupby(['sep_index', 'proj_year']):
    z = g[['z1', 'z2', 'z3']].iloc[0].values; zbar = z.mean()
    if np.ptp(z) > 0 and 1e-9 < abs(zbar) < 0.25 - 1e-9:
        k = int(np.floor(len(g) * abs(zbar) / 0.25 + 0.5))
        order = (g.own_change_rate + 0.25 * np.sign(zbar) * g.hawk).sort_values(ascending=False, kind='mergesort').index
        c30c.loc[g.index] = 0.0; c30c.loc[order[:k]] = 0.25 * np.sign(zbar)
put('B30c_grid_budget', D.index, final(D, c30c))

# ---------- B31: soft threshold, beta = 1
put('B31_soft_threshold_spf', D.index, final(D, cal0(D, soft(D.d_spf, 0.125))))

# ---------- estimated: B31e (censored LAD), B32 (linear margin + monotone trees)
BETAS = np.round(np.arange(0, 3.0001, 0.01), 2)
def fit_beta(tr, delta, censored=True):
    x = soft(tr.d_spf.values, delta); l = tr.lag.values; y = tr.ffr.values
    P = l[:, None] + np.outer(x, BETAS)
    if censored: P = np.maximum(ZLB, P)
    loss = np.abs(y[:, None] - P).mean(axis=0)
    best = np.flatnonzero(loss <= loss.min() + 1e-12)
    return float(BETAS[best[len(best) // 2]])
def beta_rows(df): return df[df.has_lag & df.h.isin(['1', '2', '3'])]
FX = news.NEWS + news.GAPS + ['hn', 'voter', 'com_prev_rev', 'own_change_rate', 'hawk', 'hawk_x']
MONO = {'d_pol': 1, 'd_fwd': 1, 'd_spf': 1, 'd_core_inflation': 1, 'd_unrate': -1, 'd_gdp_growth_y': 1, 'gap_med': 1,
        'gap_now': 1, 'gap_mkt': 1, 'gap_spf': 1, 'hn': 0, 'voter': 0, 'com_prev_rev': 1, 'own_change_rate': 0, 'hawk': 0, 'hawk_x': 0}
VARIANTS = [('all', 'b1'), ('unc', 'b1'), ('all', 'bhat'), ('unc', 'bhat')]       # order = tie-break (simpler first)
def xgb(X, y, margin, seed):
    return XGBRegressor(objective='reg:absoluteerror', base_score=0.0, n_estimators=300, max_depth=2, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, monotone_constraints=tuple(MONO[c] for c in FX),
                        random_state=seed, n_jobs=1).fit(X, y, base_margin=margin)
cache, beta_log = {}, []
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[(D.meeting_date < t) & D.has_lag]
    if hist.meeting_date.nunique() < MIN_SEPS: continue
    hf = fitset(hist); key = len(hf)
    if key not in cache:
        betas = {d: fit_beta(beta_rows(hf), d) for d in DELTAS}
        models = {}
        for cells, base in VARIANTS:
            tr = hf[hf.h != 'LR']; tr = tr if cells == 'all' else tr[tr.lag >= UNC]
            beta = 1.0 if base == 'b1' else betas[0.125]
            margin = beta * soft(tr.d_spf, 0.125)
            models[(cells, base)] = (beta, Parallel(n_jobs=4)(delayed(xgb)(tr[FX], tr.rev, margin, s) for s in SEEDS))
        cache[key] = (betas, models)
        beta_log.append(dict(fitted_through=hf.meeting_date.max().date(), n=len(hf), **{f'beta_delta_{d}': b for d, b in betas.items()}))
    betas, models = cache[key]
    for d in DELTAS: put(f'B31e_delta_{d}', cur.index, final(cur, cal0(cur, betas[d] * soft(cur.d_spf, d))))
    for (cells, base), (beta, ms) in models.items():
        r = cal0(cur, np.median([m.predict(cur[FX], base_margin=beta * soft(cur.d_spf, 0.125)) for m in ms], axis=0))
        put(f'B32_{cells}_{base}_cont', cur.index, final(cur, r)); put(f'B32_{cells}_{base}_grid', cur.index, final(cur, rd(r)))
    print(t.date(), end=' ', flush=True)
print()

# ---------- choices by rolling origin on 2013-03 to 2015-12, unconstrained calendar-year cells
SEL = D[(D.meeting_date >= SEL_FROM) & (D.meeting_date <= SEL_END) & D.has_lag & CAL & (D.lag >= UNC)]
def score(col):
    p = src(col)[SEL.index]; assert p.notna().all(), col
    return float((p - SEL.ffr).abs().mean())
sel_rows = [dict(method='B31e', option=f'delta={d}', col=f'B31e_delta_{d}', mae_bp=100 * score(f'B31e_delta_{d}')) for d in DELTAS]
sel_rows += [dict(method='B32', option=f'{c}/{b}', col=f'B32_{c}_{b}_grid', mae_bp=100 * score(f'B32_{c}_{b}_grid')) for c, b in VARIANTS]
sel_rows += [dict(method='reference', option=c, col=c, mae_bp=100 * score(c)) for c in ['B0_prior_own', 'B22b_prior_plus_spf_change', 'B31_soft_threshold_spf']]
S = pd.DataFrame(sel_rows)
def choose(m, prefer):
    s = S[S.method == m].set_index('col').mae_bp; best = s.min()
    return next(c for c in prefer if s[c] <= best + 1e-9)
b31e = choose('B31e', [f'B31e_delta_{d}' for d in [0.125, 0.0, 0.0625, 0.25]])       # ties: the prior value 0.125 first
b32 = choose('B32', [f'B32_{c}_{b}_grid' for c, b in VARIANTS])
S['chosen'] = S.col.isin([b31e, b32]); S['n_cells'] = len(SEL); S.to_csv('data/v10_selection.csv', index=False)
print(S.round(2).to_string(index=False))
put('B31e_censored_lad', D.index, preds[b31e]); put('B32_linear_tree', D.index, preds[b32])

# ---------- B33: AdaHedge over a fixed pool
class AdaHedge:
    def __init__(s, K): s.K, s.L, s.Delta = K, np.zeros(K), 0.0
    def weights(s):
        w = (s.L <= s.L.min() + 1e-12).astype(float) if s.Delta <= 0 else np.exp(-np.log(s.K) / s.Delta * (s.L - s.L.min()))
        return w / w.sum()
    def update(s, l):
        w = s.weights(); h = w @ l
        if s.Delta <= 0: m = l[w > 0].min()
        else:
            eta = np.log(s.K) / s.Delta; m = l.min() - np.log(w @ np.exp(-eta * (l - l.min()))) / eta
        s.Delta += max(h - m, 0.0); s.L += l
EXP = ['B0_prior_own', 'B22b_prior_plus_spf_change', 'B27_combo_news', 'B25b_xgb_l1_news_momentum', 'B30b_grid_median_signals',
       'B31_soft_threshold_spf', 'B32_linear_tree', 'B16_herd_to_median']
M = pd.concat([src(c) for c in EXP], axis=1, keys=EXP)
ready = [t for t in meetings if ((D.meeting_date == t) & D.has_lag).any() and M[(D.meeting_date == t) & D.has_lag].notna().all().all()]
H = {k: AdaHedge(len(EXP)) for k in ('a', 'b', 'f')}; wlog = []
for t in meetings:
    if t < ready[0]: continue
    cur = D[D.meeting_date == t]; X = M.loc[cur.index].apply(lambda c: c.fillna(cur.lag)).values; lag = cur.lag.values
    for k, name in (('a', 'B33a_hedge_committee_loss'), ('b', 'B33b_hedge_cell_loss'), ('f', 'B33b_frozen')):
        w = H[k].weights(); put(name, cur.index, final(cur, cal0(cur, rd(X @ w - lag))))
        wlog.append(dict(meeting_date=t.date(), version=k, **dict(zip(EXP, w))))
    ok = (cur.has_lag & (cur.h != 'LR')).values; g = cur[ok]; Xg = X[ok]
    lb = np.abs(Xg - g.ffr.values[:, None]).mean(axis=0)
    la = [np.abs((Xg[(g.proj_year == py).values] - g.lag.values[(g.proj_year == py).values, None]).mean(axis=0) - gg.c_now.iloc[0])
          for py, gg in g.groupby('proj_year') if pd.notna(gg.c_now.iloc[0])]
    H['a'].update(np.mean(la, axis=0)); H['b'].update(lb)
    if may_learn(t): H['f'].update(lb)
pd.DataFrame(wlog).to_csv('data/v10_weights.csv', index=False)

# ---------- B39: median of three
put('B39_median_of_three', D.index, np.median(np.column_stack([src('B30b_grid_median_signals'), src('B32_linear_tree'), src('B27_combo_news')]), axis=1))

# ---------- B30d: lower-bound rule
mac = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']); mac['mid'] = (mac.target_lower + mac.target_upper) / 2
above = mac[mac.mid > ZLB + 1e-9].meeting_date.sort_values().values
def fires(t, rate):
    last = above[above < np.datetime64(t)]
    return rate <= ZLB + 1e-9 and len(last) > 0 and (pd.Timestamp(t) - pd.Timestamp(last[-1])).days <= 365
D['zlb_rule'] = [fires(t, r) for t, r in zip(D.meeting_date, D.target_mid)]
offset = {}
for t in D.meeting_date[D.zlb_rule].unique():
    est = fitset(D[D.meeting_date < t]); est = est[(est.target_mid <= ZLB + 1e-9) & (est.h == '2')]
    offset[t] = float((est.ffr - est.target_mid).median())
for base, name in (('B39_median_of_three', 'B30d_zlb_on_B39'),):
    v = src(base).values.copy(); f = D.zlb_rule.values
    v[f & D.h.isin(['0', '1']).values] = ZLB
    y2 = f & (D.h == '2').values
    v[y2] = np.maximum(ZLB, ZLB + D.meeting_date[y2].map(offset).values)
    put(name, D.index, v)
print('B30d fires at', sorted({t.date() for t in D.meeting_date[D.zlb_rule]}), 'Y2 offsets', {k.date(): round(v, 3) for k, v in offset.items()})

# ---------- training-period validation (2013-03 to 2015-12) and diagnostics (2012-2015); no data after 2015
VAL = ['B0_prior_own', 'B22b_prior_plus_spf_change', 'B27_combo_news', 'B25b_xgb_l1_news_momentum', 'B30a_grid_spf',
       'B30b_grid_median_signals', 'B30c_grid_budget', 'B31_soft_threshold_spf', 'B31e_censored_lad', 'B32_linear_tree',
       'B33a_hedge_committee_loss', 'B33b_hedge_cell_loss', 'B39_median_of_three']
T = D[(D.meeting_date >= SEL_FROM) & (D.meeting_date <= SEL_END) & D.has_lag & CAL].copy()
EPIS = {'2013-06 taper': ('2013-06-01', '2013-06-30'), '2013-09 taper': ('2013-09-01', '2013-09-30'),
        '2014 path down': ('2014-01-01', '2014-12-31'), '2015-12 liftoff': ('2015-12-01', '2015-12-31')}
rows = []
for m in VAL:
    p = src(m)[T.index]; Tm = T[p.notna()]; p = p[p.notna()]                  # B25b and B33 start in 2013-06
    e = (p - Tm.ffr).abs(); e0 = (Tm.B0_prior_own - Tm.ffr).abs(); unc = Tm.lag >= UNC
    r = dict(method=m, first_sep=Tm.meeting_date.min().date(), mae_all=100 * e.mean(), naive_all=100 * e0.mean(),
             mae_unc=100 * e[unc].mean(), naive_unc=100 * e0[unc].mean())
    imp = (p - Tm.lag).groupby([Tm.meeting_date, Tm.proj_year]).mean(); act = Tm.groupby(['meeting_date', 'proj_year']).c_now.first()
    moved = act.abs() > 1e-9
    r['direction_right_when_median_moved'] = float((np.sign(imp[moved]) == np.sign(act[moved])).mean())
    for ep, (lo, hi) in EPIS.items():
        s = (Tm.meeting_date >= lo) & (Tm.meeting_date <= hi)
        r[f'{ep}: mae - naive'] = 100 * (e[s].mean() - e0[s].mean())
    rows.append(r)
VALT = pd.DataFrame(rows); VALT.to_csv('data/v10_validation.csv', index=False)
pd.set_option('display.width', 250); print('\n== rolling origin, 2013-03 to 2015-12, calendar-year cells (bp)'); print(VALT.round(2).to_string(index=False))

DG = D[(D.meeting_date <= EST_END) & D.has_lag & CAL]
unc = DG.lag >= UNC
diag = [dict(item='cells 2012-2015 (calendar years)', value=len(DG)), dict(item='share with lag at the lower bound', value=float((~unc).mean()))]
for h in ['0', '1', '2', '3']:
    g = DG[DG.h == h]; gu = g[g.lag >= UNC]
    diag += [dict(item=f'corr(d_spf, revision) Y{h}, all cells', value=g.d_spf.corr(g.rev)),
             dict(item=f'corr(d_spf, revision) Y{h}, lag >= 0.375', value=gu.d_spf.corr(gu.rev) if len(gu) > 5 else np.nan),
             dict(item=f'beta Y{h}, censored LAD, delta 0.125', value=fit_beta(g, 0.125)),
             dict(item=f'beta Y{h}, plain LAD (no censoring), delta 0.125', value=fit_beta(g, 0.125, censored=False))]
diag += [dict(item='beta Y1-Y3 pooled, censored LAD, delta 0.125', value=fit_beta(beta_rows(DG), 0.125)),
         dict(item='beta Y1-Y3 pooled, plain LAD, delta 0.125', value=fit_beta(beta_rows(DG), 0.125, censored=False))]
# rank-1 check (B38): does a smoothed position (b_i + a_i lambda_j, fitted by alternating least squares on earlier SEPs)
# predict the distance to the committee median better than the last observed distance?
def rank1(Xm, iters=200):
    R, C = Xm.shape; obs = ~np.isnan(Xm); b = np.nanmean(Xm, axis=1); a = np.ones(R); lam = np.zeros(C)
    for _ in range(iters):
        for j in range(C):
            o = obs[:, j]; lam[j] = (a[o] * (Xm[o, j] - b[o])).sum() / max((a[o] ** 2).sum(), 1e-9)
        for i in range(R):
            o = obs[i]; A = np.column_stack([np.ones(o.sum()), lam[o]])
            b[i], a[i] = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(2), A.T @ Xm[i, o], rcond=None)[0]
    return b, a, lam
err_last, err_r1 = [], []
for t in [m for m in meetings if SEL_FROM <= m <= SEL_END]:
    past = D[(D.meeting_date < t) & CAL]; cur = D[(D.meeting_date == t) & D.has_lag & CAL]
    W = past.pivot_table(index='person_id', columns=['sep_index', 'proj_year'], values='dev')
    b, a, lam = rank1(W.values); pos = {p: i for i, p in enumerate(W.index)}; col = {c: j for j, c in enumerate(W.columns)}
    for _, r in cur.iterrows():
        j = col.get((r.sep_index - 1, r.proj_year)); i = pos.get(r.person_id)
        if i is None or j is None or np.isnan(W.values[i, j]): continue
        err_last.append(abs(r.dev - W.values[i, j])); err_r1.append(abs(r.dev - (b[i] + a[i] * lam[j])))
diag += [dict(item='distance to median at t: MAE of the last observed distance (bp)', value=100 * np.mean(err_last)),
         dict(item='distance to median at t: MAE of the rank-1 fit (bp)', value=100 * np.mean(err_r1)),
         dict(item='rank-1 check cells', value=len(err_last))]
DIAG = pd.DataFrame(diag); DIAG.to_csv('data/v10_diagnostics.csv', index=False)
pd.DataFrame(beta_log).to_csv('data/v10_beta_log.csv', index=False)
print('\n== diagnostics, 2012-2015'); print(DIAG.round(3).to_string(index=False))

out = V9.copy()
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v10.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
