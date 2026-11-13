"""Persona analysis: how much of a participant's deviation from the committee's move can be predicted from his/her own record.

Split (as in compare_agent.py): for meeting t and target year H, c = median over participants of the actual revision r - lag
(lag = own dot for the same target at the participant's preceding SEP); the individual deviation is dev = r - lag - c.
The two-step ceiling max(0.125, lag + c) knows the committee's move exactly; its error is made only of dev. The agent is at
this ceiling, so dev is the part a persona can add.

Data-card models predict dev from information available before meeting t: the participant's own past dots and deviations,
the previous committee median, the floor, the role. Parameters are estimated on the 2012-2015 SEPs only and fixed afterwards;
the card itself (averages over the participant's past SEPs) keeps updating meeting by meeting. Specified before the first run:
  M0  ceiling: dev_hat = 0
  M1  herding toward the median: dev_hat = b * s, s = lag - previous committee median (LAD slope, no intercept)
  M2  return to the own typical stance: dev_hat = b * (s - s_bar), s_bar = own mean past s, shrunk to 0 (2 pseudo-SEPs)
  M3  habit: dev_hat = b * d_bar, d_bar = own mean past deviation, shrunk to 0 (4 pseudo-SEPs)
  M4  persistence: dev_hat = b * d_last, the own deviation at the preceding SEP (same horizon group)
  M5  floor compression (no parameters): when the committee cuts (c < 0) from a median at least 25 bp above the floor, every
      distance to the floor shrinks in proportion: 0.125 + (lag - 0.125) * (m_prev + c - 0.125) / (m_prev - 0.125)
  M6  own co-movement (no estimated parameters): dev_hat = (beta - 1) * c, beta = own past slope of r - lag on c, shrunk
      to 1 (4 pseudo-SEPs of a 25 bp move)
  M7  data card, LAD: dev on [s, s - s_bar, d_bar, d_last, (beta - 1) c, M5's dev, governor, chair], per horizon group
      (Y0, Y1, Y2-Y3, LR), L1 penalty chosen on the last quarter of the estimation SEPs
  M8  data card, monotone shallow trees: XGBoost with absolute-error loss on the M7 features + horizon, depth 2
Forecast: max(0.125, lag + c + rd25(dev_hat)); the unrounded forecasts are reported separately.
Reference (not a method): S  static persona fitted on the test sample itself (median deviation per participant and horizon
group), the most a fixed per-person shift could give.
Metrics on the unified cells: MAE (bp) on three test samples; paired test against M0 (absolute errors, two-way clustered by
participant and meeting, t reference distribution, Holm across M1-M8); MAE of the individual part as in compare_agent.py.
--feasible <predictions_v10.csv>: the same cards on top of B30b's committee move instead of the actual one.
Usage (in docs/benchmark/work/frozen_2015): python ../../code/persona_deviation.py data/predictions_v9.csv [--feasible <csv>]
Writes data/persona_metrics.csv, data/persona_facts.csv, data/persona_cards.csv, data/persona_predictions.csv."""
import argparse, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser(); ap.add_argument('pred'); ap.add_argument('--feasible'); a = ap.parse_args()
ZLB, TRAIN_END = 0.125, pd.Timestamp('2015-12-31')
SAMPLES = {'2016-2020': lambda d: (d.year >= 2016) & (d.year <= 2020),
           'excl. June 2020': lambda d: (d.year >= 2016) & (d.year <= 2020) & ~((d.year == 2020) & (d.month == 6)),
           '2016-2019': lambda d: (d.year >= 2016) & (d.year <= 2019)}
GROUP = {'0': '0', '1': '1', '2': '23', '3': '23', 'LR': 'LR'}
rd25 = lambda x: np.round(np.asarray(x, float) / 0.25) * 0.25

# ---------- cells, lag, committee move and deviation (as in evaluate_unified.py / compare_agent.py)
P = pd.read_csv(a.pred, parse_dates=['meeting_date']); P['h'] = P.h.astype(str)
P = P[P.federal_funds_rate.notna()].reset_index(drop=True)
own = P[['person_id', 'meeting_date']].drop_duplicates().sort_values('meeting_date'); prev_own = {}
for pid, g in own.groupby('person_id'):
    d = list(g.meeting_date); prev_own.update({(pid, x): y for x, y in zip(d[1:], d[:-1])})
dots = P.set_index(['person_id', 'meeting_date', 'proj_year']).federal_funds_rate
P['lag'] = [dots.get((p, prev_own.get((p, t)), y), np.nan) if prev_own.get((p, t)) is not None else np.nan
            for p, t, y in zip(P.person_id, P.meeting_date, P.proj_year)]
C = P[P.lag.notna() & P.B1a_prev_median.notna()].copy().sort_values(['meeting_date', 'person_id', 'proj_year']).reset_index(drop=True)
C['g'] = C.h.map(GROUP); C['hn'] = C.h.map({'0': 0, '1': 1, '2': 2, '3': 3, 'LR': 4})
C['rev'] = C.federal_funds_rate - C.lag
C['c'] = C.groupby(['meeting_date', 'proj_year']).rev.transform('median')
C['dev'] = C.rev - C.c
C['s'] = C.lag - C.B1a_prev_median
roles = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date']).drop_duplicates(['meeting_date', 'person_id'])
C = C.merge(roles[['meeting_date', 'person_id', 'governor', 'chair']], on=['meeting_date', 'person_id'], how='left')
C[['governor', 'chair']] = C[['governor', 'chair']].fillna(0).astype(float)
print(f'cells {len(C)}; meetings {C.meeting_date.nunique()}; participants {C.person_id.nunique()}')

# ---------- the data card: only the participant's SEPs before t, same horizon group
def card(C):
    out = {k: np.zeros(len(C)) for k in ['s_bar', 'd_bar', 'd_last', 'beta']}
    for (pid, g), G in C.groupby(['person_id', 'g']):
        G = G.sort_values('meeting_date')
        per = G.groupby('meeting_date').agg(s=('s', 'mean'), dev=('dev', 'mean'), rc=('rev', lambda x: 0.0), n=('s', 'size'))
        rcsum = G.assign(rc=G.rev * G.c, cc=G.c ** 2).groupby('meeting_date')[['rc', 'cc']].sum()
        S = per.s.cumsum().shift(1).fillna(0); N = pd.Series(np.arange(len(per)), index=per.index)
        D = per.dev.cumsum().shift(1).fillna(0); last = per.dev.shift(1).fillna(0)
        RC = rcsum.rc.cumsum().shift(1).fillna(0); CC = rcsum.cc.cumsum().shift(1).fillna(0)
        V, kb = 0.25 ** 2, 4
        m = G.meeting_date.values
        out['s_bar'][G.index] = (S / (N + 2)).reindex(m).values
        out['d_bar'][G.index] = (D / (N + 4)).reindex(m).values
        out['d_last'][G.index] = last.reindex(m).values
        out['beta'][G.index] = ((RC + kb * V) / (CC + kb * V)).reindex(m).values
    for k, v in out.items(): C[k] = v
    return C
C = card(C)

def c_features(C, c):
    """Features that depend on the committee move (the actual one, or a forecast of it)."""
    f = pd.DataFrame(index=C.index)
    f['betac'] = (C.beta - 1) * c
    room = C.B1a_prev_median - ZLB
    comp = ZLB + (C.lag - ZLB) * (C.B1a_prev_median + c - ZLB) / room.where(room > 0, np.nan)
    f['dev5'] = np.where((c < 0) & (room >= 0.25), comp - C.lag - c, 0.0)
    return f
F = c_features(C, C.c); C['betac'], C['dev5'] = F.betac, F.dev5
C['s_rel'] = C.s - C.s_bar
FX = ['s', 's_rel', 'd_bar', 'd_last', 'betac', 'dev5', 'governor', 'chair']
tr = C[C.meeting_date <= TRAIN_END]

# ---------- fit on 2012-2015 only
def lad_slope(x, y):
    return float(QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=False, solver='highs').fit(x.values.reshape(-1, 1), y.values).coef_[0])
SLOPE = {m: {g: lad_slope(tr[tr.g == g][col], tr[tr.g == g].dev) for g in GROUP.values()} for m, col in
         [('M1', 's'), ('M2', 's_rel'), ('M3', 'd_bar'), ('M4', 'd_last')]}
def lad_card(T):
    cols = [c for c in FX if T[c].std() > 1e-9]; sc = StandardScaler().fit(T[cols]); seps = sorted(T.meeting_date.unique())
    cut = seps[int(len(seps) * 0.75)]; A, B = T[T.meeting_date < cut], T[T.meeting_date >= cut]
    q = lambda al: QuantileRegressor(quantile=0.5, alpha=al, solver='highs')
    best = min([0.0, 0.002, 0.01, 0.05], key=lambda al: np.abs(B.dev - q(al).fit(sc.transform(A[cols]), A.dev).predict(sc.transform(B[cols]))).mean())
    m = q(best).fit(sc.transform(T[cols]), T.dev)
    return lambda X: m.predict(sc.transform(X[cols])), best, dict(zip(cols, m.coef_))
M7 = {g: lad_card(tr[tr.g == g]) for g in GROUP.values()}
MONO = {'s': -1, 's_rel': -1, 'd_bar': 1, 'd_last': 1, 'betac': 1, 'dev5': 1, 'governor': 0, 'chair': 0, 'hn': 0}
XG = XGBRegressor(objective='reg:absoluteerror', n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=10, monotone_constraints=tuple(MONO[c] for c in FX + ['hn']), random_state=0, n_jobs=2).fit(tr[FX + ['hn']], tr.dev)
print('LAD slopes (2012-2015):', {m: {g: round(v, 3) for g, v in s.items()} for m, s in SLOPE.items()})
print('M7 penalty and coefficients (standardized):', {g: (v[1], {k: round(c, 4) for k, c in v[2].items() if abs(c) > 1e-6}) for g, v in M7.items()})

def dev_hats(X):
    D = pd.DataFrame(index=X.index)
    D['M0'] = 0.0
    for m, col in [('M1', 's'), ('M2', 's_rel'), ('M3', 'd_bar'), ('M4', 'd_last')]:
        D[m] = X[col] * X.g.map(SLOPE[m])
    D['M5'] = X.dev5; D['M6'] = X.betac
    D['M7'] = 0.0
    for g, (f, _, _) in M7.items():
        idx = X.index[X.g == g]
        if len(idx): D.loc[idx, 'M7'] = f(X.loc[idx])
    D['M8'] = XG.predict(X[FX + ['hn']])
    return D
DH = dev_hats(C)
METHODS = list(DH.columns)

# ---------- static persona fitted on each test sample (reference only)
def static_ref(mask):
    T = C[mask]; med = T.groupby(['person_id', 'g']).dev.transform('median')
    return med

def clustered(d, g1, g2):
    """p value and standard error of the mean loss differential."""
    if np.allclose(d, 0): return np.nan, np.nan
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': np.column_stack([pd.factorize(g1)[0], pd.factorize(g2)[0]])}, use_t=True)
    return float(r.pvalues[0]), float(r.bse[0])
def holm(ps):
    ps = pd.Series(ps).dropna(); order = ps.sort_values().index; m = len(ps); adj, run = {}, 0.0
    for i, k in enumerate(order): run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return pd.Series(adj)
def ind_part(X, f):
    move = f - X.lag
    return move - move.groupby([X.meeting_date, X.proj_year]).transform('median')

def evaluate(X, c, DHx, label):
    rows = []
    for sname, sel in SAMPLES.items():
        mask = sel(X.meeting_date.dt)
        T = X[mask]; base_f = np.maximum(ZLB, T.lag + c[mask])
        e0 = (base_f - T.federal_funds_rate).abs()
        fc = {m: np.maximum(ZLB, T.lag + c[mask] + rd25(DHx.loc[mask, m])) for m in METHODS}
        fc_cont = {m: np.maximum(ZLB, T.lag + c[mask] + DHx.loc[mask, m]) for m in METHODS}
        fc['S'] = np.maximum(ZLB, T.lag + c[mask] + rd25(static_ref(mask)))
        tests = {m: clustered(((fc[m] - T.federal_funds_rate).abs() - e0).values, T.person_id.values, T.meeting_date.values) for m in METHODS[1:]}
        ps = {m: v[0] for m, v in tests.items()}; adj = holm(ps)
        true_ind = T.dev if label == 'actual move' else ind_part(T, T.federal_funds_rate)
        for m in METHODS + ['S']:
            e = (fc[m] - T.federal_funds_rate).abs()
            rows.append(dict(committee=label, sample=sname, method=m, n=len(T), MAE_bp=100 * e.mean(),
                             MAE_unrounded_bp=100 * (fc_cont[m] - T.federal_funds_rate).abs().mean() if m in fc_cont else np.nan,
                             vs_M0_bp=100 * (e.mean() - e0.mean()), se_bp=100 * tests.get(m, (np.nan, np.nan))[1], p=ps.get(m, np.nan), p_holm=adj.get(m, np.nan),
                             individual_MAE_bp=100 * (ind_part(T, fc[m]) - true_ind).abs().mean(),
                             share_changed_by_card=float((rd25(DHx.loc[mask, m]) != 0).mean()) if m in DHx else np.nan))
    return rows

rows = evaluate(C, C.c, DH, 'actual move')

# ---------- feasible: B30b's committee move instead of the actual one
if a.feasible:
    V = pd.read_csv(a.feasible, parse_dates=['meeting_date'], usecols=['meeting_date', 'person_id', 'proj_year', 'B30b_grid_median_signals'])
    X = C.merge(V, on=['meeting_date', 'person_id', 'proj_year'], how='left')
    X = X[X.B30b_grid_median_signals.notna()].copy()
    chat = (X.B30b_grid_median_signals - X.lag).groupby([X.meeting_date, X.proj_year]).transform('median')
    F2 = c_features(X, chat); X['betac'], X['dev5'] = F2.betac, F2.dev5
    rows += evaluate(X, chat, dev_hats(X), 'B30b move')
R = pd.DataFrame(rows); R.to_csv('data/persona_metrics.csv', index=False)

# ---------- facts about the deviation
facts = []
for sname, sel in {'2012-2015': lambda d: d.year <= 2015, **SAMPLES}.items():
    T = C[sel(C.meeting_date.dt)]
    fe = T.groupby(['person_id', 'g']).dev.transform('mean'); fe_p = T.groupby('person_id').dev.transform('mean')
    seq = T.sort_values('meeting_date')
    facts.append(dict(sample=sname, cells=len(T), mean_abs_dev_bp=100 * T.dev.abs().mean(), share_dev_zero=float((T.dev.abs() < 1e-9).mean()),
                      share_abs_dev_le_25bp=float((T.dev.abs() <= 0.25 + 1e-9).mean()),
                      r2_person=1 - ((T.dev - fe_p) ** 2).sum() / ((T.dev - T.dev.mean()) ** 2).sum(),
                      r2_person_horizon=1 - ((T.dev - fe) ** 2).sum() / ((T.dev - T.dev.mean()) ** 2).sum(),
                      corr_dev_stance=T.dev.corr(T.s), corr_dev_last=T.dev.corr(T.d_last.where(T.d_last != 0)),
                      share_abs_err_ceiling_from_floor_meeting=np.nan))
pd.DataFrame(facts).to_csv('data/persona_facts.csv', index=False)

# ---------- descriptive cards per participant (all 2012-2020 SEPs; for the design document, not for forecasting)
K = C.copy(); K['changed'] = K.rev.abs() > 1e-9
cards = K.groupby(['person_id', 'participant_name', 'g']).agg(seps=('meeting_date', 'nunique'), stance_bp=('s', lambda x: 100 * x.mean()),
        stance_sd_bp=('s', lambda x: 100 * x.std()), share_unchanged=('changed', lambda x: 1 - x.mean()),
        dev_bp=('dev', lambda x: 100 * x.mean()), abs_dev_bp=('dev', lambda x: 100 * x.abs().mean())).reset_index()
cards.to_csv('data/persona_cards.csv', index=False)
pd.concat([C[['meeting_date', 'person_id', 'participant_name', 'h', 'g', 'hn', 'proj_year', 'federal_funds_rate', 'lag', 'B1a_prev_median', 'rev', 'c', 'dev',
              's', 's_bar', 's_rel', 'd_bar', 'd_last', 'beta', 'betac', 'dev5', 'governor', 'chair']],
           DH.add_prefix('devhat_')], axis=1).to_csv('data/persona_predictions.csv', index=False)

pd.set_option('display.width', 220)
for (lab, sname), g in R.groupby(['committee', 'sample'], sort=False):
    print(f'\n== committee: {lab} | {sname}'); print(g[['method', 'n', 'MAE_bp', 'MAE_unrounded_bp', 'vs_M0_bp', 'se_bp', 'p', 'p_holm', 'individual_MAE_bp', 'share_changed_by_card']].round(3).to_string(index=False))
print('\nfacts'); print(pd.DataFrame(facts).round(3).to_string(index=False))
