"""All dot-plot baselines on the group's unified protocol (11.5 meeting): estimate on SEPs up to 2015, test on 2016-2020 and
2016-2019 (without the COVID year). Same cells and metrics as the group's SEP forecast comparison (26.10.27):
  - cells: dots whose participant has a dot for the same calendar target year (or LR) at his/her own preceding SEP;
  - MAE and RMSE in basis points, share of forecasts within 25 bp (inclusive), overall and by horizon (Y0-Y3, LR);
  - paired test of absolute errors against the naive forecast, standard errors clustered two ways (participant and meeting),
    t reference distribution, Holm correction across the methods of each sample.
Also re-implements the group's pooled AR(1) and ARX (coefficients from 2012-2015 only, Y0-Y3 share the slopes with
horizon intercepts, LR estimated separately, forecasts rounded to two decimals) as a cross-check.
Usage: python evaluate_unified.py <predictions_v6.csv from a FEDAI_TRAIN_END=2015-12-31 run> [--expanding <predictions_v6.csv>]
         [--add your.csv NAME ...]   any other method: CSV with meeting_date, participant (person_id or name), horizon (0-3 or LR), prediction
Writes data/unified_metrics.csv (and data/unified_protocols.csv with --expanding)."""
import argparse, numpy as np, pandas as pd
import statsmodels.api as sm

ap = argparse.ArgumentParser(); ap.add_argument('pred'); ap.add_argument('--expanding')
ap.add_argument('--add', nargs=2, action='append', default=[], metavar=('CSV', 'NAME')); a = ap.parse_args()
TRAIN_END, START = pd.Timestamp('2015-12-31'), pd.Timestamp('2016-01-01')
SAMPLES = {'2016-2020': ('2016-01-01', '2020-12-31'), '2016-2019': ('2016-01-01', '2019-12-31')}
H = ['0', '1', '2', '3', 'LR']

def load(path):
    P = pd.read_csv(path, parse_dates=['meeting_date']); P['h'] = P.h.astype(str)
    return P[P.federal_funds_rate.notna()].reset_index(drop=True)
P = load(a.pred)

# ---- own preceding SEP and the dot it had for the same calendar target year (LR by LR)
own = P[['person_id', 'meeting_date']].drop_duplicates().sort_values('meeting_date')
prev_own = {}
for pid, g in own.groupby('person_id'):
    d = list(g.meeting_date)
    prev_own.update({(pid, x): y for x, y in zip(d[1:], d[:-1])})
dots = P.set_index(['person_id', 'meeting_date', 'proj_year']).federal_funds_rate
P['t_lag'] = [prev_own.get((p, t)) for p, t in zip(P.person_id, P.meeting_date)]
P['lag'] = [dots.get((p, tl, y), np.nan) if tl is not None else np.nan for p, tl, y in zip(P.person_id, P.t_lag, P.proj_year)]

# ---- the group's pooled AR(1) and ARX (re-implemented from the group's report)
mac = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')[['inflation', 'unrate']]
P['d_pi'] = [mac.inflation.get(t, np.nan) - mac.inflation.get(tl, np.nan) if tl is not None else np.nan for t, tl in zip(P.meeting_date, P.t_lag)]
P['d_u'] = [mac.unrate.get(t, np.nan) - mac.unrate.get(tl, np.nan) if tl is not None else np.nan for t, tl in zip(P.meeting_date, P.t_lag)]
half_up = lambda x: np.floor(np.asarray(x, float) * 100 + 0.5 + 1e-9) / 100
dev = P[(P.meeting_date <= TRAIN_END) & P.lag.notna()]
for name, extra in [('G1_group_pooled_AR1', []), ('G2_group_pooled_ARX', ['d_pi', 'd_u'])]:
    P[name] = np.nan
    for lr in (False, True):
        tr = dev[(dev.h == 'LR') == lr]; te = P[((P.h == 'LR') == lr) & P.lag.notna()]
        def X(df):
            cols = {'const': 1.0, 'lag': df.lag}
            if not lr:
                for k in ['1', '2', '3']: cols[f'h{k}'] = (df.h == k).astype(float)
            for e in extra: cols[e] = df[e]
            return pd.DataFrame(cols, index=df.index)
        fit = sm.OLS(tr.federal_funds_rate, X(tr)).fit()
        P.loc[te.index, name] = half_up(fit.predict(X(te)))
    print(f'{name}: estimated on {int((dev.h != "LR").sum())} calendar-target and {int((dev.h == "LR").sum())} LR dots (2012-2015)')

# ---- other methods (e.g. time-series models, the agent): same cells, same tests
for path, name in a.add:
    X = pd.read_csv(path); X['meeting_date'] = pd.to_datetime(X.meeting_date)
    X['h'] = X.horizon.astype(str).str.upper().replace({'LONGER RUN': 'LR', 'LONG_RUN': 'LR'})
    by_name = P.drop_duplicates('participant_name').set_index('participant_name').person_id
    X['person_id'] = [p if str(p) in set(P.person_id) else by_name.get(p) for p in X.participant]
    col = f'X_{name}'
    P = P.merge(X[['meeting_date', 'person_id', 'h', 'prediction']].rename(columns={'prediction': col}), on=['meeting_date', 'person_id', 'h'], how='left')
    print(f'{col}: {int(P[col].notna().sum())} of {len(X)} rows matched')
METHODS = [c for c in P.columns if c.startswith(('B', 'G', 'X_'))]
FAIR = [m for m in METHODS if not m.endswith('*')]

def clustered_p(d, g1, g2):
    if np.allclose(d, 0): return np.nan
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': np.column_stack([pd.factorize(g1)[0], pd.factorize(g2)[0]])}, use_t=True)
    return float(r.pvalues[0])

def holm(ps):
    ps = pd.Series(ps).dropna(); order = ps.sort_values().index; m = len(ps); adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return pd.Series(adj)

def evaluate(P, protocol):
    rows = []
    for sname, (lo, hi) in SAMPLES.items():
        T = P[(P.meeting_date >= lo) & (P.meeting_date <= hi) & P.lag.notna()]            # common cells
        e0 = (T.B0_prior_own - T.federal_funds_rate).abs()
        ps = {}
        for m in METHODS:
            ok = T[m].notna()
            e = (T[m] - T.federal_funds_rate).abs()
            if m != 'B0_prior_own' and m in FAIR:
                ps[m] = clustered_p((e - e0)[ok].values, T.person_id[ok].values, T.meeting_date[ok].values)
            for h in ['all'] + H:
                s = ok & ((T.h == h) if h != 'all' else True)
                if not s.any(): continue
                rows.append(dict(protocol=protocol, sample=sname, method=m, horizon=h, n=int(s.sum()), MAE_bp=e[s].mean() * 100,
                                 RMSE_bp=np.sqrt(((T[m] - T.federal_funds_rate)[s] ** 2).mean()) * 100,
                                 within_25bp=(e[s] <= 0.25 + 1e-9).mean() * 100))
        adj = holm(ps)
        for r in rows:
            if r['sample'] == sname and r['protocol'] == protocol and r['horizon'] == 'all':
                r['p_vs_naive'] = ps.get(r['method'], np.nan); r['p_holm'] = adj.get(r['method'], np.nan)
    return pd.DataFrame(rows)

R = evaluate(P, 'estimated on 2012-2015 only')
R.to_csv('data/unified_metrics.csv', index=False)
pd.set_option('display.width', 250)
for sname in SAMPLES:
    W = R[(R['sample'] == sname)].pivot_table(index='method', columns='horizon', values='MAE_bp')[['all'] + H]
    o = R[(R['sample'] == sname) & (R.horizon == 'all')].set_index('method')
    W.insert(0, 'n', o.n); W['RMSE'] = o.RMSE_bp; W['<=25bp %'] = o.within_25bp; W['p_holm'] = o.p_holm
    print(f'\n== {sname} (common cells), MAE in bp'); print(W.sort_values('all').round(2).to_string())

if a.expanding:                                        # same cells, parameters re-estimated at every SEP (the default protocol)
    E = load(a.expanding).merge(P[['meeting_date', 'person_id', 'h', 'lag'] + [c for c in METHODS if c.startswith(('G', 'X_'))]],
                                on=['meeting_date', 'person_id', 'h'], how='left')
    RE = evaluate(E, 'expanding window')
    C = pd.concat([R, RE]); C.to_csv('data/unified_protocols.csv', index=False)
    Z = C[C.horizon == 'all'].pivot_table(index='method', columns=['sample', 'protocol'], values='MAE_bp')
    print('\n== MAE (bp), parameters fixed after 2015 vs re-estimated at every SEP'); print(Z.round(2).to_string())
