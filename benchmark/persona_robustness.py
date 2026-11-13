"""Robustness of the persona analysis. Decided after persona_deviation.py and persona_information.py, before this run.
  R1 expanding window: re-estimate every January on all SEPs before that year (2012 onward) and predict that year's SEPs, so the
     2016-2019 behaviour can enter the estimates. Reported next to the fixed 2012-2015 estimates (which reproduce M1, M7, P1, B1).
       M1e  LAD slope on s per horizon group
       M1b  conditional median of the deviation per stance bin x horizon group (bins of persona_stance_bins.csv; cells with
            fewer than 20 training cells predict 0)
       M7e  the M7 data card
       P1e  three-class logit on s, |s| against the training frequencies P0e (log loss)
       B1e  OLS on s against 0 (squared error)
     MAE methods: forecast max(0.125, lag + c + rd25(dev_hat)), paired test against M0, Holm across M1e, M1b, M7e.
  R2 dissents as person-specific news: sign of the participant's dissents (tighter +1, easier -1) at FOMC meetings after his/her
     preceding SEP and before this SEP (primary), or from the preceding SEP onward (secondary). Parameter-free rule
     M9: deviation = 25 bp in the dissent's direction at every horizon. Reported on 2012-2015 and on the test samples.
     Added after the first run showed deviations against the dissent's direction: OLS of the deviation on the dissent sign with
     and without the stance s (horizon-group dummies, two-way clustered; one-way by meeting as a check).
  R3 is there person heterogeneity at all: permutation test, person labels shuffled among the participants of each SEP (all
     horizons of a participant move together), 2,000 draws, per period. Statistics: between-participant dispersion of the share
     deviating (chi-square form) and of the mean deviation net of the stance effect (OLS on s per horizon group in the period).
  R4 inference: wild cluster bootstrap p values (Rademacher weights, null imposed, 9,999 draws), clustering by meeting and by
     participant, for every comparison in R1 and R2.
Usage (in docs/benchmark/work/frozen_2015, after persona_deviation.py): python ../../code/persona_robustness.py
Reads data/persona_predictions.csv, data/dataset.csv, data/policy_votes.csv.
Writes data/persona_robustness.csv, data/persona_heterogeneity.csv, data/persona_expanding_coefs.csv."""
import warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor, LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

ZLB = 0.125
SAMPLES = {'2016-2020': lambda d: (d.year >= 2016) & (d.year <= 2020),
           'excl. June 2020': lambda d: (d.year >= 2016) & (d.year <= 2020) & ~((d.year == 2020) & (d.month == 6)),
           '2016-2019': lambda d: (d.year >= 2016) & (d.year <= 2019)}
GROUPS = ['0', '1', '23', 'LR']
CARD = ['s', 's_rel', 'd_bar', 'd_last', 'betac', 'dev5', 'governor', 'chair']
rd25 = lambda x: np.round(np.asarray(x, float) / 0.25) * 0.25
rng = np.random.default_rng(0)

D = pd.read_csv('data/persona_predictions.csv', parse_dates=['meeting_date'], dtype={'h': str, 'g': str})
D = D.sort_values(['meeting_date', 'person_id', 'proj_year']).reset_index(drop=True)
D['y'] = np.select([D.dev <= -0.1, D.dev >= 0.1], [0, 2], 1)          # 0 down, 1 zero, 2 up
D['abs_s'] = D.s.abs()
for g in ['1', '23', 'LR']: D[f'g{g}'] = (D.g == g).astype(float)
D['bin'] = np.select([D.s <= -0.375, D.s <= -0.125, D.s < 0.125, D.s < 0.375], [0, 1, 2, 3], 4)

def clustered(d, g1, g2):
    if np.allclose(d, 0): return np.nan, np.nan
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': np.column_stack([pd.factorize(g1)[0], pd.factorize(g2)[0]])}, use_t=True)
    return float(r.bse[0]), float(r.pvalues[0])
def holm(ps):
    ps = pd.Series(ps).dropna(); order = ps.sort_values().index; m = len(ps); adj, run = {}, 0.0
    for i, k in enumerate(order): run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return pd.Series(adj)
def wild_p(d, groups, B=9999):
    """Wild cluster bootstrap of H0: E[d] = 0 (restricted residuals are d itself), one-way clustered t statistic."""
    d = np.asarray(d, float)
    if np.allclose(d, 0): return np.nan
    gi = pd.factorize(groups)[0]; G = gi.max() + 1; n = len(d)
    def t(x):
        s = np.bincount(gi, weights=x - x.mean(), minlength=G)
        return x.mean() / (np.sqrt(G / (G - 1) * (s ** 2).sum()) / n)
    W = rng.choice([-1.0, 1.0], size=(B, G))
    tb = np.array([t(d * w[gi]) for w in W])
    return float((np.abs(tb) >= abs(t(d))).mean())

# ---------- R1: fixed and expanding estimation windows
def lad_slope(x, y):
    return float(QuantileRegressor(quantile=0.5, alpha=0.0, fit_intercept=False, solver='highs').fit(x.values.reshape(-1, 1), y.values).coef_[0])
def lad_card(T):
    cols = [c for c in CARD if T[c].std() > 1e-9]; sc = StandardScaler().fit(T[cols]); seps = sorted(T.meeting_date.unique())
    cut = seps[int(len(seps) * 0.75)]; A, B = T[T.meeting_date < cut], T[T.meeting_date >= cut]
    q = lambda al: QuantileRegressor(quantile=0.5, alpha=al, solver='highs')
    best = min([0.0, 0.002, 0.01, 0.05], key=lambda al: np.abs(B.dev - q(al).fit(sc.transform(A[cols]), A.dev).predict(sc.transform(B[cols]))).mean())
    m = q(best).fit(sc.transform(T[cols]), T.dev)
    return lambda X: m.predict(sc.transform(X[cols]))
def fit_logit(T, cols):
    sc = StandardScaler().fit(T[cols]); seps = sorted(T.meeting_date.unique()); cut = seps[int(len(seps) * 0.75)]
    A, B = T[T.meeting_date < cut], T[T.meeting_date >= cut]
    mk = lambda C_: LogisticRegression(C=C_, max_iter=5000)
    best = min([0.01, 0.1, 1.0, 10.0], key=lambda C_: log_loss(B.y, mk(C_).fit(sc.transform(A[cols]), A.y).predict_proba(sc.transform(B[cols])), labels=[0, 1, 2]))
    m = mk(best).fit(sc.transform(T[cols]), T.y)
    return lambda X: m.predict_proba(sc.transform(X[cols]))

def predict_window(tr, X, label):
    """Fit on tr, predict X. Returns deviation predictions, probabilities and the fitted coefficients."""
    out = pd.DataFrame(0.0, index=X.index, columns=['M1e', 'M1b', 'M7e', 'B1e']); P0 = np.zeros((len(X), 3)); coefs = []
    for g in GROUPS:
        T = tr[tr.g == g]; idx = X.index[X.g == g]
        if not len(idx): continue
        b = lad_slope(T.s, T.dev); out.loc[idx, 'M1e'] = b * X.loc[idx, 's']
        med = T.groupby('bin').dev.agg(['median', 'size']); m = med['median'].where(med['size'] >= 20, 0.0)
        out.loc[idx, 'M1b'] = X.loc[idx, 'bin'].map(m).fillna(0.0).values
        out.loc[idx, 'M7e'] = lad_card(T)(X.loc[idx])
        lr = LinearRegression().fit(T[['s']], T.dev); out.loc[idx, 'B1e'] = lr.predict(X.loc[idx, ['s']])
        coefs.append(dict(window=label, g=g, train_cells=len(T), M1e_slope=b, B1e_slope=float(lr.coef_[0]),
                          **{f'M1b_bin{k}_bp': 100 * m.get(k, np.nan) for k in range(5)}))
    freq = tr.groupby('g').y.value_counts(normalize=True).unstack().reindex(columns=[0, 1, 2]).fillna(0)
    P0 = np.vstack([freq.loc[g].values for g in X.g])
    P1 = fit_logit(tr, ['s', 'abs_s', 'g1', 'g23', 'gLR'])(X)
    return out, P0, P1, coefs

test = D[D.meeting_date.dt.year >= 2016]
WIN = {}
fx = predict_window(D[D.meeting_date.dt.year <= 2015], test, 'fixed 2012-2015'); WIN['fixed 2012-2015'] = fx
parts, coefs = [], list(fx[3])
for Y in range(2016, 2021):
    X = D[D.meeting_date.dt.year == Y]; o, p0, p1, cf = predict_window(D[D.meeting_date.dt.year < Y], X, f'expanding, through {Y - 1}')
    parts.append((o, pd.DataFrame(p0, index=X.index), pd.DataFrame(p1, index=X.index))); coefs += cf
WIN['expanding'] = (pd.concat([p[0] for p in parts]).reindex(test.index), pd.concat([p[1] for p in parts]).reindex(test.index).values,
                    pd.concat([p[2] for p in parts]).reindex(test.index).values, None)
pd.DataFrame(coefs).to_csv('data/persona_expanding_coefs.csv', index=False)

rows = []
for wname, (DH, P0, P1, _) in WIN.items():
    DH = DH.reindex(test.index); P0 = pd.DataFrame(P0, index=test.index); P1 = pd.DataFrame(P1, index=test.index)
    for sname, sel in SAMPLES.items():
        mask = sel(test.meeting_date.dt).values; T = test[mask]; gp, gm = T.person_id.values, T.meeting_date.values
        e0 = (np.maximum(ZLB, T.lag + T.c) - T.federal_funds_rate).abs()
        diffs, info = {}, {}
        for m in ['M1e', 'M1b', 'M7e']:
            dh = rd25(DH.loc[mask, m]); e = (np.maximum(ZLB, T.lag + T.c + dh) - T.federal_funds_rate).abs()
            diffs[m] = (e - e0).values; info[m] = ('MAE (bp)', 100 * e0.mean(), 100 * e.mean(), float((dh != 0).mean()))
        ll = lambda P: -np.log(np.clip(P.loc[mask].values[np.arange(mask.sum()), T.y.values], 1e-12, 1))
        diffs['P1e'] = ll(P1) - ll(P0); info['P1e'] = ('log loss', ll(P0).mean(), ll(P1).mean(), np.nan)
        se_b = (T.dev - DH.loc[mask, 'B1e']) ** 2
        diffs['B1e'] = (se_b - T.dev ** 2).values; info['B1e'] = ('RMSE (bp)', 100 * np.sqrt((T.dev ** 2).mean()), 100 * np.sqrt(se_b.mean()), np.nan)
        tests = {m: clustered(v, gp, gm) for m, v in diffs.items()}
        adj = holm({m: tests[m][1] for m in ['M1e', 'M1b', 'M7e']})
        for m, v in diffs.items():
            metric, base, model, changed = info[m]
            rows.append(dict(check='R1', window=wname, sample=sname, method=m, metric=metric, n=int(mask.sum()), base=base, model=model,
                             diff=model - base, p=tests[m][1], p_holm=adj.get(m, np.nan), p_wild_meeting=wild_p(v, gm), p_wild_person=wild_p(v, gp),
                             share_changed=changed))

# ---------- R2: dissents between SEPs
ds = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])[['meeting_date', 'person_id']].drop_duplicates().sort_values('meeting_date')
prev = {}
for pid, g in ds.groupby('person_id'):
    d = list(g.meeting_date); prev.update({(pid, x): y for x, y in zip(d[1:], d[:-1])})
V = pd.read_csv('data/policy_votes.csv', parse_dates=['meeting_date'])
V = V[(V.vote == 'against') & V.direction.notna()].assign(sgn=lambda v: v.direction.map({'tighter': 1, 'easier': -1}))
def signal(pid, t, include_prev):
    tp = prev.get((pid, t))
    if tp is None: return 0
    w = V[(V.person_id == pid) & (V.meeting_date < t) & ((V.meeting_date >= tp) if include_prev else (V.meeting_date > tp))]
    return int(np.sign(w.sgn.sum()))
keys = D[['person_id', 'meeting_date']].drop_duplicates()
for col, inc in [('dis_between', False), ('dis_from_prev', True)]:
    keys[col] = [signal(p, t, inc) for p, t in zip(keys.person_id, keys.meeting_date)]
D = D.merge(keys, on=['person_id', 'meeting_date'], how='left')
for col, wlabel in [('dis_between', 'dissent after the preceding SEP'), ('dis_from_prev', 'dissent at or after the preceding SEP')]:
    for sname, sel in {'2012-2015': lambda d: d.year <= 2015, **SAMPLES}.items():
        T = D[sel(D.meeting_date.dt).values]; S = T[T[col] != 0]
        e0 = (np.maximum(ZLB, T.lag + T.c) - T.federal_funds_rate).abs()
        e9 = (np.maximum(ZLB, T.lag + T.c + 0.25 * T[col]) - T.federal_funds_rate).abs(); dd = (e9 - e0).values
        se, p = clustered(dd, T.person_id.values, T.meeting_date.values)
        rows.append(dict(check='R2', window=wlabel, sample=sname, method='M9', metric='MAE (bp)', n=len(T), base=100 * e0.mean(), model=100 * e9.mean(),
                         diff=100 * (e9.mean() - e0.mean()), p=p, p_wild_meeting=wild_p(dd, T.meeting_date.values), p_wild_person=wild_p(dd, T.person_id.values),
                         share_changed=float((T[col] != 0).mean()), signal_cells=len(S), signal_person_seps=len(S[['person_id', 'meeting_date']].drop_duplicates()),
                         share_in_dissent_direction=float(((S.y != 1) & (np.sign(S.dev) == S[col])).mean()) if len(S) else np.nan,
                         share_no_deviation=float((S.y == 1).mean()) if len(S) else np.nan,
                         share_opposite=float(((S.y != 1) & (np.sign(S.dev) == -S[col])).mean()) if len(S) else np.nan,
                         mean_dev_in_dissent_direction_bp=float(100 * (S.dev * S[col]).mean()) if len(S) else np.nan))
    # is the dissent more than a marker of a stance far from the median? OLS of the deviation on the dissent sign, with and without s
    for pname, sel in {'2012-2015': lambda d: d.year <= 2015, '2016-2020': SAMPLES['2016-2020'], '2012-2020': lambda d: d.year <= 2020}.items():
        T = D[sel(D.meeting_date.dt).values]; S = T[T[col] != 0]
        for ctrl in ['', ' + s']:
            Xo = pd.concat([T[[col]], T[['s']] if ctrl else None, pd.get_dummies(T.g, prefix='g', drop_first=True).astype(float)], axis=1)
            Xo = sm.add_constant(Xo); fit = lambda grp: sm.OLS(T.dev, Xo).fit(cov_type='cluster', cov_kwds={'groups': grp})
            two, one = fit(np.column_stack([pd.factorize(T.person_id)[0], pd.factorize(T.meeting_date)[0]])), fit(pd.factorize(T.meeting_date)[0])
            rows.append(dict(check='R2 regression', window=wlabel, sample=pname, method='OLS dev ~ dissent' + ctrl, metric='dissent coefficient (bp)',
                             n=len(T), model=100 * two.params[col], p=two.pvalues[col], p_meeting=one.pvalues[col], signal_cells=len(S),
                             mean_stance_in_dissent_direction_bp=float(100 * (S.s * S[col]).mean()),
                             share_stance_in_dissent_direction=float((np.sign(S.s) == S[col]).mean())))
R = pd.DataFrame(rows); R.to_csv('data/persona_robustness.csv', index=False)

# ---------- R3: person heterogeneity, permutation within SEP
def het(T, R_=2000):
    T = T.copy(); T['nz'] = (T.y != 1).astype(float); T['r'] = 0.0
    for g, G in T.groupby('g'):
        T.loc[G.index, 'r'] = G.dev - LinearRegression().fit(G[['s']], G.dev).predict(G[['s']])
    pc = pd.factorize(T.person_id)[0]; P = pc.max() + 1; nz, r = T.nz.values, T.r.values
    Q, M = nz.mean(), r.mean()
    def stats(lab):
        n = np.bincount(lab, minlength=P); ok = n > 0
        q = np.bincount(lab, weights=nz, minlength=P)[ok] / n[ok]; m = np.bincount(lab, weights=r, minlength=P)[ok] / n[ok]
        return (n[ok] * (q - Q) ** 2).sum() / (Q * (1 - Q)), (n[ok] * (m - M) ** 2).sum()
    groups = [(idx, pc[idx]) for idx in T.groupby('meeting_date').indices.values()]
    obs = stats(pc); sims = []
    for _ in range(R_):
        lab = np.empty_like(pc)
        for idx, codes in groups:
            u = np.unique(codes); lab[idx] = rng.permutation(u)[np.searchsorted(u, codes)]
        sims.append(stats(lab))
    sims = np.array(sims)
    return [dict(statistic=k, observed=obs[i], permuted_mean=sims[:, i].mean(), ratio=obs[i] / sims[:, i].mean(),
                 p=(1 + (sims[:, i] >= obs[i]).sum()) / (1 + R_), participants=P, cells=len(T))
            for i, k in enumerate(['share deviating', 'mean deviation net of stance'])]
H = []
for pname, sel in {'2012-2015': lambda d: d.year <= 2015, '2016-2020': SAMPLES['2016-2020'], '2016-2019': SAMPLES['2016-2019']}.items():
    H += [dict(period=pname, **h) for h in het(D[sel(D.meeting_date.dt).values])]
H = pd.DataFrame(H); H.to_csv('data/persona_heterogeneity.csv', index=False)

pd.set_option('display.width', 250)
print(pd.DataFrame(coefs).round(3).to_string(index=False))
print(R[R.check == 'R1'].drop(columns=[c for c in R.columns if c.startswith('signal') or c.startswith('share_in') or c.startswith('share_no') or c.startswith('share_opp') or c.startswith('mean_dev')]).round(4).to_string(index=False))
print(R[R.check == 'R2'].dropna(axis=1, how='all').round(3).to_string(index=False))
print(R[R.check == 'R2 regression'].dropna(axis=1, how='all').round(3).to_string(index=False))
print(H.round(4).to_string(index=False))
