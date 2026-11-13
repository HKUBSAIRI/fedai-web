"""Follow-up to persona_deviation.py, decided after its MAE results (exploratory, not pre-specified).

Under MAE the data cards do not move the two-step ceiling, because 6 in 10 deviations are exactly zero and the conditional
median is zero almost everywhere. This script asks whether the card carries information about the deviation that MAE does
not reward, and whether a participant's tendency to deviate is a stable trait.
  A  probabilities of down / zero / up (deviation <= -10 bp, in between, >= +10 bp), multinomial logit fitted on 2012-2015:
       P0 training frequencies per horizon group; P1 stance (s, |s|); P2 the full card of M7 plus c and s*c;
       P3 P2 plus the participant's own past frequency of deviating (shrunk to the training frequency, 4 pseudo-cells).
     Scores: log loss and Brier, paired test against P0 (two-way clustered, t reference, Holm across P1-P3).
  B  means: OLS of the deviation on s (B1) or on the M7 card (B2) per horizon group, fitted on 2012-2015, unrounded;
     RMSE against 0 (the ceiling), out-of-sample R2 = 1 - SSE / SSE of 0, paired test on squared errors.
  C  where the ceiling's errors are, and whether "how often a participant deviates" carries over from 2012-2015 to 2016-2020.
Usage (in docs/benchmark/work/frozen_2015, after persona_deviation.py): python ../../code/persona_information.py
Reads data/persona_predictions.csv; writes data/persona_information.csv, data/persona_concentration.csv, data/persona_traits.csv."""
import warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

TRAIN_END = pd.Timestamp('2015-12-31')
SAMPLES = {'2016-2020': lambda d: (d.year >= 2016) & (d.year <= 2020),
           'excl. June 2020': lambda d: (d.year >= 2016) & (d.year <= 2020) & ~((d.year == 2020) & (d.month == 6)),
           '2016-2019': lambda d: (d.year >= 2016) & (d.year <= 2019)}
D = pd.read_csv('data/persona_predictions.csv', parse_dates=['meeting_date'], dtype={'h': str, 'g': str})
D = D.sort_values(['meeting_date', 'person_id', 'proj_year']).reset_index(drop=True)
D['y'] = np.select([D.dev <= -0.1, D.dev >= 0.1], [0, 2], 1)          # 0 down, 1 zero, 2 up
D['abs_s'] = D.s.abs(); D['sc'] = D.s * D.c
for g in ['1', '23', 'LR']: D[f'g{g}'] = (D.g == g).astype(float)
tr_mask = D.meeting_date <= TRAIN_END
base_rate = float((D.y[tr_mask] != 1).mean())

# participant's own past frequency of deviating (same horizon group, SEPs before t), shrunk to the training frequency
D['p_dev'] = 0.0
for _, G in D.groupby(['person_id', 'g']):
    per = G.assign(nz=(G.y != 1).astype(float)).groupby('meeting_date').nz.agg(['sum', 'size'])
    k, n = per['sum'].cumsum().shift(1).fillna(0), per['size'].cumsum().shift(1).fillna(0)
    D.loc[G.index, 'p_dev'] = ((k + 4 * base_rate) / (n + 4)).reindex(G.meeting_date).values
GD = ['g1', 'g23', 'gLR']
CARD = ['s', 's_rel', 'd_bar', 'd_last', 'betac', 'dev5', 'governor', 'chair']
FEAT = {'P1': ['s', 'abs_s'] + GD, 'P2': CARD + ['abs_s', 'c', 'sc'] + GD, 'P3': CARD + ['abs_s', 'c', 'sc', 'p_dev'] + GD}
tr = D[tr_mask]

def clustered(d, g1, g2):
    if np.allclose(d, 0): return np.nan, np.nan
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': np.column_stack([pd.factorize(g1)[0], pd.factorize(g2)[0]])}, use_t=True)
    return float(r.params[0]), float(r.pvalues[0])
def holm(ps):
    ps = pd.Series(ps).dropna(); order = ps.sort_values().index; m = len(ps); adj, run = {}, 0.0
    for i, k in enumerate(order): run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return pd.Series(adj)

# ---------- A: probabilities
def fit_logit(cols):
    sc = StandardScaler().fit(tr[cols]); seps = sorted(tr.meeting_date.unique()); cut = seps[int(len(seps) * 0.75)]
    A, B = tr[tr.meeting_date < cut], tr[tr.meeting_date >= cut]
    mk = lambda C_: LogisticRegression(C=C_, max_iter=5000)
    best = min([0.01, 0.1, 1.0, 10.0], key=lambda C_: log_loss(B.y, mk(C_).fit(sc.transform(A[cols]), A.y).predict_proba(sc.transform(B[cols])), labels=[0, 1, 2]))
    m = mk(best).fit(sc.transform(tr[cols]), tr.y)
    return (lambda X: m.predict_proba(sc.transform(X[cols]))), best, dict(zip(cols, m.coef_[2] - m.coef_[0]))
freq = tr.groupby('g').y.value_counts(normalize=True).unstack().reindex(columns=[0, 1, 2]).fillna(0)
PROB = {'P0': np.vstack([freq.loc[g].values for g in D.g])}
INFO = {}
for m, cols in FEAT.items():
    f, best, coef = fit_logit(cols); PROB[m] = f(D); INFO[m] = (best, coef)
    print(m, 'C =', best, '| up-minus-down coefficients (standardized):', {k: round(v, 2) for k, v in coef.items()})
Y1h = np.eye(3)[D.y.values]
LL = {m: -np.log(np.clip(p[np.arange(len(D)), D.y.values], 1e-12, 1)) for m, p in PROB.items()}
BR = {m: ((p - Y1h) ** 2).sum(1) for m, p in PROB.items()}

rows = []
for sname, sel in SAMPLES.items():
    mask = sel(D.meeting_date.dt).values; T = D[mask]
    for score, S in [('log loss', LL), ('Brier', BR)]:
        tests = {m: clustered(S[m][mask] - S['P0'][mask], T.person_id.values, T.meeting_date.values) for m in ['P1', 'P2', 'P3']}
        adj = holm({m: v[1] for m, v in tests.items()})
        for m in ['P0', 'P1', 'P2', 'P3']:
            p = PROB[m][mask]
            rows.append(dict(part='A', sample=sname, model=m, score=score, n=int(mask.sum()), value=float(S[m][mask].mean()),
                             vs_P0=tests.get(m, (0.0, np.nan))[0], p=tests.get(m, (np.nan, np.nan))[1], p_holm=adj.get(m, np.nan),
                             share_nonzero_class_above_half=float(((p[:, 0] > 0.5) | (p[:, 2] > 0.5)).mean()),
                             actual_share_deviating=float((T.y != 1).mean())))

# ---------- B: means (OLS per horizon group, fitted on 2012-2015)
MEANS = {'B1': ['s'], 'B2': CARD}
for m, cols in MEANS.items():
    D[m] = 0.0
    for g, G in tr.groupby('g'):
        use = [c for c in cols if G[c].std() > 1e-9]
        lr = LinearRegression().fit(G[use], G.dev); idx = D.index[D.g == g]; D.loc[idx, m] = lr.predict(D.loc[idx, use])
for sname, sel in SAMPLES.items():
    mask = sel(D.meeting_date.dt).values; T = D[mask]; se0 = T.dev ** 2
    tests = {m: clustered(((T.dev - T[m]) ** 2 - se0).values, T.person_id.values, T.meeting_date.values) for m in MEANS}
    adj = holm({m: v[1] for m, v in tests.items()})
    for m in ['B0'] + list(MEANS):
        se = se0 if m == 'B0' else (T.dev - T[m]) ** 2
        rows.append(dict(part='B', sample=sname, model=m, score='RMSE (bp)', n=int(mask.sum()), value=100 * float(np.sqrt(se.mean())),
                         vs_P0=np.nan if m == 'B0' else 1e4 * tests[m][0], p=np.nan if m == 'B0' else tests[m][1], p_holm=adj.get(m, np.nan),
                         oos_R2=0.0 if m == 'B0' else float(1 - se.sum() / se0.sum())))
R = pd.DataFrame(rows); R.to_csv('data/persona_information.csv', index=False)

# ---------- C: where the ceiling's errors are; stability of the tendency to deviate
te = D[SAMPLES['2016-2020'](D.meeting_date.dt)].copy(); te['ae'] = te.dev.abs(); tot = te.ae.sum()
by_meet = te.groupby('meeting_date').agg(cells=('ae', 'size'), share_of_error=('ae', lambda x: x.sum() / tot), c_Y0_bp=('c', lambda x: np.nan),
                                         share_deviating=('y', lambda x: (x != 1).mean())).sort_values('share_of_error', ascending=False)
by_meet['c_Y0_bp'] = te[te.g == '0'].groupby('meeting_date').c.first().mul(100).reindex(by_meet.index)
by_pers = te.groupby('participant_name').agg(cells=('ae', 'size'), share_of_error=('ae', lambda x: x.sum() / tot),
                                             mean_abs_dev_bp=('ae', lambda x: 100 * x.mean())).sort_values('share_of_error', ascending=False)
pd.concat([by_meet.reset_index().assign(kind='meeting').rename(columns={'meeting_date': 'key'}),
           by_pers.reset_index().assign(kind='participant').rename(columns={'participant_name': 'key'})]).to_csv('data/persona_concentration.csv', index=False)

D['period'] = np.where(tr_mask, '2012-2015', '2016-2020')
D['side'] = np.select([D.s <= -0.375, D.s <= -0.125, D.s < 0.125, D.s < 0.375], ['below >= 37.5', 'below 12.5-37.5', 'within 12.5', 'above 12.5-37.5'], 'above >= 37.5')
D['move'] = np.select([D.y == 1, (D.y == 2) == (D.s < 0)], ['none', 'toward median'], 'away from median')
D.loc[D.side == 'within 12.5', 'move'] = np.where(D.y[D.side == 'within 12.5'] == 2, 'up', np.where(D.y[D.side == 'within 12.5'] == 0, 'down', 'none'))
bins = D.groupby(['period', 'side']).move.value_counts(normalize=True).unstack().fillna(0)
bins['cells'] = D.groupby(['period', 'side']).size(); bins.reset_index().to_csv('data/persona_stance_bins.csv', index=False)

D['nz'] = (D.y != 1).astype(float); D['period'] = np.where(tr_mask, 'train', 'test')
D['toward'] = np.where(D.s.abs() >= 0.125, ((np.sign(D.dev) == -np.sign(D.s)) & (D.y != 1)).astype(float), np.nan)
tr_ = D.groupby(['participant_name', 'period']).agg(cells=('nz', 'size'), share_deviating=('nz', 'mean'), mean_abs_dev_bp=('dev', lambda x: 100 * x.abs().mean()),
                                                    share_toward_median=('toward', 'mean'), stance_bp=('s', lambda x: 100 * x.mean())).unstack('period')
tr_.columns = [f'{a}_{b}' for a, b in tr_.columns]; tr_.reset_index().to_csv('data/persona_traits.csv', index=False)
both = tr_[(tr_.cells_train >= 20) & (tr_.cells_test >= 20)]

pd.set_option('display.width', 220)
print(f'\nA/B (training share deviating {base_rate:.3f})'); print(R.round(4).to_string(index=False))
print('\nC: moves by distance to the previous median'); print(bins.round(3).to_string())
print('\nC: meetings with the largest share of the ceiling error, 2016-2020'); print(by_meet.head(8).round(3).to_string())
print('\nC: participants with the largest share'); print(by_pers.head(8).round(3).to_string())
print(f'\nC: participants with >= 20 cells in both periods: {len(both)}')
for col in ['share_deviating', 'mean_abs_dev_bp', 'share_toward_median', 'stance_bp']:
    x, y = both[f'{col}_train'], both[f'{col}_test']; ok = x.notna() & y.notna()
    print(f'  {col}: corr(2012-2015, 2016-2020) = {np.corrcoef(x[ok], y[ok])[0, 1]:.2f} (Spearman {x[ok].rank().corr(y[ok].rank()):.2f}, n={ok.sum()})')
