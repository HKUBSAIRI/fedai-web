"""Evaluate a persona signal on the individual deviation from the committee's move (definitions in persona_deviation.py).
Usage (in the work directory, after persona_deviation.py): python evaluate_persona.py your.csv NAME
       python evaluate_persona.py --selftest       writes data/selftest_persona_P1.csv; evaluating it must reproduce P1
your.csv: meeting_date, participant (person_id or name), horizon (0-3 or LR), and one or both of
  p_down, p_same, p_up   probabilities that the dot ends below / at / above (own previous dot + committee move), +-10 bp as
                         the boundary; each row sums to 1
  deviation              predicted deviation from (own previous dot + committee move), in percentage points (0.25 = 25 bp)
Cells missing from your.csv take P1's probabilities and deviation 0, so any gain comes from the cells you cover.
Compared with P0 (2012-2015 frequencies per horizon group), P1 (logit on the distance to the previous median and its absolute
value, 2012-2015) and M0 (deviation 0, the two-step ceiling), on 2016-2020, excl. June 2020, 2016-2019 and the six
turning-point meetings (largest shares of the ceiling's error; chosen after seeing 2016-2020, so a diagnostic subset).
Tests: loss differential on a constant, two-way clustered (participant, meeting), t reference; wild cluster bootstrap by meeting.
Writes data/persona_eval_<NAME>.csv."""
import argparse, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser(); ap.add_argument('csv', nargs='?'); ap.add_argument('name', nargs='?'); ap.add_argument('--selftest', action='store_true')
a = ap.parse_args()
ZLB, TRAIN_END = 0.125, pd.Timestamp('2015-12-31')
TURNING = pd.to_datetime(['2016-03-15', '2016-06-14', '2016-12-13', '2019-03-19', '2019-12-10', '2020-06-09'])
SAMPLES = {'2016-2020': lambda d: (d.dt.year >= 2016) & (d.dt.year <= 2020),
           'excl. June 2020': lambda d: (d.dt.year >= 2016) & (d.dt.year <= 2020) & ~((d.dt.year == 2020) & (d.dt.month == 6)),
           '2016-2019': lambda d: (d.dt.year >= 2016) & (d.dt.year <= 2019),
           'turning points': lambda d: d.isin(TURNING)}
rd25 = lambda x: np.round(np.asarray(x, float) / 0.25) * 0.25
rng = np.random.default_rng(0)

D = pd.read_csv('data/persona_predictions.csv', parse_dates=['meeting_date'], dtype={'h': str, 'g': str})
D = D.sort_values(['meeting_date', 'person_id', 'proj_year']).reset_index(drop=True)
D['y'] = np.select([D.dev <= -0.1, D.dev >= 0.1], [0, 2], 1); D['abs_s'] = D.s.abs()
for g in ['1', '23', 'LR']: D[f'g{g}'] = (D.g == g).astype(float)
tr = D[D.meeting_date <= TRAIN_END]; te = D[D.meeting_date > TRAIN_END].copy()

# baselines, fitted on 2012-2015 exactly as in persona_information.py
freq = tr.groupby('g').y.value_counts(normalize=True).unstack().reindex(columns=[0, 1, 2]).fillna(0)
P0 = np.vstack([freq.loc[g].values for g in te.g])
cols = ['s', 'abs_s', 'g1', 'g23', 'gLR']; sc = StandardScaler().fit(tr[cols]); seps = sorted(tr.meeting_date.unique()); cut = seps[int(len(seps) * 0.75)]
A, B = tr[tr.meeting_date < cut], tr[tr.meeting_date >= cut]; mk = lambda C_: LogisticRegression(C=C_, max_iter=5000)
best = min([0.01, 0.1, 1.0, 10.0], key=lambda C_: log_loss(B.y, mk(C_).fit(sc.transform(A[cols]), A.y).predict_proba(sc.transform(B[cols])), labels=[0, 1, 2]))
P1 = mk(best).fit(sc.transform(tr[cols]), tr.y).predict_proba(sc.transform(te[cols]))

if a.selftest:
    pd.DataFrame({'meeting_date': te.meeting_date.dt.date, 'participant': te.person_id, 'horizon': te.h,
                  'p_down': P1[:, 0], 'p_same': P1[:, 1], 'p_up': P1[:, 2]}).to_csv('data/selftest_persona_P1.csv', index=False)
    print('wrote data/selftest_persona_P1.csv'); raise SystemExit
if not (a.csv and a.name): ap.error('give your.csv and NAME, or --selftest')

# the submitted signal
U = pd.read_csv(a.csv, parse_dates=['meeting_date']); U['horizon'] = U.horizon.astype(str).str.replace(r'\.0$', '', regex=True)
names = D.drop_duplicates('person_id').set_index('participant_name').person_id
U['person_id'] = U.participant.map(lambda p: names.get(p, p))
has_p, has_d = {'p_down', 'p_same', 'p_up'} <= set(U.columns), 'deviation' in U.columns
if not (has_p or has_d): ap.error('your.csv needs p_down, p_same, p_up and/or deviation')
if has_p:
    s = U[['p_down', 'p_same', 'p_up']].sum(1)
    if (s - 1).abs().max() > 1e-6: raise SystemExit(f'probabilities must sum to 1 (max deviation {(s - 1).abs().max():.2g})')
U = U.drop_duplicates(['meeting_date', 'person_id', 'horizon'])
te = te.reset_index(drop=True).merge(U.drop(columns='participant').rename(columns={'horizon': 'h'}), on=['meeting_date', 'person_id', 'h'], how='left')
covered = (te.p_down.notna() if has_p else te.deviation.notna()).values
print(f'{a.name}: {covered.sum()} of {len(te)} test cells covered ({covered.mean():.1%}); unmatched rows in your file: {len(U) - covered.sum()}')
PM = np.where(covered[:, None], te[['p_down', 'p_same', 'p_up']].values if has_p else P1, P1)
dev_hat = te.deviation.fillna(0.0).values if has_d else np.zeros(len(te))

def clustered(d, g1, g2):
    if np.allclose(d, 0): return np.nan
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': np.column_stack([pd.factorize(g1)[0], pd.factorize(g2)[0]])}, use_t=True)
    return float(r.pvalues[0])
def wild_p(d, groups, B=9999):
    d = np.asarray(d, float)
    if np.allclose(d, 0): return np.nan
    gi = pd.factorize(groups)[0]; G = gi.max() + 1; n = len(d)
    t = lambda x: x.mean() / (np.sqrt(G / (G - 1) * (np.bincount(gi, weights=x - x.mean(), minlength=G) ** 2).sum()) / n)
    W = rng.choice([-1.0, 1.0], size=(B, G))
    return float((np.abs([t(d * w[gi]) for w in W]) >= abs(t(d))).mean())

Y = np.eye(3)[te.y.values]
ll = lambda P: -np.log(np.clip(P[np.arange(len(te)), te.y.values], 1e-12, 1))
br = lambda P: ((P - Y) ** 2).sum(1)
e0 = (np.maximum(ZLB, te.lag + te.c) - te.federal_funds_rate).abs().values
em = (np.maximum(ZLB, te.lag + te.c + rd25(dev_hat)) - te.federal_funds_rate).abs().values
LOSS = {}
if has_p: LOSS.update({'log loss': {'P0': ll(P0), 'P1': ll(P1), a.name: ll(PM)}, 'Brier': {'P0': br(P0), 'P1': br(P1), a.name: br(PM)}})
if has_d: LOSS.update({'MAE (bp)': {'M0': 100 * e0, a.name: 100 * em},
                       'RMSE of the deviation (bp)': {'M0': 1e4 * te.dev.values ** 2, a.name: 1e4 * (te.dev.values - dev_hat) ** 2}})
rows = []
for sname, sel in SAMPLES.items():
    m = sel(te.meeting_date).values; gp, gm = te.person_id.values[m], te.meeting_date.values[m]
    for metric, L in LOSS.items():
        ref = 'P1' if 'P1' in L else 'M0'
        for k, v in L.items():
            val = np.sqrt(v[m].mean()) if metric.startswith('RMSE') else v[m].mean()
            d = v[m] - L[ref][m]
            rows.append(dict(sample=sname, cells=int(m.sum()), covered=int(covered[m].sum()), metric=metric, model=k, value=val,
                             vs_ref=None if k == ref else (val - (np.sqrt(L[ref][m].mean()) if metric.startswith('RMSE') else L[ref][m].mean())),
                             ref=ref, p=np.nan if k == ref else clustered(d, gp, gm), p_wild_meeting=np.nan if k == ref else wild_p(d, gm)))
R = pd.DataFrame(rows); R.to_csv(f'data/persona_eval_{a.name}.csv', index=False)
pd.set_option('display.width', 200); print(R.round(4).to_string(index=False))
