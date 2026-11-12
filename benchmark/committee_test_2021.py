"""Committee-level test of the committee-shift rules on the SEPs of 2021-2026, whose individual dots are not public yet.
The two-step baselines forecast each participant's dot as own previous dot + c_hat, where c_hat is the predicted change of
the committee median for the same target year. The committee median is public on the meeting day, so c_hat can be tested
on 2021-2026 now. The same rules are scored on 2012-2020 for comparison (there the individual results are known).

Cells: SEP t x calendar target year H with a median for H at the previous SEP (new far years excluded); LR by LR.
Truth: c = median_t(H) - median_{t-}(H). Medians from the individual dots (2012-2020, database export) and from the
published dot-plot counts (2021-2026, docs/public_fomc; March 2022 from a Wayback snapshot of the Fed's page).
News, same construction as news.py: d_spf = change of the SPF median bill-rate path for the end of year H since the
previous SEP (latest survey released before the meeting); d_fwd = change of the Treasury instantaneous forward for
31 December of year H (day before the meeting); c_prev = the median's change at the previous SEP for the same year.

Rules for c_hat (calendar years; LR: 0), all written before this script was first run:
  N0 naive 0 | C22 d_fwd | C22b d_spf | C30a rd25(d_spf) | C30b median(rd25 d_spf, rd25 d_fwd, rd25 c_prev/2) |
  C31 S(d_spf, 12.5bp), beta 1 | C31e 0.31 S(d_spf, 12.5bp), beta of the 2012-2015 censored LAD fit | Cmom c_prev/2 |
  Ceq mean(d_spf, d_fwd, c_prev/2) | Cg12 rd12.5(d_spf) (the median moves on a 12.5 bp grid)
Metrics: MAE of c_hat against c in bp by period and horizon; direction when the median moved (|c| >= 12.5 bp); paired
test against naive with standard errors clustered by SEP. Also: what the lower-bound rule of B30d would have done.
Usage: python committee_test_2021.py   (from anywhere; paths are relative to the repository)
Writes docs/benchmark/results/committee_2021/{cells.csv, mae_by_period.csv, mae_by_horizon.csv, tests.csv, zlb.csv}."""
import os, sys, hashlib, numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import news
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
P = lambda *a: os.path.join(REPO, *a)
OUT = P('docs', 'benchmark', 'results', 'committee_2021'); os.makedirs(OUT, exist_ok=True)
print('committee_test_2021.py sha256', hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16])
ZLB, BETA_HAT = 0.125, 0.31

# ---------- committee medians, 2012-2020 (individual dots) and 2021-2026 (published counts)
I = pd.read_csv(P('docs', 'benchmark', 'work', 'frozen_2015', 'data', 'projections.csv'), parse_dates=['meeting_date'])
I = I[I.federal_funds_rate.notna()].copy()
I['federal_funds_rate'] = np.round(I.federal_funds_rate / 0.125) * 0.125   # the database stores dots to two decimals (0.13 for 0.125); the published 2021+ dots are exact
I['target'] = np.where(I.is_long_run, 'LR', I.projection_year.astype('Int64').astype(str))
M1 = I.groupby(['meeting_date', 'target']).federal_funds_rate.agg(['median', 'count']).reset_index()
pub = pd.concat([pd.read_csv(P('docs', 'public_fomc', 'data', 'sep_dots.csv')), pd.read_csv(P('docs', 'benchmark', 'code', 'inputs', 'sep_dots_2022_03.csv'))])
pub['meeting_date'] = pd.to_datetime(pub.meeting_date); pub['target'] = pub.target.replace({'Longer run': 'LR', 'longer_run': 'LR'}).astype(str)
M2 = pub.groupby(['meeting_date', 'target']).apply(lambda g: pd.Series({'median': float(np.median(np.repeat(g.rate.values, g.n_participants.values))),
                                                                         'count': int(g.n_participants.sum())}), include_groups=False).reset_index()
summ = pd.read_csv(P('docs', 'public_fomc', 'data', 'sep_summary.csv'), parse_dates=['meeting_date'])
summ = summ[summ.variable.str.contains('fed', case=False)].copy(); summ['target'] = summ.target.astype(str).replace({'Longer run': 'LR', 'longer_run': 'LR'})
chk = M2.merge(summ[['meeting_date', 'target', 'median']], on=['meeting_date', 'target'], suffixes=('', '_tbl'))
print('2021-2026 medians vs Table 1 (one decimal): max |diff| =', round((chk['median'] - chk.median_tbl).abs().max(), 3), 'over', len(chk), 'cells')
M = pd.concat([M1, M2]).sort_values(['meeting_date', 'target']).reset_index(drop=True)
seps = sorted(M.meeting_date.unique()); prev = {t: (seps[i - 1] if i else None) for i, t in enumerate(seps)}
med = dict(zip(zip(M.meeting_date, M.target), M['median']))

# ---------- news
macro = pd.read_csv(P('docs', 'benchmark', 'work', 'frozen_2015', 'data', 'macro.csv'), parse_dates=['meeting_date']).set_index('meeting_date')
macro['mid'] = (macro.target_lower + macro.target_upper) / 2
Y = None
for s in news.MAT:
    parts = [pd.read_csv(P('docs', 'benchmark', 'work', d, 'data', f'fred_{s}.csv'), parse_dates=['observation_date']).rename(columns={'observation_date': 'date'})
             for d in ('frozen_2015', 'committee_2021')]
    f = pd.concat(parts).drop_duplicates('date'); f[s] = pd.to_numeric(f[s], errors='coerce')
    Y = f if Y is None else Y.merge(f, on='date', how='outer')
Y = Y.sort_values('date').set_index('date').ffill()
S = pd.read_csv(P('docs', 'benchmark', 'code', 'inputs', 'spf_median_2010_2026.csv'))
S['available'] = pd.to_datetime([f'{y}-{3 * q - 1:02d}-20' for y, q in zip(S.YEAR, S.QUARTER)]); S = S.sort_values('available').reset_index(drop=True)

rows = []
for t in seps:
    tp = prev[t]
    if tp is None: continue
    for target in M[M.meeting_date == t].target:
        if (tp, target) not in med: continue
        lr = target == 'LR'; py = -1 if lr else int(target); h = 'LR' if lr else str(py - t.year)
        if not lr and h not in ('0', '1', '2', '3'): continue
        m_prev, m_now = med[(tp, target)], med[(t, target)]
        tpp = prev[tp]; c_prev = (med[(tp, target)] - med[(tpp, target)]) if tpp is not None and (tpp, target) in med else 0.0
        d_spf = news.spf(S, t, py) - news.spf(S, tp, py); d_fwd = news.market(Y, t, py) - news.market(Y, tp, py)
        rows.append(dict(meeting_date=t, target=target, h=h, m_prev=m_prev, m_now=m_now, c=m_now - m_prev, c_prev=c_prev, d_spf=d_spf, d_fwd=d_fwd,
                         rate=macro.mid.get(t, np.nan), d_pol=macro.mid.get(t, np.nan) - macro.mid.get(tp, np.nan)))
C = pd.DataFrame(rows)
def rd(x, step=0.25): x = np.asarray(x, float); return np.sign(x) * np.floor(np.abs(x) / step + 0.5 - 1e-9) * step
def soft(x, d): x = np.asarray(x, float); return np.sign(x) * np.maximum(np.abs(x) - d, 0.0)
cal = (C.h != 'LR').values
z1, z2, z3 = rd(C.d_spf), rd(C.d_fwd), rd(C.c_prev / 2)
R = {'N0 naive': np.zeros(len(C)), 'C22 d_fwd': C.d_fwd.values, 'C22b d_spf': C.d_spf.values, 'C30a rd25(d_spf)': z1,
     'C30b median(z1,z2,z3)': np.median(np.column_stack([z1, z2, z3]), axis=1), 'C31 soft beta=1': soft(C.d_spf, 0.125),
     'C31e soft beta=0.31': BETA_HAT * soft(C.d_spf, 0.125), 'Cmom c_prev/2': C.c_prev.values / 2,
     'Ceq mean(d_spf,d_fwd,c_prev/2)': (C.d_spf + C.d_fwd + C.c_prev / 2).values / 3, 'Cg12 rd12.5(d_spf)': rd(C.d_spf, 0.125),
     # added after the first run (post hoc for this test): the grid version of Ceq, to choose between B40 and B40g for 2021
     'Ceqg rd25(mean) [post hoc]': rd((C.d_spf + C.d_fwd + C.c_prev / 2) / 3)}
for k in R: R[k] = np.where(cal, R[k], 0.0)
for k, v in R.items(): C[k] = v
METHODS = list(R)
C['period'] = pd.cut(C.meeting_date.dt.year, [2011, 2015, 2019, 2020, 2021, 2022, 2023, 2026],
                     labels=['2012-2015 (estimation)', '2016-2019', '2020', '2021', '2022', '2023', '2024-2026']).astype(str)
C.to_csv(os.path.join(OUT, 'cells.csv'), index=False)

def mae_table(df, by):
    out = {}
    for m in METHODS:
        ok = df[m].notna(); e = (df[m] - df.c).abs() * 100
        out[m] = e[ok].groupby(df[by][ok]).mean()
    T = pd.DataFrame(out).T; T.loc['n cells'] = df.groupby(by).size(); return T
pd.set_option('display.width', 250)
A = mae_table(C[cal], 'period'); A.to_csv(os.path.join(OUT, 'mae_by_period.csv'))
print('\n== MAE of the committee-shift forecast (bp), calendar-year cells, by period'); print(A.round(1).to_string())
T21 = C[cal & (C.meeting_date >= '2021-01-01')]
B = mae_table(T21, 'h'); B['all'] = pd.Series({m: ((T21[m] - T21.c).abs() * 100)[T21[m].notna()].mean() for m in METHODS})
B.to_csv(os.path.join(OUT, 'mae_by_horizon.csv')); print('\n== 2021-2026 by horizon (bp)'); print(B.round(1).to_string())
LR21 = C[~cal & (C.meeting_date >= '2021-01-01')]
print('\n== 2021-2026 longer run: naive', round((LR21.c.abs() * 100).mean(), 1), 'bp on', len(LR21), 'cells')

# ---------- direction, hits and paired tests against naive (2021-2026 and 2016-2020)
def tests(df, label):
    out = []; e0 = (df.c.abs() * 100)
    for m in METHODS:
        ok = df[m].notna(); e = (df[m] - df.c).abs() * 100; d = (e - e0)[ok]
        if ok.sum() < 8 or np.allclose(d, 0): p = np.nan
        else: p = float(sm.OLS(d.values, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df.meeting_date[ok])[0]}, use_t=True).pvalues[0])
        moved = ok & (df.c.abs() >= 0.125 - 1e-9)
        out.append(dict(sample=label, method=m, n=int(ok.sum()), MAE_bp=e[ok].mean(), naive_MAE_bp=e0[ok].mean(),
                        within_6bp=float((e[ok] <= 0.0625 + 1e-9).mean()), direction_right=float((np.sign(df[m][moved]) == np.sign(df.c[moved])).mean()) if moved.any() else np.nan,
                        n_moved=int(moved.sum()), p_vs_naive=p))
    return pd.DataFrame(out)
TS = pd.concat([tests(C[cal & (C.meeting_date >= '2021-01-01')], '2021-2026'), tests(C[cal & (C.meeting_date >= '2022-01-01') & (C.meeting_date <= '2023-12-31')], '2022-2023 hiking'),
                tests(C[cal & (C.meeting_date >= '2016-01-01') & (C.meeting_date <= '2020-12-31')], '2016-2020'), tests(C[cal & (C.meeting_date >= '2016-01-01') & (C.meeting_date <= '2019-12-31')], '2016-2019')])
TS.to_csv(os.path.join(OUT, 'tests.csv'), index=False)
print('\n== paired against naive (SEs clustered by SEP); within_6bp = exact on the 12.5 bp grid; direction when the median moved >= 12.5 bp')
print(TS.round(3).to_string(index=False))

# ---------- what the lower-bound rule (B30d) would have done: SEPs with the rate at 0.125
above = macro[macro.mid > ZLB + 1e-9].index
Z = C[cal & (C.rate <= ZLB + 1e-9)].copy()
Z['days_since_above'] = [(t - above[above < t].max()).days if (above < t).any() else np.nan for t in Z.meeting_date]
Z['rule_fires'] = Z.days_since_above <= 365
Z['median_level'] = Z.m_now; Z['rule_level_Y0Y1'] = np.where(Z.h.isin(['0', '1']), ZLB, np.nan)
Z = Z[['meeting_date', 'target', 'h', 'days_since_above', 'rule_fires', 'm_prev', 'm_now', 'c']].sort_values(['meeting_date', 'h'])
Z.to_csv(os.path.join(OUT, 'zlb.csv'), index=False)
print('\n== SEPs with the policy rate at the lower bound: medians by horizon (the B30d rule fires when days_since_above <= 365)')
print(Z.assign(c_bp=lambda d: (d.c * 100).round(0)).drop(columns='c').to_string(index=False))
print('\nwritten to', OUT)
