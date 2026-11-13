"""Compare an external forecaster (the agent, an LLM baseline) with the statistical baselines on the unified cells.
Rates, on the three test samples (2016-2020; 2016-2020 without June 2020; 2016-2019):
  - MAE in bp on the cells the external forecaster covers, with the baselines on the same cells;
  - paired tests of absolute errors against naive (B0) and against B30b, B32 and B39, standard errors clustered by participant
    and by meeting (as in evaluate_unified.py);
  - a split of every forecast into a committee part and an individual part. For the cells of one meeting and target year:
      committee part  = median over participants of (forecast - own previous dot);  truth: median of the actual revisions
      individual part = (forecast - own previous dot) - committee part;              truth: actual revision - that median
    Rules that move everyone by the same amount have an individual part of (almost) zero; a forecaster that knows the person
    should predict the individual part. The two-step ceiling (previous dot + the actual median revision, floored at 0.125) is
    shown as the reference with a perfect committee part.
Macro variables, if the CSV has them: MAE in percentage points against naive (own previous projection for the same target).
Usage, in the work directory (data/projections.csv is read for the macro variables):
  python compare_agent.py data/predictions_v10.csv --add agent.csv Frozen [--add initial.csv Initial ...]
  CSV columns: meeting_date, participant (person_id or name), horizon (0-3, "Y+1" style or LR), prediction (percent), and
  optionally variable (federal funds rate, GDP growth, unemployment, PCE inflation, core PCE inflation; free spelling).
Writes data/agent_comparison.csv, data/agent_decomposition.csv and data/agent_macro.csv."""
import argparse, re, numpy as np, pandas as pd
import statsmodels.api as sm

ap = argparse.ArgumentParser(); ap.add_argument('pred')
ap.add_argument('--add', nargs=2, action='append', required=True, metavar=('CSV', 'NAME'))
ap.add_argument('--vs', nargs='+', default=['B30b_grid_median_signals', 'B32_linear_tree', 'B39_median_of_three'])
a = ap.parse_args()
ZLB = 0.125
SAMPLES = {'2016-2020': lambda d: (d.year >= 2016) & (d.year <= 2020),
           'excl. June 2020': lambda d: (d.year >= 2016) & (d.year <= 2020) & ~((d.year == 2020) & (d.month == 6)),
           '2016-2019': lambda d: (d.year >= 2016) & (d.year <= 2019)}
VARS = ['federal_funds_rate', 'gdp_growth', 'unemployment_rate', 'pce_inflation', 'core_pce_inflation']

# ---------- baselines and the unified cells (as in evaluate_unified.py)
P = pd.read_csv(a.pred, parse_dates=['meeting_date']); P['h'] = P.h.astype(str)
P = P[P.federal_funds_rate.notna()].reset_index(drop=True)
own = P[['person_id', 'meeting_date']].drop_duplicates().sort_values('meeting_date'); prev_own = {}
for pid, g in own.groupby('person_id'):
    d = list(g.meeting_date); prev_own.update({(pid, x): y for x, y in zip(d[1:], d[:-1])})
dots = P.set_index(['person_id', 'meeting_date', 'proj_year']).federal_funds_rate
P['lag'] = [dots.get((p, prev_own.get((p, t)), y), np.nan) if prev_own.get((p, t)) is not None else np.nan
            for p, t, y in zip(P.person_id, P.meeting_date, P.proj_year)]
seps = np.array(sorted(P.meeting_date.unique()), dtype='datetime64[ns]')
names = P.drop_duplicates('person_id')[['person_id', 'participant_name']]

def norm_var(v):
    s = re.sub(r'[^a-z]', '', str(v).lower())
    if 'core' in s: return 'core_pce_inflation'
    if 'unemploy' in s or 'unrate' in s: return 'unemployment_rate'
    if 'gdp' in s or 'growth' in s: return 'gdp_growth'
    if 'pce' in s or 'inflation' in s: return 'pce_inflation'
    if 'fund' in s or 'ffr' in s or 'policy' in s or s in ('rate', 'fedrate'): return 'federal_funds_rate'
    return None

def norm_h(h):
    s = str(h).upper().replace(' ', '').replace('_', '')
    if s in ('LR', 'LONGERRUN', 'LONGRUN'): return 'LR'
    s = s.replace('Y+', '').replace('Y', '').replace('+', '')
    return str(int(float(s))) if s.replace('.', '').isdigit() else None

def person(p):
    if p in set(names.person_id): return p
    s = str(p).strip().lower()
    hit = names[names.participant_name.str.lower() == s]
    if len(hit) == 1: return hit.person_id.iloc[0]
    last = s.replace(',', ' ').split()[-1] if s else ''
    hit = names[names.participant_name.str.lower().str.split().str[-1] == last]
    return hit.person_id.iloc[0] if len(hit) == 1 else None

def load_external(path, name):
    X = pd.read_csv(path)
    X['meeting_date'] = pd.to_datetime(X.meeting_date)
    near = {}
    for t in X.meeting_date.unique():                         # the agent may date a meeting by its first day
        k = np.abs(seps - np.datetime64(t, 'ns')).argmin()
        near[t] = pd.Timestamp(seps[k]) if abs(pd.Timestamp(seps[k]) - pd.Timestamp(t)).days <= 3 else pd.NaT
    X['meeting_date'] = X.meeting_date.map(near)
    X['person_id'] = X.participant.map({p: person(p) for p in X.participant.unique()})
    X['h'] = X.horizon.map(norm_h)
    X['variable'] = X.variable.map(norm_var) if 'variable' in X else 'federal_funds_rate'
    X['prediction'] = pd.to_numeric(X.prediction, errors='coerce')
    bad = X.meeting_date.isna() | X.person_id.isna() | X.h.isna() | X.variable.isna()
    print(f'{name}: {len(X)} rows, {int(bad.sum())} unmatched' +
          (f" (participants: {sorted(map(str, X.participant[X.person_id.isna()].unique()))[:8]})" if X.person_id.isna().any() else ''))
    return X[~bad]

EXT = {}
for path, name in a.add:
    X = load_external(path, name); EXT[name] = X
    R = X[X.variable == 'federal_funds_rate'][['meeting_date', 'person_id', 'h', 'prediction']].drop_duplicates(['meeting_date', 'person_id', 'h'])
    P = P.merge(R.rename(columns={'prediction': name}), on=['meeting_date', 'person_id', 'h'], how='left')

# ---------- two-step ceiling: previous dot + the actual median revision of the cell's meeting and target year
U = P.lag.notna()
P['rev'] = P.federal_funds_rate - P.lag
P['ceiling'] = np.where(U, np.maximum(ZLB, P.lag + P[U].groupby(['meeting_date', 'proj_year']).rev.transform('median')), np.nan)
BASE = ['B0_prior_own'] + [m for m in a.vs if m in P]
NAMES = list(EXT)

def clustered_p(d, g1, g2):
    if len(d) < 10 or np.allclose(d, 0): return np.nan
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type='cluster', cov_kwds={'groups': np.column_stack([pd.factorize(g1)[0], pd.factorize(g2)[0]])}, use_t=True)
    return float(r.pvalues[0])

rows, dec = [], []
for s, f in SAMPLES.items():
    T0 = P[U & f(P.meeting_date.dt)]
    for x in NAMES:
        T = T0[T0[x].notna()]
        ex = (T[x] - T.federal_funds_rate).abs()
        rows.append(dict(sample=s, method=x, n=len(T), coverage=len(T) / len(T0), MAE_bp=100 * ex.mean(), compared_with='', diff_bp=np.nan, p=np.nan))
        for b in BASE + ['ceiling']:
            eb = (T[b] - T.federal_funds_rate).abs()
            rows.append(dict(sample=s, method=b, n=len(T), coverage=np.nan, MAE_bp=100 * eb.mean(), compared_with=x,
                             diff_bp=100 * (ex - eb).mean(), p=clustered_p((ex - eb).values, T.person_id.values, T.meeting_date.values) if b != 'ceiling' else np.nan))
    # committee and individual parts, on the cells every method covers
    M = NAMES + BASE + ['ceiling']
    T = T0.dropna(subset=M).copy(); key = [T.meeting_date, T.proj_year]
    c_true = T.groupby(key).rev.transform('median'); i_true = T.rev - c_true
    for m in M:
        fm = T[m] - T.lag; c_hat = fm.groupby(key).transform('median'); i_hat = fm - c_hat
        moved = i_true.abs() > 1e-9
        dec.append(dict(sample=s, method=m, n=len(T), MAE_bp=100 * (T[m] - T.federal_funds_rate).abs().mean(),
                        committee_MAE_bp=100 * (c_hat - c_true).abs().mean(), individual_MAE_bp=100 * (i_hat - i_true).abs().mean(),
                        individual_corr=float(np.corrcoef(i_hat, i_true)[0, 1]) if i_hat.std() > 1e-9 and i_true.std() > 1e-9 else np.nan,
                        individual_sign_right=float((np.sign(i_hat[moved]) == np.sign(i_true[moved])).mean()) if moved.any() else np.nan))
C = pd.DataFrame(rows); D = pd.DataFrame(dec)
C.to_csv('data/agent_comparison.csv', index=False); D.to_csv('data/agent_decomposition.csv', index=False)
pd.set_option('display.width', 250)
for s in SAMPLES:
    print(f'\n== {s}: rate MAE (bp); diff = external minus baseline on the same cells, p from the paired test')
    print(C[C['sample'] == s].drop(columns='sample').round(3).to_string(index=False))
    print(f'-- {s}: committee and individual parts (bp), cells covered by every method')
    print(D[D['sample'] == s].drop(columns='sample').round(3).to_string(index=False))

# ---------- macro variables against naive, if any
try:
    J = pd.read_csv('data/projections.csv', parse_dates=['meeting_date'])
except FileNotFoundError:
    J = None
mac = []
for x, X in EXT.items():
    Xm = X[X.variable != 'federal_funds_rate']
    if Xm.empty or J is None: continue
    J['h'] = np.where(J.is_long_run, 'LR', (J.projection_year - J.meeting_date.dt.year).astype('Int64').astype(str))
    J['target'] = np.where(J.is_long_run, 'LR', J.projection_year.astype('Int64').astype(str))
    Lg = J.melt(id_vars=['meeting_date', 'person_id', 'h', 'target'], value_vars=VARS[1:], var_name='variable', value_name='actual').dropna(subset=['actual'])
    Lg = Lg.sort_values('meeting_date')
    seq = J[['person_id', 'meeting_date']].drop_duplicates().sort_values('meeting_date'); pv = {}
    for pid, g in seq.groupby('person_id'):
        d = list(g.meeting_date); pv.update({(pid, u): w for u, w in zip(d[1:], d[:-1])})
    val = Lg.set_index(['person_id', 'meeting_date', 'target', 'variable']).actual
    Lg['naive'] = [val.get((p, pv.get((p, t)), tg, v), np.nan) if pv.get((p, t)) is not None else np.nan
                   for p, t, tg, v in zip(Lg.person_id, Lg.meeting_date, Lg.target, Lg.variable)]
    Z = Lg.merge(Xm[['meeting_date', 'person_id', 'h', 'variable', 'prediction']], on=['meeting_date', 'person_id', 'h', 'variable']).dropna(subset=['naive'])
    for s, f in SAMPLES.items():
        for v, g in Z[f(Z.meeting_date.dt)].groupby('variable'):
            ex, e0 = (g.prediction - g.actual).abs(), (g.naive - g.actual).abs()
            mac.append(dict(sample=s, method=x, variable=v, n=len(g), MAE_pp=ex.mean(), naive_MAE_pp=e0.mean(),
                            p_vs_naive=clustered_p((ex - e0).values, g.person_id.values, g.meeting_date.values)))
if mac:
    Mc = pd.DataFrame(mac); Mc.to_csv('data/agent_macro.csv', index=False)
    print('\n== macro variables: MAE (percentage points) against naive on matched cells'); print(Mc.round(3).to_string(index=False))
