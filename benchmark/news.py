"""Shared inputs for the seventh and later batches (benchmarks_v7.py, benchmarks_v8.py). For every dot r_{i,t,H}:
  - the participant's own dot and SEP projections at his/her previous SEP(s) for the same calendar year H (LR by LR);
  - news between the previous SEP and t: policy rate (target midpoint), Treasury forward for the end of year H, SPF median
    bill-rate path for the end of year H, real-time macro data;
  - gaps of the own previous dot to the previous committee median, the current policy rate, the market and the SPF.
Pre-meeting information only: yields of the day before the meeting, the latest SPF survey released before the meeting."""
import numpy as np, pandas as pd
ZLB = 0.125
VARS = {'federal_funds_rate': 'ffr', 'gdp_growth_x': 'gdp', 'unemployment_rate': 'ur', 'pce_inflation': 'pce', 'core_pce_inflation': 'core'}
MAT = {'DTB3': .25, 'DTB6': .5, 'DGS1': 1, 'DGS2': 2, 'DGS3': 3, 'DGS5': 5, 'DGS10': 10}
MACRO = ['target_mid', 'core_inflation', 'unrate', 'gdp_growth_y']
NEWS = ['d_pol', 'd_fwd', 'd_spf', 'd_core_inflation', 'd_unrate', 'd_gdp_growth_y']
GAPS = ['gap_med', 'gap_now', 'gap_mkt', 'gap_spf']

def load_curve():
    Y = None
    for s in MAT:
        f = pd.read_csv(f'data/fred_{s}.csv', parse_dates=['observation_date']).rename(columns={'observation_date': 'date'})
        f[s] = pd.to_numeric(f[s], errors='coerce'); Y = f if Y is None else Y.merge(f, on='date', how='outer')
    return Y.sort_values('date').set_index('date').ffill()

def market(Y, day, py):
    """Instantaneous Treasury forward for 31 December of year py, the day before `day`; 5y5y forward for LR (py == -1).
    Same construction as B9 in benchmarks_v3.py."""
    row = Y[Y.index < day].iloc[-1]; m = np.array(list(MAT.values())); y = row[list(MAT)].values.astype(float)
    if py == -1: return (10 * y[-1] - 5 * y[-2]) / 5
    tau = max((pd.Timestamp(f'{int(py)}-12-31') - day).days / 365.25, .1); e = .05
    m0 = np.r_[0.0, m]; yt = np.r_[0.0, m * y]; a, b = max(tau - e, 0.0), tau + e
    return (np.interp(b, m0, yt) - np.interp(a, m0, yt)) / (b - a)

def load_spf():
    S = pd.read_csv('data/spf.csv')
    S['available'] = pd.to_datetime([f'{y}-{3 * q - 1:02d}-20' for y, q in zip(S.YEAR, S.QUARTER)])   # released mid-quarter
    return S.sort_values('available').reset_index(drop=True)

def spf(S, day, py):
    """SPF median 3-month bill rate for the end of year py (LR: expected 10-year average), latest survey before `day`.
    Quarterly forecasts reach four quarters ahead; further years use the midpoint of adjacent annual averages."""
    s = S[S.available < day]; r = s.iloc[-1]
    if py == -1: return s.BILL10.dropna().iloc[-1]
    k, q = int(py) - int(r.YEAR), int(r.QUARTER)
    if k == 0: return r[f'TBILL{6 - q}']
    if k == 1: return r.TBILL6 if q == 4 else (r.TBILLB + r.TBILLC) / 2
    if k == 2: return (r.TBILLC + r.TBILLD) / 2
    return r.TBILLD

def panel(V):
    """V: a predictions file of the pipeline (same rows and order as data/dataset.csv)."""
    D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date']).rename(columns=VARS)
    D['h'] = D.h.astype(str); D['proj_year'] = V.proj_year.values
    D['lag'] = V.B0_prior_own.values                 # own previous dot (fallbacks: own previous last year, committee median)
    D['prev_med'] = V.B1a_prev_median.values
    seps = sorted(D.meeting_date.unique()); prev = {t: (seps[i - 1] if i else None) for i, t in enumerate(seps)}
    for v in VARS.values():                          # own projections at the previous two SEPs, same calendar year
        val = dict(zip(zip(D.person_id, D.sep_index, D.proj_year), D[v]))
        for k in (1, 2):
            D[f'{v}_l{k}'] = [val.get((p, s - k, py), np.nan) for p, s, py in zip(D.person_id, D.sep_index, D.proj_year)]
    D['has_lag'] = D.ffr_l1.notna()
    dots = dict(zip(zip(D.person_id, D.sep_index, D.proj_year), D.ffr))
    D['next_l1'] = [dots.get((p, s - 1, py + 1), np.nan) if py > 0 else np.nan for p, s, py in zip(D.person_id, D.sep_index, D.proj_year)]
    D['lr_l1'] = [dots.get((p, s - 1, -1), np.nan) for p, s in zip(D.person_id, D.sep_index)]
    last = D[D.proj_year > 0].groupby(['person_id', 'sep_index']).proj_year.max()
    D['last_l1'] = [dots.get((p, s - 1, last.get((p, s - 1))), np.nan) for p, s in zip(D.person_id, D.sep_index)]
    mac = D.groupby('meeting_date')[MACRO].first()
    for c in MACRO:
        D['d_' + c] = [mac.loc[t, c] - mac.loc[prev[t], c] if prev[t] is not None else np.nan for t in D.meeting_date]
    D = D.rename(columns={'d_target_mid': 'd_pol'})
    Y, S = load_curve(), load_spf()
    cache = {}
    def get(f, src, t, py):
        if (f, t, py) not in cache: cache[(f, t, py)] = f(src, pd.Timestamp(t), py)
        return cache[(f, t, py)]
    D['fwd'] = [get(market, Y, t, py) for t, py in zip(D.meeting_date, D.proj_year)]
    D['spf'] = [get(spf, S, t, py) for t, py in zip(D.meeting_date, D.proj_year)]
    D['d_fwd'] = [D.fwd[i] - get(market, Y, prev[t], py) if prev[t] is not None else np.nan for i, t, py in zip(D.index, D.meeting_date, D.proj_year)]
    D['d_spf'] = [D.spf[i] - get(spf, S, prev[t], py) if prev[t] is not None else np.nan for i, t, py in zip(D.index, D.meeting_date, D.proj_year)]
    D['gap_med'] = D.prev_med - D.lag; D['gap_now'] = D.target_mid - D.lag
    D['gap_mkt'] = D.fwd - D.lag; D['gap_spf'] = D.spf - D.lag
    D['hn'] = D.h.replace('LR', '4').astype(int)
    return D
