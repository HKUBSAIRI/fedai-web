"""First benchmarks for individual SEP fed funds rate dots. Every prediction for SEP t uses only information available
before meeting t: pre-meeting real-time macro data and the dots of earlier SEPs. Models with parameters are re-estimated
on an expanding window (all earlier SEPs) for every SEP."""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from protocol import fitset
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
D['proj_year'] = np.where(D.h == 'LR', -1, D.year + D.h.replace('LR', '0').astype(int))
meetings = sorted(D.meeting_date.unique())
ZLB = 0.125
H = ['0', '1', '2', '3', 'LR']

def prev_sep(t):  # dots of the most recent earlier SEP
    i = meetings.index(t)
    return D[D.meeting_date == meetings[i - 1]] if i > 0 else None

preds = {}
def put(name, idx, values):
    preds.setdefault(name, pd.Series(np.nan, index=D.index)).loc[idx] = values

for t in meetings:
    cur = D[D.meeting_date == t]; prev = prev_sep(t); hist = D[D.meeting_date < t]
    tmid = cur.target_mid.iloc[0]
    # ---- committee median of the previous SEP, by calendar target year (LR by LR); new far year -> previous last year
    med = prev.groupby('proj_year').federal_funds_rate.median() if prev is not None else pd.Series(dtype=float)
    def med_for(py):
        if py in med.index: return med[py]
        fin = med[med.index > 0]
        return fin.iloc[-1] if py > 0 and len(fin) else (med.get(-1, np.nan) if py == -1 else np.nan)
    # B1b: no change — current target midpoint for calendar years; LR from previous median
    b1b = [tmid if py > 0 else med_for(-1) for py in cur.proj_year]
    put('B1b_no_change', cur.index, b1b)
    # B1a: previous committee median (fallback: no change)
    b1a = [med_for(py) if not np.isnan(med_for(py)) else b for py, b in zip(cur.proj_year, b1b)]
    put('B1a_prev_median', cur.index, b1a)
    # B0: own previous dot for the same calendar year (LR by LR); new far year -> own previous last year; newcomer -> B1a
    own = prev.set_index(['person_id', 'proj_year']).federal_funds_rate if prev is not None else pd.Series(dtype=float)
    b0 = [] if prev is not None else list(b1a)
    for (pid, py), fb in (zip(zip(cur.person_id, cur.proj_year), b1a) if prev is not None else []):
        if (pid, py) in own.index: b0.append(own[(pid, py)]); continue
        mine = own[own.index.get_level_values(0) == pid]
        fin = mine[mine.index.get_level_values(1) > 0]
        b0.append(fin.iloc[-1] if py > 0 and len(fin) else fb)
    put('B0_prior_own', cur.index, b0)
    # B2a: Taylor (1993) with real-time data, unemployment gap via Okun (2x), r*=2, u*=5 (fixed), floored at ZLB; LR -> previous median
    tay = max(ZLB, 2 + cur.core_inflation.iloc[0] + 0.5 * (cur.core_inflation.iloc[0] - 2) + 2 * 0.5 * (5.0 - cur.unrate.iloc[0]))
    put('B2a_taylor1993', cur.index, [tay if py > 0 else med_for(-1) for py in cur.proj_year])
    # B2b: estimated real-time reaction function per horizon (committee-level, expanding window)
    #      dot ~ target_mid + core_inflation + unrate (+ previous median for the same year)
    if len(hist) >= 30:
        hist = hist.copy(); cur = cur.copy()
        for frame in (hist, cur):
            frame['prev_med'] = np.nan
        # previous-median feature for history rows
        hm = []
        for tt, g in hist.groupby('meeting_date'):
            pv = prev_sep(tt)
            m2 = pv.groupby('proj_year').federal_funds_rate.median() if pv is not None else pd.Series(dtype=float)
            hm.append(pd.Series([m2.get(py, m2[m2.index > 0].iloc[-1] if py > 0 and (m2.index > 0).any() else (m2.get(-1, np.nan))) if len(m2) else np.nan for py in g.proj_year], index=g.index))
        hist['prev_med'] = pd.concat(hm)
        cur['prev_med'] = b1a
        X = ['target_mid', 'core_inflation', 'unrate']
        for h in H:
            tr = hist[(hist.h == h)].dropna(subset=X + ['prev_med'])
            tf = fitset(tr)                                   # rows used to estimate (all of tr unless FEDAI_TRAIN_END)
            te = cur[cur.h == h]
            if len(tf) < 20 or te.empty: continue
            m = LinearRegression().fit(tf[X], tf.federal_funds_rate)
            put('B2b_rt_rule', te.index, np.maximum(ZLB, m.predict(te[X])))
            m2 = LinearRegression().fit(tf[X + ['prev_med']], tf.federal_funds_rate)
            p2 = np.maximum(ZLB, m2.predict(te[X + ['prev_med']]))
            put('B2c_rt_rule_smooth', te.index, p2)
            # B4: B2c + person-specific average past residual (shrunk toward 0)
            res = tr.federal_funds_rate - m2.predict(tr[X + ['prev_med']])
            fe = res.groupby(tr.person_id).agg(['sum', 'count'])
            k = 5.0
            shrunk = (fe['sum'] / (fe['count'] + k)).to_dict()
            put('B4_rule_plus_person', te.index, np.maximum(ZLB, p2 + te.person_id.map(shrunk).fillna(0).values))
    # B3: personal Taylor rule on the participant's OWN same-round projections (uses same-SEP info: reference only)
    if len(hist) >= 30:
        for h in [x for x in H if x != 'LR']:
            tr = fitset(hist[(hist.h == h)].dropna(subset=['core_pce_inflation', 'unemployment_rate']))
            te = cur[cur.h == h].dropna(subset=['core_pce_inflation', 'unemployment_rate'])
            if len(tr) < 20 or te.empty: continue
            Xs = ['core_pce_inflation', 'unemployment_rate', 'target_mid']
            m = LinearRegression().fit(tr[Xs], tr.federal_funds_rate)
            put('B3_own_projection_rule*', te.index, np.maximum(ZLB, m.predict(te[Xs])))

out = D[['meeting_date', 'person_id', 'participant_name', 'h', 'proj_year', 'federal_funds_rate']].copy()
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v1.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
