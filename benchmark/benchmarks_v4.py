"""Fourth batch: add the hawk/dove tone of each participant's remarks at meetings BEFORE the SEP to the revision model."""
import numpy as np, pandas as pd, warnings
from protocol import fitset
from sklearn.linear_model import RidgeCV
from scipy import stats
warnings.filterwarnings('ignore')
V3 = pd.read_csv('data/predictions_v3.csv', parse_dates=['meeting_date'])
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
for c in V3.columns:
    if c.startswith('B'): D[c] = V3[c]
S = pd.read_csv('data/text_tone.csv', parse_dates=['meeting_date'])
meetings = sorted(D.meeting_date.unique()); ZLB = 0.125
# tone features: exponentially weighted average of the person's relative tone over meetings strictly before t (half-life 2 meetings)
tone_avg, tone_last = [], []
for t, pid in zip(D.meeting_date, D.person_id):
    s = S[(S.person_id == pid) & (S.meeting_date < t)].sort_values('meeting_date').tail(6)
    if s.empty: tone_avg.append(0.0); tone_last.append(0.0); continue
    w = 0.5 ** (np.arange(len(s))[::-1] / 2); tone_avg.append(float(np.sum(w * s.tone_rel) / w.sum())); tone_last.append(float(s.tone_rel.iloc[-1]))
D['tone_avg'] = tone_avg; D['tone_last'] = tone_last
# does tone predict how far a person sits above the committee median? (deviation from same-SEP median, all horizons except LR)
dev = D.federal_funds_rate - D.groupby(['meeting_date', 'h']).federal_funds_rate.transform('median')
m = (D.h != 'LR') & (D.tone_avg != 0)
print('corr(tone_avg, deviation from median):', round(np.corrcoef(D.tone_avg[m], dev[m])[0, 1], 3), 'n', int(m.sum()), 'p', stats.pearsonr(D.tone_avg[m], dev[m])[1])
rev = D.federal_funds_rate - D.B0_prior_own
m2 = m & rev.notna()
print('corr(tone_last, revision vs own previous dot):', round(np.corrcoef(D.tone_last[m2], rev[m2])[0, 1], 3), 'p', stats.pearsonr(D.tone_last[m2], rev[m2])[1])
# revision features as in v2/v3
mac = D.groupby('meeting_date')[['target_mid', 'core_inflation', 'unrate']].first()
prevm = {mm: (meetings[i - 1] if i else None) for i, mm in enumerate(meetings)}
for c in ['target_mid', 'core_inflation', 'unrate']:
    D['d_' + c] = [mac.loc[mm, c] - mac.loc[prevm[mm], c] if prevm[mm] is not None else 0.0 for mm in D.meeting_date]
D['gap_med'] = D.B1a_prev_median - D.B0_prior_own; D['gap_now'] = D.target_mid - D.B0_prior_own
D['hn'] = D.h.replace('LR', '4').astype(int)
X = ['d_target_mid', 'd_core_inflation', 'd_unrate', 'gap_med', 'gap_now', 'hn', 'voter', 'target_mid', 'unrate', 'core_inflation', 'tone_avg', 'tone_last']
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[D.meeting_date < t].dropna(subset=X + ['B0_prior_own'])
    if hist.meeting_date.nunique() < 6: continue
    ok = cur.dropna(subset=X + ['B0_prior_own'])
    for hs in (['0'], ['1'], ['2', '3'], ['LR']):
        tr = fitset(hist[hist.h.isin(hs)]); te = ok[ok.h.isin(hs)]
        if len(tr) < 30 or te.empty: continue
        r = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(tr[X], tr.federal_funds_rate - tr.B0_prior_own)
        put('B6c_ridge_text_tone', te.index, np.maximum(ZLB, te.B0_prior_own + r.predict(te[X])))
out = V3.copy()
for k, v in preds.items(): out[k] = v
# B14: combination B12 + text ridge on the current-year horizon only (where ridge models are strongest)
out['B14_combo_plus_text_h0'] = np.where(out.h == '0', out[['B12_combo_B0_B5_B8', 'B6c_ridge_text_tone']].mean(axis=1), out.B12_combo_B0_B5_B8)
out.to_csv('data/predictions_v4.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
