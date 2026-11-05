"""Score any method against all benchmarks on the same rows.

Input CSV (one row per participant, SEP and horizon), columns:
  meeting_date   SEP meeting start date, YYYY-MM-DD
  participant    person_id (uuid) or the name used in the SEP data (e.g. "Loretta J. Mester")
  horizon        0, 1, 2, 3 (years after the meeting year) or LR; or give target_year (e.g. 2021) / "LR"
  prediction     predicted year-end fed funds rate (midpoint, percent)
Usage: python evaluate_external.py my_method.csv [--name MyMethod] [--split 2014-2020|2019-2020|all]
Output: the method added to the benchmark table on the rows it covers, with paired t / Wilcoxon tests against B0 and B14."""
import argparse, numpy as np, pandas as pd
from scipy import stats
from splits import mask
ap = argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--name', default=None); ap.add_argument('--split', default='2014-2020', help='2014-2020 | 2019-2020 | person6040 | all')
a = ap.parse_args()
name = a.name or a.csv.rsplit('/', 1)[-1].rsplit('.', 1)[0]
B = pd.read_csv('data/predictions_v6.csv', parse_dates=['meeting_date'])
B['year'] = B.meeting_date.dt.year
X = pd.read_csv(a.csv); X['meeting_date'] = pd.to_datetime(X.meeting_date)
if 'horizon' not in X and 'target_year' in X:
    X['horizon'] = [('LR' if str(t).upper() == 'LR' else str(int(t) - d.year)) for t, d in zip(X.target_year, X.meeting_date)]
X['h'] = X.horizon.astype(str).str.upper().replace({'LONGER RUN': 'LR', 'LONG_RUN': 'LR'})
key = 'person_id' if X.participant.astype(str).str.match(r'^[0-9a-f-]{36}$').all() else 'participant_name'
B['_in_split'] = mask(B, {'2014-2020': 'test 2014-2020', '2019-2020': 'test 2019-2020', 'person6040': 'person 60/40', 'all': 'all'}[a.split]).values   # on ALL rows, so the split never depends on which rows a method covers
M = B.merge(X.rename(columns={'participant': key, 'prediction': name})[['meeting_date', key, 'h', name]], on=['meeting_date', key, 'h'], how='inner')
SPLIT = {'2014-2020': 'test 2014-2020', '2019-2020': 'test 2019-2020', 'person6040': 'person 60/40', 'all': 'all'}[a.split]
M = M[M._in_split].dropna(subset=[name])
print(f'{name}: {len(M)} rows matched ({M.meeting_date.nunique()} SEPs, {M[key].nunique()} participants), split {a.split}; unmatched input rows: {len(X) - len(M)} (outside split or no true dot)')
methods = [c for c in B.columns if c.startswith('B') and not c.startswith('_')] + [name]
def met(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); y, p = y[m], p[m]
    return pd.Series({'n': len(y), 'MAE': np.mean(np.abs(p - y)), 'RMSE': np.sqrt(np.mean((p - y) ** 2)), 'Corr': np.corrcoef(p, y)[0, 1] if len(y) > 2 else np.nan})
rows = []
for m in methods:
    r = met(M.federal_funds_rate.values, M[m].values).astype(object)
    for h in ['0', '1', '2', '3', 'LR']:
        s = M[M.h == h]; r[f'MAE_h{h}'] = np.mean(np.abs(s[m] - s.federal_funds_rate)) if len(s) else np.nan
    C = M.groupby(['meeting_date', 'h'])[['federal_funds_rate', m]].median(); r['com_MAE'] = np.mean(np.abs(C[m] - C.federal_funds_rate))
    for ref in ['B0_prior_own', 'B14_combo_plus_text_h0']:
        if m == ref: continue
        s = M[['federal_funds_rate', m, ref]].dropna(); e1, e0 = (s[m] - s.federal_funds_rate).abs(), (s[ref] - s.federal_funds_rate).abs()
        if len(s) > 5 and (e1 - e0).abs().sum() > 0:
            p = stats.ttest_rel(e1, e0).pvalue; w = stats.wilcoxon(e1, e0).pvalue
            r[f'vs_{ref[:3]}'] = f"{(e0.mean() - e1.mean()) / e0.mean() * 100:+.0f}% (t p={p:.3f}, W p={w:.3f})"
    rows.append(r.rename(m))
T = pd.DataFrame(rows).sort_values('MAE')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print(T[['n', 'MAE', 'MAE_h0', 'MAE_h1', 'MAE_h2', 'MAE_h3', 'MAE_hLR', 'RMSE', 'Corr', 'com_MAE', 'vs_B0_', 'vs_B14']].round(3).to_string())
T.to_csv(f'data/compare_{name}_{a.split}.csv')
