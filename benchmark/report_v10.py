"""Summary of the tenth batch (benchmarks_v10.py) on the unified protocol, run in the frozen work directory after
evaluate_unified.py has written data/unified_metrics.csv (and data/unified_protocols.csv with --expanding):
  - overall MAE in both test samples and both protocols, RMSE, share within 25 bp, p against naive;
  - Holm correction over the three pre-registered main methods only (B30b, B32, B39), next to the p of the full table;
  - MAE by horizon and by year (2016-2020) for the new methods and the references.
Usage: python report_v10.py [--agent MAE_2016_2020 MAE_2016_2019]   writes data/v10_report.csv and data/v10_by_year.csv"""
import argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--agent', nargs=2, type=float); a = ap.parse_args()
MAIN = ['B30b_grid_median_signals', 'B32_linear_tree', 'B39_median_of_three']
REF = ['B0_prior_own', 'B22b_prior_plus_spf_change', 'B27_combo_news', 'B25_xgb_l1_news', 'B25b_xgb_l1_news_momentum',
       'B28_expert_two_step', 'B28b_combo_b25_expert']
R = pd.read_csv('data/unified_protocols.csv') if __import__('os').path.exists('data/unified_protocols.csv') else pd.read_csv('data/unified_metrics.csv')
P = pd.read_csv('data/predictions_v10.csv', parse_dates=['meeting_date']); P['h'] = P.h.astype(str)
NEW = [c for c in P.columns if c.startswith(('B30', 'B31', 'B32', 'B33', 'B39'))]
FROZEN = 'estimated on 2012-2015 only'

def holm(ps):
    ps = pd.Series(ps).dropna(); order = ps.sort_values().index; m = len(ps); out, run = {}, 0.0
    for i, k in enumerate(order): run = max(run, min(1.0, (m - i) * ps[k])); out[k] = run
    return pd.Series(out)
O = R[R.horizon == 'all']
rows = []
for m in REF + NEW:
    r = {'method': m}
    for s in ('2016-2020', '2016-2019'):
        x = O[(O.protocol == FROZEN) & (O['sample'] == s) & (O.method == m)]
        if x.empty: continue
        x = x.iloc[0]
        r.update({f'{s} MAE': x.MAE_bp, f'{s} RMSE': x.RMSE_bp, f'{s} <=25bp %': x.within_25bp, f'{s} p': x.p_vs_naive, f'{s} p_holm_all': x.p_holm})
        y = O[(O.protocol == 'expanding window') & (O['sample'] == s) & (O.method == m)]
        if not y.empty: r[f'{s} MAE expanding'] = y.MAE_bp.iloc[0]
    rows.append(r)
T = pd.DataFrame(rows).set_index('method')
for s in ('2016-2020', '2016-2019'):
    T[f'{s} p_holm_main3'] = holm(T.loc[MAIN, f'{s} p'])
T.to_csv('data/v10_report.csv')
pd.set_option('display.width', 250)
cols = [c for c in T.columns if 'MAE' in c or 'p_holm' in c or c.endswith(' p')]
print('== overall MAE (bp)' + (f'; agent (reference) {a.agent[0]} / {a.agent[1]}' if a.agent else ''))
print(T[cols].round(3).to_string())

H = R[(R.protocol == FROZEN) & (R.horizon != 'all')].pivot_table(index='method', columns=['sample', 'horizon'], values='MAE_bp')
print('\n== MAE by horizon (bp), parameters fixed after 2015'); print(H.loc[[m for m in REF + NEW if m in H.index]].round(1).to_string())

# by year, on the unified cells: own preceding SEP has a dot for the same target
own = P[['person_id', 'meeting_date']].drop_duplicates().sort_values('meeting_date')
prev = {}
for pid, g in own.groupby('person_id'):
    d = list(g.meeting_date); prev.update({(pid, x): y for x, y in zip(d[1:], d[:-1])})
dots = P.set_index(['person_id', 'meeting_date', 'proj_year']).federal_funds_rate
P['lag_eval'] = [dots.get((p, prev.get((p, t)), y), np.nan) if prev.get((p, t)) is not None else np.nan for p, t, y in zip(P.person_id, P.meeting_date, P.proj_year)]
C = P[(P.meeting_date >= '2016-01-01') & P.lag_eval.notna() & P.federal_funds_rate.notna()]
Y = pd.DataFrame({m: (C[m] - C.federal_funds_rate).abs().groupby(C.meeting_date.dt.year).mean() * 100 for m in REF + NEW if m in C}).T
Y.loc['n_cells'] = C.groupby(C.meeting_date.dt.year).size()
Y.to_csv('data/v10_by_year.csv'); print('\n== MAE by year (bp), parameters fixed after 2015'); print(Y.round(1).to_string())
