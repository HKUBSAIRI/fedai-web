"""Clean out-of-sample test after the LLM's training cutoff (June 2024): the 9 SEPs from Sep 2024 to Sep 2026.
Targets: committee median per horizon and the whole anonymous dot distribution. Pre-meeting information only.
Usage: python post_cutoff_eval.py [--llm]"""
import sys, json, numpy as np, pandas as pd, concurrent.futures as cf
D = pd.read_csv('data/post_cutoff_dots.csv'); D['meeting_date'] = pd.to_datetime(D.meeting_date)
D['h'] = [('LR' if t == 'Longer run' else str(int(t) - d.year)) for t, d in zip(D.target, D.meeting_date)]
macro = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')
macro['target_mid'] = (macro.target_lower + macro.target_upper) / 2
seps = sorted(D.meeting_date.unique()); TEST = seps[1:]
MAT = {'DTB3': .25, 'DTB6': .5, 'DGS1': 1, 'DGS2': 2, 'DGS3': 3, 'DGS5': 5, 'DGS10': 10}
Y = None
for s in MAT:
    f = pd.read_csv(f'data/fred2_{s}.csv', parse_dates=['observation_date']).rename(columns={'observation_date': 'date'})
    f[s] = pd.to_numeric(f[s], errors='coerce'); Y = f if Y is None else Y.merge(f, on='date', how='outer')
Y = Y.sort_values('date').set_index('date').ffill()
def market(day, h):
    row = Y[Y.index < day].iloc[-1]; m = np.array(list(MAT.values())); y = row[list(MAT)].values.astype(float)
    if h == 'LR': return (10 * y[-1] - 5 * y[-2]) / 5
    tau = max((pd.Timestamp(f'{day.year + int(h)}-12-31') - day).days / 365.25, .1); yt = m * y; e = .05
    a, b = max(tau - e, m[0]), tau + e
    return (np.interp(b, m, yt) - np.interp(a, m, yt)) / (b - a)
def dots(t, target): return np.sort(D[(D.meeting_date == t) & (D.target == target)].rate.values)
def prev_dots(t, target):
    p = seps[seps.index(t) - 1]; tg = D[D.meeting_date == p].target.unique()
    if target in tg: return dots(p, target)
    yrs = sorted(int(x) for x in tg if x != 'Longer run'); return dots(p, str(yrs[-1]))      # new far year: previous last year
def w1(a, b):   # mean absolute difference between sorted dot sets (resampled to equal size): 1-Wasserstein distance
    q = np.linspace(0, 1, 101); return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))
pred = []   # (meeting, target, h, method, median_pred, dots_pred)
for t in TEST:
    m = macro.loc[t]; p = seps[seps.index(t) - 1]; dpol = m.target_mid - macro.loc[p].target_mid
    for target in D[D.meeting_date == t].target.unique():
        h = D[(D.meeting_date == t) & (D.target == target)].h.iloc[0]; pdots = prev_dots(t, target)
        cand = {'S1 previous distribution': pdots,
                'S2 no change': np.full(len(pdots), m.target_mid) if h != 'LR' else pdots,
                'S3 previous + policy change': np.maximum(0.125, pdots + (dpol if h != 'LR' else 0)),
                'S4 market forward': np.full(len(pdots), market(t, h))}
        for k, v in cand.items(): pred.append((t, target, h, k, float(np.median(v)), v))
P = pd.DataFrame(pred, columns=['meeting_date', 'target', 'h', 'method', 'median_pred', 'dots_pred'])
if '--llm' in sys.argv:
    import llm
    SYS = 'You forecast the FOMC Summary of Economic Projections dot plot. Use only the information given. Reply in JSON.'
    def ask(t):
        m = macro.loc[t]; p = seps[seps.index(t) - 1]
        tg = list(D[D.meeting_date == t].target.unique())
        prev = {x: dots(p, x) for x in D[D.meeting_date == p].target.unique()}
        lines = [f'FOMC meeting starting {t:%B %d, %Y}. Forecast the dot plot the participants will submit.', '', 'Information available the day before the meeting (as first published):',
                 f'- Target range {m.target_lower:.2f}–{m.target_upper:.2f}%; effective fed funds {m.fedfunds:.2f}% ({pd.Timestamp(m.fedfunds_period):%B %Y})',
                 f'- Unemployment {m.unrate:.1f}% ({pd.Timestamp(m.unrate_period):%B %Y}); PCE inflation {m.inflation:.1f}%, core PCE {m.core_inflation:.1f}% (12-month, {pd.Timestamp(m.pce_period):%B %Y}); real GDP growth {m.gdp_growth:.1f}% (4-quarter)',
                 f'- Previous SEP ({p:%B %Y}) dots, sorted, per target: ' + '; '.join(f'{k}: ' + ', '.join(f'{x:.3f}' for x in v) for k, v in prev.items()), '',
                 f'Predict, for each of {tg}, all {len(dots(t, tg[0]))} participants\' year-end midpoints (multiples of 0.125, sorted) and the median.',
                 'Return JSON {"reasoning": "<three sentences>", "dots": {"<target>": [..]}, "median": {"<target>": x}}.']
        return t, llm.chat(SYS, '\n'.join(lines), max_tokens=1500)
    with cf.ThreadPoolExecutor(4) as ex:
        for t, a in ex.map(ask, TEST):
            for target in D[D.meeting_date == t].target.unique():
                h = D[(D.meeting_date == t) & (D.target == target)].h.iloc[0]
                try: v = np.sort(np.array(a['dots'][target], dtype=float)); md = float(a['median'][target])
                except Exception: continue
                P.loc[len(P)] = (t, target, h, 'L1 LLM (gpt-4.1-mini)', md, v)
    print('LLM usage', llm.USAGE)
P['true_median'] = [np.median(dots(t, g)) for t, g in zip(P.meeting_date, P.target)]
P['median_err'] = (P.median_pred - P.true_median).abs()
P['dist_W1'] = [w1(v, dots(t, g)) for t, g, v in zip(P.meeting_date, P.target, P.dots_pred)]
P.drop(columns='dots_pred').to_csv('data/post_cutoff_results.csv', index=False)
pd.set_option('display.width', 200)
T = P.pivot_table(index='method', columns='h', values='median_err', aggfunc='mean').round(3)
T['all (median MAE)'] = P.groupby('method').median_err.mean().round(3); T['dist W1'] = P.groupby('method').dist_W1.mean().round(3); T['n'] = P.groupby('method').size()
print(T.sort_values('all (median MAE)').to_string())
from scipy import stats
if '--llm' in sys.argv:
    a = P[P.method == 'L1 LLM (gpt-4.1-mini)'].set_index(['meeting_date', 'target']); 
    for ref in ['S1 previous distribution', 'S3 previous + policy change']:
        b = P[P.method == ref].set_index(['meeting_date', 'target']); j = a.join(b, lsuffix='_l', rsuffix='_r', how='inner')
        print(f'LLM vs {ref}: median err {j.median_err_l.mean():.3f} vs {j.median_err_r.mean():.3f}, Wilcoxon p={stats.wilcoxon(j.median_err_l, j.median_err_r).pvalue:.3f}; '
              f'W1 {j.dist_W1_l.mean():.3f} vs {j.dist_W1_r.mean():.3f}, Wilcoxon p={stats.wilcoxon(j.dist_W1_l, j.dist_W1_r).pvalue:.3f} (n={len(j)})')
