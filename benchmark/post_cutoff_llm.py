"""Experiment 4: can public pre-meeting materials help the LLM beat 'previous dots' on the post-cutoff SEPs?
L2 numbers + materials; L3 numbers + materials, anchored on the previous dots; L4 anchored, no materials."""
import json, numpy as np, pandas as pd, concurrent.futures as cf
from scipy import stats
import llm
D = pd.read_csv('data/post_cutoff_dots.csv'); D['meeting_date'] = pd.to_datetime(D.meeting_date)
D['h'] = [('LR' if t == 'Longer run' else str(int(t) - d.year)) for t, d in zip(D.target, D.meeting_date)]
macro = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')
MAT = json.load(open('data/materials.json'))
seps = sorted(D.meeting_date.unique()); TEST = seps[1:]
def dots(t, g): return np.sort(D[(D.meeting_date == t) & (D.target == g)].rate.values)
def w1(a, b): q = np.linspace(0, 1, 101); return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))
SYS = 'You forecast the FOMC Summary of Economic Projections dot plot. Use only the information given; do not use knowledge of later events. Reply in JSON.'
def prompt(t, materials, anchored):
    m = macro.loc[t]; p = seps[seps.index(t) - 1]; tg = list(D[D.meeting_date == t].target.unique())
    prev = {x: dots(p, x) for x in D[D.meeting_date == p].target.unique()}; mt = MAT[str(t.date())]
    L = [f'FOMC meeting starting {t:%B %d, %Y}. Forecast the dot plot the participants will submit.', '', 'Data available the day before the meeting (as first published):',
         f'- Target range {m.target_lower:.2f}–{m.target_upper:.2f}%; effective fed funds {m.fedfunds:.2f}% ({pd.Timestamp(m.fedfunds_period):%B %Y})',
         f'- Unemployment {m.unrate:.1f}% ({pd.Timestamp(m.unrate_period):%B %Y}); PCE inflation {m.inflation:.1f}%, core PCE {m.core_inflation:.1f}% (12-month, {pd.Timestamp(m.pce_period):%B %Y}); real GDP growth {m.gdp_growth:.1f}% (4-quarter)',
         f'- Previous SEP ({p:%B %Y}) dots, sorted, per target: ' + '; '.join(f'{k}: ' + ', '.join(f'{x:.3f}' for x in v) for k, v in prev.items())]
    if materials:
        L += ['', f'FOMC statement after the previous meeting ({mt["prev_meeting"]}):', mt['statement'][:4000],
              '', f'Minutes of the previous meeting (policy discussion):', mt['minutes'][:24000],
              '', f'Beige Book published {mt["beige_published"]} (summary):', mt['beige'][:9000]]
    L += ['']
    if anchored:
        L += ['Start from the previous SEP dots. Participants revise their dots only when new information warrants it; most dots often stay the same or move by 25 basis points.',
              'Decide, using the information above, how each part of the distribution is likely to shift, and write the revised dots.']
    L += [f'Predict, for each of {tg}, all {len(dots(t, tg[0]))} participants\' year-end midpoints (multiples of 0.125, sorted) and the median. For a target year that was not in the previous SEP, extend from the last year.',
          'Return JSON {"reasoning": "<four sentences>", "dots": {"<target>": [..]}, "median": {"<target>": x}}.']
    return '\n'.join(L)
import os
MODEL = os.environ.get('LLM_MODEL', 'gpt-4.1-mini')
ARMS = {'L2 LLM + materials': (True, False), 'L3 LLM + materials, anchored': (True, True), 'L4 LLM anchored, no materials': (False, True)}
if MODEL != 'gpt-4.1-mini':
    ARMS = {f'{k} [{MODEL}]': v for k, v in {'L1 LLM numbers only': (False, False), **ARMS}.items()}
R = pd.read_csv('data/post_cutoff_results.csv', parse_dates=['meeting_date'])
R = R[R.method.str[:1].isin(['S', 'L']) & ~R.method.isin(ARMS)]
new = []
def run(job):
    arm, t = job; mat, anc = ARMS[arm]
    try: return arm, t, llm.chat(SYS, prompt(t, mat, anc), max_tokens=1800, model=MODEL)
    except Exception as e:
        print('ERROR', arm, t.date(), str(e)[:200]); return arm, t, {'error': str(e)}
with cf.ThreadPoolExecutor(int(os.environ.get('LLM_WORKERS', 4))) as ex:
    for arm, t, a in ex.map(run, [(arm, t) for arm in ARMS for t in TEST]):
        for g in D[D.meeting_date == t].target.unique():
            try: v = np.sort(np.array(a['dots'][g], dtype=float)); md = float(a['median'][g])
            except Exception: continue
            h = D[(D.meeting_date == t) & (D.target == g)].h.iloc[0]; tr = dots(t, g)
            new.append({'meeting_date': t, 'target': g, 'h': h, 'method': arm, 'median_pred': md, 'true_median': float(np.median(tr)),
                        'median_err': abs(md - np.median(tr)), 'dist_W1': w1(v, tr)})
R = pd.concat([R, pd.DataFrame(new)], ignore_index=True); R.to_csv('data/post_cutoff_results.csv', index=False)
print('usage', llm.USAGE)
T = R.pivot_table(index='method', columns='h', values='median_err', aggfunc='mean').round(3)
T['median MAE'] = R.groupby('method').median_err.mean().round(3); T['dist W1'] = R.groupby('method').dist_W1.mean().round(3); T['n'] = R.groupby('method').size()
pd.set_option('display.width', 200); print(T.sort_values('median MAE').to_string())
ref = R[R.method == 'S1 previous distribution'].set_index(['meeting_date', 'target'])
for m in [x for x in R.method.unique() if x.startswith('L')]:
    a = R[R.method == m].set_index(['meeting_date', 'target']).join(ref, how='inner', rsuffix='_ref')
    print(f'{m:32s} vs S1: median {a.median_err.mean():.3f} vs {a.median_err_ref.mean():.3f} (Wilcoxon p={stats.wilcoxon(a.median_err, a.median_err_ref).pvalue:.3f}); '
          f'W1 {a.dist_W1.mean():.3f} vs {a.dist_W1_ref.mean():.3f} (p={stats.wilcoxon(a.dist_W1, a.dist_W1_ref).pvalue:.3f}); n={len(a)}')
