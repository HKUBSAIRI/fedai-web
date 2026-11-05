"""Two-step LLM design vs direct per-person prediction (anonymized, 2014-2020).
Step 1 (LLM, one call per SEP): forecast how the committee median moves for each horizon, given only pre-meeting data and the
previous SEP's anonymous dot distribution (no names, no dates, years as 'this year' / 'next year').
Step 2 (mechanical): each participant = own previous dot + predicted median change [+ herding a*(prev median - own), a from B16].
Usage: LLM_MODEL=<deployment> python twostep.py"""
import os, json, numpy as np, pandas as pd, concurrent.futures as cf
import llm
MODEL = os.environ.get('LLM_MODEL', 'gpt-4.1-mini')
P = pd.read_csv('data/predictions_v6.csv', parse_dates=['meeting_date'])
macro = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')
meetings = sorted(P.meeting_date.unique()); TEST = [m for m in meetings if m >= pd.Timestamp('2014-01-01')]
REL = {0: 'this year', 1: 'next year', 2: 'in two years', 3: 'in three years', -1: 'longer run'}
def rel(y, t): return REL[-1] if y == -1 else REL.get(int(y) - t.year, f'{int(y) - t.year} years ahead')
def prompt(t):
    i = meetings.index(t); p = meetings[i - 1]; m = macro.loc[t]; mp = macro.loc[p]
    prev = P[P.meeting_date == p]
    dist = {y: np.sort(g.federal_funds_rate.values) for y, g in prev.groupby('proj_year')}
    cur_years = sorted(P[P.meeting_date == t].proj_year.unique(), key=lambda y: (y == -1, y))
    L = ['Forecast how the FOMC dot plot median will change at this meeting. No names or dates are given on purpose.', '',
         'Data available the day before this meeting (as first published), with the value at the previous SEP (about three months earlier) in brackets:',
         f'- Federal funds target range {m.target_lower:.2f}-{m.target_upper:.2f}% [{mp.target_lower:.2f}-{mp.target_upper:.2f}%]',
         f'- Unemployment rate {m.unrate:.1f}% [{mp.unrate:.1f}%]; core PCE inflation {m.core_inflation:.1f}% [{mp.core_inflation:.1f}%]; PCE inflation {m.inflation:.1f}% [{mp.inflation:.1f}%]; real GDP growth {m.gdp_growth:.1f}% [{mp.gdp_growth:.1f}%]',
         '- Previous SEP dots (anonymous, sorted), by target year as seen from the previous meeting:']
    for y, v in dist.items():
        L.append(f'  {rel(y, p).replace("this year", "that year")}: ' + ', '.join(f'{x:.3f}' for x in v) + f' (median {np.median(v):.3f})')
    keys = [rel(y, t) for y in cur_years]
    L += ['', 'For each target below, give the new median year-end federal funds rate (midpoint, multiples of 0.125). "this year" here is the year of the current meeting; if the previous meeting was in the previous calendar year, shift targets accordingly.',
          'Return JSON {"reasoning": "<three sentences>", "median": {' + ', '.join(f'"{k}": x' for k in keys) + '}}']
    return '\n'.join(L), dict(zip(keys, cur_years))
def run(t):
    q, keymap = prompt(t)
    try: a = llm.chat('You forecast FOMC dot plots. Use only the information given. Reply in JSON.', q, max_tokens=500, model=MODEL)
    except Exception as e: print('ERROR', t.date(), e); return t, {}
    out = {}
    for k, y in keymap.items():
        try: out[y] = float(a['median'][k])
        except Exception: pass
    return t, out
pred_med = {}
with cf.ThreadPoolExecutor(int(os.environ.get('LLM_WORKERS', 4))) as ex:
    for t, out in ex.map(run, TEST): pred_med[t] = out
rows_ts, rows_th = [], []
alpha = {'0': 0.37, '1': 0.22, '2': 0.09, '3': 0.09, 'LR': 0.06}     # B16 average estimates (expanding window), see benchmarks_v6.py
for t in TEST:
    cur = P[P.meeting_date == t]
    for r in cur.itertuples():
        if r.proj_year not in pred_med[t]: continue
        dmed = pred_med[t][r.proj_year] - r.B1a_prev_median                 # predicted change of the committee median for this target
        base = r.B0_prior_own + dmed
        rows_ts.append({'meeting_date': t.strftime('%Y-%m-%d'), 'participant': r.person_id, 'horizon': r.h, 'prediction': max(0.125, base)})
        rows_th.append({'meeting_date': t.strftime('%Y-%m-%d'), 'participant': r.person_id, 'horizon': r.h, 'prediction': max(0.125, base + alpha[r.h] * (r.B1a_prev_median - r.B0_prior_own))})
tag = MODEL.replace('.', '')
pd.DataFrame(rows_ts).to_csv(f'data/twostep_{tag}.csv', index=False); pd.DataFrame(rows_th).to_csv(f'data/twostep_herd_{tag}.csv', index=False)
# committee-level check: predicted median vs true median, against "median unchanged"
cm = []
for t in TEST:
    true = P[P.meeting_date == t].groupby('proj_year').federal_funds_rate.median(); prevm = P[P.meeting_date == t].groupby('proj_year').B1a_prev_median.first()
    for y, v in pred_med[t].items():
        if y in true: cm.append((abs(v - true[y]), abs(prevm[y] - true[y]), np.sign(v - prevm[y]), np.sign(true[y] - prevm[y])))
C = np.array(cm)
moved = C[:, 2] != 0
print(f'{MODEL}: committee median MAE {C[:, 0].mean():.3f} vs unchanged {C[:, 1].mean():.3f} (n={len(C)}); LLM moved the median in {moved.sum()} cells, right direction {np.mean(C[moved, 2] == C[moved, 3]):.0%}')
print('usage', {k: v for k, v in llm.USAGE.items() if k not in ('in', 'out')})
