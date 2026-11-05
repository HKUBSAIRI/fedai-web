"""LLM benchmark: the model plays one FOMC participant and writes his/her fed funds rate dots, given only pre-meeting
information (real-time macro data, current target range, own previous dots, previous committee median).
Two arms with identical numbers: NAMED (real name and dates) and ANON (no name, no dates, years as 'this year', 'next year').
Usage: python llm_benchmark.py [named|anon] [start] [end]"""
import sys, json, numpy as np, pandas as pd, concurrent.futures as cf
import llm
ARM = sys.argv[1]; START = sys.argv[2] if len(sys.argv) > 2 else '2014-01-01'; END = sys.argv[3] if len(sys.argv) > 3 else '2020-12-31'
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
P = pd.read_csv('data/predictions_v5.csv', parse_dates=['meeting_date'])
D['proj_year'] = P.proj_year
macro = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')
meetings = sorted(D.meeting_date.unique())
HN = {'0': 'end of this year', '1': 'end of next year', '2': 'end of the year after next', '3': 'end of three years from now', 'LR': 'longer run'}
def mon(s, anon, t):
    d = pd.Timestamp(s)
    if not anon: return d.strftime('%B %Y')
    k = (t.year - d.year) * 12 + t.month - d.month
    return f'{k} months before the meeting'
def yr(y, t, anon):
    if y == -1: return 'longer run'
    return {0: 'this year', 1: 'next year', 2: 'in two years', 3: 'in three years'}.get(int(y) - t.year, str(y)) if anon else str(int(y))
def prompt(t, pid, rows):
    anon = ARM == 'anon'; m = macro.loc[t]; r0 = rows.iloc[0]
    i = meetings.index(t); prev = D[(D.meeting_date == meetings[i - 1])] if i else D.iloc[0:0]
    own = prev[prev.person_id == pid].sort_values('proj_year'); med = prev.groupby('proj_year').federal_funds_rate.median()
    role = ('a Federal Reserve Board governor' if 'Reserve Bank' not in str(r0.affiliation) else 'a Reserve Bank president') if anon else f'{r0.participant_name}'
    who = role + (' (the Chair)' if r0.meeting_role == 'Chair' else '') + (', a voting member this year' if r0.voter else ', not a voting member this year')
    when = 'this FOMC meeting' if anon else f'the FOMC meeting of {t:%B %d, %Y}'
    lines = [f'You are {who}. You are submitting your Summary of Economic Projections for {when}.', '',
             'Information available before the meeting (as first published, not revised):',
             f'- Federal funds target range: {m.target_lower:.2f}–{m.target_upper:.2f}% ; effective rate {m.fedfunds:.2f}% ({mon(m.fedfunds_period, anon, t)})',
             f'- Unemployment rate {m.unrate:.1f}% ({mon(m.unrate_period, anon, t)}); change over 3 months {m.delta_unrate:+.1f} pp',
             f'- PCE inflation {m.inflation:.1f}% and core PCE inflation {m.core_inflation:.1f}% (12-month, {mon(m.pce_period, anon, t)})',
             f'- Real GDP growth {m.gdp_growth:.1f}% (4-quarter, quarter starting {mon(m.gdp_period, anon, t)})']
    tp = meetings[i - 1] if i else None
    if len(own):
        lines.append(f'- Your own fed funds projections in the previous SEP ({"about three months ago" if anon else f"{tp:%B %Y}"}): ' +
                     ', '.join(f'{yr(y, tp if not anon else t, anon) if not anon else yr(y, pd.Timestamp(tp), True).replace("this year", "that year")}: {v:.3f}%' for y, v in zip(own.proj_year, own.federal_funds_rate)))
    if len(med):
        lines.append('- Committee median in the previous SEP: ' + ', '.join(f'{yr(y, pd.Timestamp(tp), anon) if not anon else yr(y, pd.Timestamp(tp), True).replace("this year", "that year")}: {v:.3f}%' for y, v in med.items()))
    keys = [h for h in ['0', '1', '2', '3', 'LR'] if h in set(rows.h)]
    labels = {h: (HN[h] if anon else (f'end of {t.year + int(h)}' if h != 'LR' else 'longer run')) for h in keys}
    lines += ['', 'Write your appropriate year-end midpoint of the target range for the federal funds rate, in your own judgment and consistent with your past views.',
              'Return JSON {"reasoning": "<two sentences>", "projections": {' + ', '.join(f'"{h}": <number>' for h in keys) + '}} where ' +
              '; '.join(f'"{h}" = {labels[h]}' for h in keys) + '. Use multiples of 0.125.']
    return '\n'.join(lines), keys
SYS = 'You simulate a specific FOMC participant writing the federal funds rate path in the Summary of Economic Projections. Use only the information given. Reply in JSON.'
T = D[(D.meeting_date >= START) & (D.meeting_date <= END)]
jobs = [(t, pid, g) for (t, pid), g in T.groupby(['meeting_date', 'person_id'])]
def run(job):
    t, pid, g = job; q, keys = prompt(t, pid, g)
    try: a = llm.chat(SYS, q, max_tokens=300)
    except Exception as e: return []
    pr = a.get('projections', {}) if isinstance(a, dict) else {}
    out = []
    for h in keys:
        try: out.append({'meeting_date': t.strftime('%Y-%m-%d'), 'participant': pid, 'horizon': h, 'prediction': float(pr[h])})
        except Exception: pass
    return out
rows = []
with cf.ThreadPoolExecutor(6) as ex:
    for o in ex.map(run, jobs): rows += o
pd.DataFrame(rows).to_csv(f'data/llm_{ARM}.csv' if llm.MODEL == 'gpt-4.1-mini' else f'data/llm_{ARM}_{llm.MODEL.replace(".", "")}.csv', index=False)
print(ARM, 'person-SEPs', len(jobs), 'predictions', len(rows), 'usage', llm.USAGE)
if len(sys.argv) > 4: print(prompt(*jobs[0])[0])
