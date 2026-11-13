"""LLM baseline for the SEP projections (11.12 meeting): the agent's language model with no persona, no rules and no memory,
asked directly for one participant's projections. Same request mask as the agent's Task D: one request per participant, test
SEP (2016-2020) and target horizon, with all five variables in each request (core PCE has no longer-run value): 1,427 requests.
Versions:
  zero      zero-shot, named: "You are <name>. It is the FOMC meeting of <date>. Give your projections for <target>."
  few       few-shot, named: as zero, plus the participant's own projections for the same target at their previous k SEPs
  few_anon  few-shot, anonymous: as few, without the name and the dates (the target and the earlier meetings are described
            relative to the current meeting). few minus few_anon measures what the model remembers about the named person.
Usage (in the work directory, e.g. docs/benchmark/work/frozen_2015):
  python llm_baseline.py VERSION [--model gpt-6-luna] [--effort high] [--k 3] [--meetings 2016-03-16 ...] [--limit N]
                         [--workers 6] [--dry-run] [--mock]
  --dry-run prints the request count and two prompts; --mock answers with the participant's previous values (no API calls).
Credentials: llm.py (FEDAI_AZURE_ENV or ~/.ssh/azure_fedai.env). Answers are cached in data/llm_baseline/<version>_<model>.jsonl,
so a rerun only sends the missing requests. Outputs: data/llm_baseline/<version>_<model>.csv (all variables, long format) and
<version>_<model>_rate.csv (meeting_date, participant, horizon, prediction) for compare_agent.py and evaluate_unified.py --add."""
import argparse, json, os, threading, concurrent.futures as cf
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('version', choices=['zero', 'few', 'few_anon'])
ap.add_argument('--model', default='gpt-6-luna'); ap.add_argument('--effort', default='high', help='reasoning effort: low, medium, high')
ap.add_argument('--k', type=int, default=3); ap.add_argument('--meetings', nargs='*'); ap.add_argument('--limit', type=int)
ap.add_argument('--workers', type=int, default=6); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--mock', action='store_true')
a = ap.parse_args()
VARS = ['gdp_growth', 'unemployment_rate', 'pce_inflation', 'core_pce_inflation', 'federal_funds_rate']
DEF = {'gdp_growth': 'real GDP growth, fourth quarter over fourth quarter (percent)',
       'unemployment_rate': 'unemployment rate, fourth-quarter average (percent)',
       'pce_inflation': 'PCE inflation, fourth quarter over fourth quarter (percent)',
       'core_pce_inflation': 'core PCE inflation, fourth quarter over fourth quarter (percent)',
       'federal_funds_rate': 'appropriate federal funds rate at the end of the year, midpoint of the target range (percent)'}
DEF_LR = {'gdp_growth': 'longer-run growth rate of real GDP (percent)', 'unemployment_rate': 'longer-run unemployment rate (percent)',
          'pce_inflation': 'longer-run PCE inflation (percent)', 'federal_funds_rate': 'longer-run federal funds rate, midpoint (percent)'}
REL = {'0': 'the end of this year', '1': 'the end of next year', '2': 'the end of the year after next', '3': 'the end of the third year from now',
       'LR': 'the longer run'}
LR_NOTE = ' (the value each variable would converge to under appropriate monetary policy and in the absence of further shocks)'
SYS = 'You are a participant in the Federal Open Market Committee (FOMC) submitting your Summary of Economic Projections. Reply in JSON only.'

# ---------- request mask: every participant x test SEP x target horizon in the database (as Task D)
J = pd.read_csv('data/projections.csv', parse_dates=['meeting_date'])
J['h'] = np.where(J.is_long_run, 'LR', (J.projection_year - J.meeting_date.dt.year).astype('Int64').astype(str))
J = J[J.h.isin(['0', '1', '2', '3', 'LR'])].copy()
J['target'] = np.where(J.is_long_run, 'LR', J.projection_year.astype('Int64').astype(str))
REQ = J[(J.meeting_date >= '2016-01-01') & (J.meeting_date <= '2020-12-31')].drop_duplicates(['meeting_date', 'person_id', 'h'])
if a.meetings: REQ = REQ[REQ.meeting_date.dt.strftime('%Y-%m-%d').isin(a.meetings)]
REQ = REQ.sort_values(['meeting_date', 'participant_name', 'h']).reset_index(drop=True)
if a.limit: REQ = REQ.head(a.limit)
BY = {k: g.sort_values('meeting_date', ascending=False) for k, g in J.groupby(['person_id', 'target'])}

def keys(h): return [v for v in VARS if not (h == 'LR' and v == 'core_pce_inflation')]
def day(t): return f'{t:%B} {t.day}, {t.year}'
def shown(v, x): return round(x / 0.125) * 0.125 if v == 'federal_funds_rate' else x   # the database stores 2.125 as 2.13

def earlier(r):
    """The participant's own projections for the same target at their previous SEPs, most recent first."""
    g = BY.get((r.person_id, r.target))
    return g[g.meeting_date < r.meeting_date].head(a.k) if g is not None else J.iloc[0:0]

def prompt(r):
    anon = a.version == 'few_anon'
    target = REL[r.h] if anon else ('the longer run' if r.h == 'LR' else f'the end of {r.target}')
    head = ('You are a participant in the FOMC. You are submitting your projections at an FOMC meeting.' if anon
            else f'You are {r.participant_name}. It is the FOMC meeting of {day(r.meeting_date)}.')
    lines = [head, '']
    if a.version != 'zero':
        E = earlier(r)
        if len(E):
            lines.append(f'Your projections for {target} at your previous SEP meetings (most recent first):')
            for _, e in E.iterrows():
                when = (f'{round((r.meeting_date - e.meeting_date).days / 30.4)} months earlier' if anon else f'{e.meeting_date:%B %Y}')
                lines.append(f'- {when}: ' + ', '.join(f'{v} {shown(v, e[v]):g}' for v in keys(r.h) if pd.notna(e[v])))
        else:
            lines.append(f'You have no earlier projection for {target}.')
        lines.append('')
    D = DEF_LR if r.h == 'LR' else DEF
    lines.append(f'Give your projections for {target}{LR_NOTE if r.h == "LR" else ""}:')
    lines += [f'- {v}: {D[v]}' for v in keys(r.h)]
    lines += ['', 'Return a JSON object with exactly these keys and numeric values.']
    return '\n'.join(lines)

def mock(r):
    E = earlier(r)
    return {v: float(E[v].iloc[0]) for v in keys(r.h) if len(E) and pd.notna(E[v].iloc[0])}

tag = f'{a.version}_{a.model.replace(".", "")}' + ('_mock' if a.mock else '')
os.makedirs('data/llm_baseline', exist_ok=True); CACHE = f'data/llm_baseline/{tag}.jsonl'
done = set()
if os.path.exists(CACHE):
    for line in open(CACHE, encoding='utf-8'):
        d = json.loads(line)
        if d.get('answer') is not None: done.add((d['meeting_date'], d['person_id'], d['h']))
todo = [r for r in REQ.itertuples() if (r.meeting_date.strftime('%Y-%m-%d'), r.person_id, r.h) not in done]
print(f'{a.version}: {len(REQ)} requests, {len(REQ) - len(todo)} cached, {len(todo)} to send; model {a.model}, effort {a.effort}, k {a.k}')
if a.dry_run:
    for r in [todo[0]] + [x for x in todo if x.h == 'LR'][:1] if todo else []:
        print('\n' + '-' * 60 + '\n' + prompt(r))
    raise SystemExit

if not a.mock:
    os.environ['LLM_REASONING_EFFORT'] = a.effort
    import llm
lock = threading.Lock()
def run(r):
    rec = dict(meeting_date=r.meeting_date.strftime('%Y-%m-%d'), person_id=r.person_id, participant_name=r.participant_name,
               h=r.h, target=r.target, version=a.version, model=a.model, k=a.k)
    try:
        rec['answer'] = mock(r) if a.mock else llm.chat(SYS, prompt(r), max_tokens=400, model=a.model)
    except Exception as e:
        rec['answer'] = None; rec['error'] = str(e)[:300]
    with lock:
        with open(CACHE, 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec) + '\n')
    return rec['answer'] is not None
with cf.ThreadPoolExecutor(a.workers) as ex:
    ok = sum(ex.map(run, todo))
print(f'answered {ok} of {len(todo)}')
if not a.mock:
    u = llm.USAGE.get(a.model, [0, 0]); n = max(ok, 1)
    print(f'tokens: input {u[0]}, output {u[1]} (incl. reasoning); per request {u[0] / n:.0f} / {u[1] / n:.0f}; '
          f'projected for all 1,427 requests: {u[0] / n * 1427 / 1e6:.2f}M / {u[1] / n * 1427 / 1e6:.2f}M')

# ---------- outputs from the cache (last answer per request)
rows = {}
for line in open(CACHE, encoding='utf-8'):
    d = json.loads(line)
    if d.get('answer') is not None: rows[(d['meeting_date'], d['person_id'], d['h'])] = d
long = []
for d in rows.values():
    for v in keys(d['h']):
        x = d['answer'].get(v) if isinstance(d['answer'], dict) else None
        try: long.append(dict(meeting_date=d['meeting_date'], participant=d['person_id'], participant_name=d['participant_name'],
                              horizon=d['h'], variable=v, prediction=float(x)))
        except (TypeError, ValueError): pass
L = pd.DataFrame(long)
L.to_csv(f'data/llm_baseline/{tag}.csv', index=False)
L[L.variable == 'federal_funds_rate'].drop(columns=['variable', 'participant_name']).to_csv(f'data/llm_baseline/{tag}_rate.csv', index=False)
print(f'wrote data/llm_baseline/{tag}.csv ({len(L)} values) and {tag}_rate.csv')
