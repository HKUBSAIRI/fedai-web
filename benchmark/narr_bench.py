"""Narrative (text-answer) benchmark for the SEP questionnaire, scored with the group's SUBSTANTIVE_ALIGNMENT judge.
Baselines (no LLM):  N0 own previous answer to the same question; N1 committee answer at the previous SEP (medoid, TF-IDF);
                     ORACLE other participants' answers at the SAME meeting (medoid; not available before the meeting, a ceiling).
LLM answers:         L_ctx  pre-meeting material + data only; L_prev  + own previous answer;
                     L_named  + own previous answer + name, Bank and meeting date. L_ctx / L_prev carry no names.
Usage: python narr_bench.py gen|judge|score [--from 2019-01-01] [--to 2020-12-31]
       env GEN_MODEL (default gpt-5-mini), JUDGE_MODEL (default gpt-5.6-luna), LLM_WORKERS (default 8).
Cached in data/narr_gen.jsonl and data/narr_judge.jsonl, so an interrupted run resumes where it stopped."""
import argparse, json, os, threading, concurrent.futures as cf
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import llm, narr_prompts

GEN_MODEL = os.environ.get('GEN_MODEL', 'gpt-5-mini'); JUDGE_MODEL = os.environ.get('JUDGE_MODEL', 'gpt-5.6-luna')
WORKERS = int(os.environ.get('LLM_WORKERS', 8))
FAMILY_NAME = {'Q1c longer-run convergence': 'Longer-run projections', 'Q2a uncertainty': 'Uncertainty and risks', 'Q2b risk balance': 'Uncertainty and risks',
               'Q3 fed funds path': 'Policy path', 'Q3 balance sheet': 'Policy path', 'Q4a outlook': 'Forecast narratives',
               'Q4b changes since last SEP': 'Forecast narratives', 'Q4c vs Tealbook': 'Forecast narratives'}
BASELINES = ['N0', 'N1', 'ORACLE']; LLM_METHODS = ['L_ctx', 'L_prev', 'L_named']

N = pd.read_csv('data/narratives.csv', parse_dates=['meeting_date'])
N = N[(N.response_status == 'answered') & (N.family != 'other') & N.answer.str.strip().astype(bool)].copy()
N['key'] = N.meeting_date.dt.strftime('%Y-%m-%d') + '|' + N.person_id + '|' + N.question_code
N = N.drop_duplicates('key').sort_values(['meeting_date', 'family', 'participant_number']).reset_index(drop=True)
CTX = json.load(open('data/narr_context.json'))
MACRO = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')
NAMES = pd.read_csv('data/persons.csv').set_index('person_id').canonical_name.to_dict()
AFF = pd.read_csv('data/participants.csv').set_index('person_id').aff.to_dict()
SEPS = sorted(N.meeting_date.unique())

def medoid(texts):
    if len(texts) == 1: return texts[0]
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    S = (X @ X.T).toarray(); np.fill_diagonal(S, np.nan)
    return texts[int(np.nanargmax(np.nanmean(S, axis=1)))]

def baselines(r):
    own = N[(N.person_id == r.person_id) & (N.family == r.family) & (N.meeting_date < r.meeting_date)]
    prev_sep = [m for m in SEPS if m < r.meeting_date and ((N.meeting_date == m) & (N.family == r.family)).any()]
    out = {'N0': own.iloc[-1].answer if len(own) else None}
    out['N1'] = medoid(N[(N.meeting_date == prev_sep[-1]) & (N.family == r.family)].answer.tolist()) if prev_sep else None
    peers = N[(N.meeting_date == r.meeting_date) & (N.family == r.family) & (N.person_id != r.person_id)].answer.tolist()
    out['ORACLE'] = medoid(peers) if peers else None
    return out

def data_line(t):
    m = MACRO.loc[t]
    return (f'Federal funds target range {m.target_lower:.2f}-{m.target_upper:.2f}%; unemployment rate {m.unrate:.1f}%; '
            f'PCE inflation {m.inflation:.1f}% and core PCE inflation {m.core_inflation:.1f}% (12-month); real GDP growth {m.gdp_growth:.1f}% (4-quarter), '
            f'as first published by {str(m.vintage_date)[:10]}.')

def gen_prompt(r, method, prev):
    L = []
    if method == 'L_named':
        L.append(f'You are {NAMES.get(r.person_id, r.source_name)} ({AFF.get(r.person_id, "FOMC participant")}), preparing your SEP submission '
                 f'for the FOMC meeting of {r.meeting_date:%B %d, %Y}.')
    else:
        L.append('You are an FOMC participant preparing your Summary of Economic Projections (SEP) submission. Your identity is withheld on purpose.')
    L += ['Answer the questionnaire question below as this participant would, using only the material given.', '',
          f'QUESTION: {r.question_text}', '']
    if method in ('L_prev', 'L_named') and prev:
        L += ['YOUR ANSWER TO THE SAME QUESTION AT THE PREVIOUS SEP (about three months ago):', prev, '']
    L += ['LATEST PUBLISHED DATA: ' + data_line(r.meeting_date), '',
          'PRE-MEETING MATERIAL (summary of the staff Tealbook and the Beige Book):', CTX[f'{r.meeting_date:%Y-%m-%d}']['text'], '',
          'Return JSON {"answer": "<your answer in the first person, ' +
          ('similar in length and style to your previous answer' if method != 'L_ctx' and prev else 'typically 40-150 words, in the style of an SEP questionnaire response') + '>"}']
    return '\n'.join(L)

class Cache:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.d = {}
        if os.path.exists(path):
            for line in open(path):
                try: o = json.loads(line)
                except json.JSONDecodeError: continue          # a line still being written by another process
                self.d[o['id']] = o
    def add(self, o):
        with self.lock:
            self.d[o['id']] = o
            with open(self.path, 'a') as f: f.write(json.dumps(o, ensure_ascii=False) + '\n')

def rows(a):
    return N[(N.meeting_date >= a.date_from) & (N.meeting_date <= a.date_to)]

def run_pool(jobs, fn, label):
    done = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for f in cf.as_completed([ex.submit(fn, *j) for j in jobs]):
            f.result(); done += 1
            if done % 100 == 0 or done == len(jobs): print(f'{label}: {done}/{len(jobs)}  tokens {llm.USAGE}', flush=True)

def gen(a):
    C = Cache('data/narr_gen.jsonl'); jobs = []
    for r in rows(a).itertuples():
        b = baselines(r)
        for m in BASELINES:
            if f'{r.key}|{m}' not in C.d: C.add({'id': f'{r.key}|{m}', 'answer': b[m]})
        for m in LLM_METHODS:
            if f'{r.key}|{m}|{GEN_MODEL}' not in C.d: jobs.append((r, m, b['N0']))
    def one(r, m, prev):
        try: ans = llm.chat('You are a Federal Reserve policymaker. Reply in JSON.', gen_prompt(r, m, prev), max_tokens=700, model=GEN_MODEL).get('answer')
        except Exception as e: print('ERROR gen', r.key, m, e); return
        C.add({'id': f'{r.key}|{m}|{GEN_MODEL}', 'answer': ans})
    run_pool(jobs, one, f'generate ({GEN_MODEL})')

def predictions(a):
    G = Cache('data/narr_gen.jsonl').d
    for r in rows(a).itertuples():
        for m in BASELINES + LLM_METHODS:
            gid = f'{r.key}|{m}' + ('' if m in BASELINES else f'|{GEN_MODEL}')
            ans = (G.get(gid) or {}).get('answer')
            if isinstance(ans, str) and ans.strip(): yield r, m, gid, ans

def judge(a):
    J = Cache('data/narr_judge.jsonl')
    jobs = [(r, gid, ans) for r, m, gid, ans in predictions(a) if f'{gid}|{JUDGE_MODEL}' not in J.d]
    def one(r, gid, ans):
        p = narr_prompts.SUBSTANTIVE_ALIGNMENT.format(question=r.question_text, reference_answer=r.answer, predicted_answer=ans)
        try: res = llm.chat(narr_prompts.SYSTEM_PROMPT, p, max_tokens=900, model=JUDGE_MODEL)
        except Exception as e: print('ERROR judge', gid, e); return
        J.add({'id': f'{gid}|{JUDGE_MODEL}', 'eval': res})
    run_pool(jobs, one, f'judge ({JUDGE_MODEL})')

POS = {'full_match': 1.0, 'partial_match': 0.5, 'mismatch': 0.0}; COV = {'covered': 1.0, 'partially_covered': 0.5, 'not_covered': 0.0}
CON = {'none': 1.0, 'minor': 0.5, 'major': 0.0}
def score_one(e):
    """Same formula as the repo (evaluation/narrative_results.py): 0.4 position + 0.4 coverage + 0.2 no-contradiction."""
    p = POS.get(e.get('position_match')); c = CON.get((e.get('contradiction') or {}).get('label'))
    cov = [COV[x.get('coverage')] for x in e.get('coverage_labels') or [] if x.get('coverage') in COV]
    cv = float(np.mean(cov)) if cov else None
    return p, cv, c, (0.4 * p + 0.4 * cv + 0.2 * c) if None not in (p, cv, c) else None

def scored(a, judge_model=None):
    J = Cache('data/narr_judge.jsonl').d; out = []
    for r, m, gid, ans in predictions(a):
        e = (J.get(f'{gid}|{judge_model or JUDGE_MODEL}') or {}).get('eval')
        if not e: continue
        p, cv, c, f = score_one(e)
        out.append({'key': r.key, 'meeting_date': r.meeting_date, 'person_id': r.person_id, 'family': r.family, 'type': FAMILY_NAME[r.family],
                    'method': m, 'position': p, 'coverage': cv, 'no_contradiction': c, 'final': f, 'pred_words': len(ans.split()), 'ref_words': len(r.answer.split())})
    return pd.DataFrame(out)

def validate(a):
    """Judge checks: SELF (the reference itself, should score ~1), SHUFFLE (another participant's answer to a different question at
    another meeting, should score ~0), and a second judge (VALIDATE_JUDGE, default gpt-5-mini) on a random sample of the real methods."""
    J = Cache('data/narr_judge.jsonl'); R = rows(a).sample(min(120, len(rows(a))), random_state=7)
    other = N[N.meeting_date < pd.Timestamp(a.date_from)]
    second = os.environ.get('VALIDATE_JUDGE', 'gpt-5-mini'); jobs = []
    for i, r in enumerate(R.itertuples()):
        sh = other[(other.family != r.family)].sample(1, random_state=i).answer.iloc[0]
        for m, ans in (('SELF', r.answer), ('SHUFFLE', sh)):
            if f'{r.key}|{m}|{JUDGE_MODEL}' not in J.d: jobs.append((r, f'{r.key}|{m}|{JUDGE_MODEL}', ans, JUDGE_MODEL))
    preds = list(predictions(a)); rng = np.random.default_rng(7)
    for k in rng.choice(len(preds), min(300, len(preds)), replace=False):
        r, m, gid, ans = preds[k]
        if f'{gid}|{second}' not in J.d: jobs.append((r, f'{gid}|{second}', ans, second))
    def one(r, jid, ans, model):
        p = narr_prompts.SUBSTANTIVE_ALIGNMENT.format(question=r.question_text, reference_answer=r.answer, predicted_answer=ans)
        try: res = llm.chat(narr_prompts.SYSTEM_PROMPT, p, max_tokens=900, model=model)
        except Exception as e: print('ERROR validate', jid, e); return
        J.add({'id': jid, 'eval': res})
    run_pool(jobs, one, 'validate')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['gen', 'judge', 'validate', 'score'])
    ap.add_argument('--from', dest='date_from', default='2019-01-01'); ap.add_argument('--to', dest='date_to', default='2020-12-31')
    a = ap.parse_args()
    if a.stage == 'gen': gen(a)
    elif a.stage == 'judge': judge(a)
    elif a.stage == 'validate': validate(a)
    else:
        S = scored(a); print(S.groupby('method')[['final', 'position', 'coverage', 'no_contradiction', 'pred_words']].mean().round(3))
