"""Policy decision and vote benchmark: AI arms (Azure OpenAI; costs money).
  anon   : data published before the meeting (real-time macro data, current target, last four decisions with dissent counts,
           each voter's last four votes); no date, no names
  named  : the same plus the meeting date and the voters' names
  anon_market : anon plus the market expectation (4-week Treasury bill, see policy_data.py)
  anon_dir : anon, with the direction of each past dissent (wanted tighter / easier policy) in the vote histories
  memory : the meeting date only, no data ("what did the FOMC decide?"); measures what the model remembers
For anon and named the model gives the committee decision and target range, and each voter's preferred action and range,
as the group's simulation does. policy_bench.py scores them with the group's rule (vote "for" when the preferred action
equals the actual decision) and ex ante (vote "for" when it equals the model's own committee decision).
Writes data/policy_llm_<arm>_<model>.csv; raw answers are cached in data/policy_llm_cache.jsonl.
Usage: LLM_MODEL=gpt-5-mini python policy_llm.py [anon named memory anon_market anon_dir]"""
import hashlib, json, os, sys, threading, concurrent.futures as cf
import pandas as pd
import db, llm

MODEL = os.environ.get('LLM_MODEL', 'gpt-5-mini')
ARMS = [a for a in sys.argv[1:] if a in ('anon', 'named', 'memory', 'anon_market', 'anon_dir')] or ['anon', 'named', 'memory']
M = pd.read_csv('data/policy_meetings.csv').sort_values('meeting_date').reset_index(drop=True)
V = pd.read_csv('data/policy_votes.csv')
mac = pd.read_sql('SELECT * FROM fedai_corpus.meeting_macro_data', db.connect()).assign(meeting_date=lambda d: d.meeting_date.astype(str)).set_index('meeting_date')
stated = M.set_index('meeting_date').stated_voters.dropna()

def target_text(t):
    m = mac.loc[t]
    return f'{m.target_lower:.2f}%' if m.target_lower == m.target_upper else f'{m.target_lower:.2f}-{m.target_upper:.2f}%'
def bp(x): return 'no change' if x == 0 else f'{"+" if x > 0 else ""}{round(x * 100)} bp'

def prompt(i, arm):
    r = M.iloc[i]; t = r.meeting_date; m = mac.loc[t]
    named = arm == 'named'
    L = [f'FOMC meeting of {pd.Timestamp(t):%B %d, %Y}.' if named else 'An FOMC meeting (date withheld).',
         '', 'Information available the day before the meeting (data as first published):',
         f'- Federal funds target: {target_text(t)}; effective federal funds rate {m.fedfunds:.2f}%'
         + (f' ({pd.Timestamp(m.fedfunds_period):%B %Y})' if named else ''),
         f'- Unemployment rate {m.unrate:.1f}% (change from previous month {m.delta_unrate:+.1f})'
         + (f' ({pd.Timestamp(m.unrate_period):%B %Y})' if named else ''),
         f'- PCE inflation {m.inflation:.1f}%, core PCE inflation {m.core_inflation:.1f}% (12-month)'
         + (f' ({pd.Timestamp(m.pce_period):%B %Y})' if named else ''),
         f'- Real GDP growth {m.gdp_growth:.1f}% (4-quarter)' + (f' ({pd.Timestamp(m.gdp_period):%Y}Q{(pd.Timestamp(m.gdp_period).month - 1) // 3 + 1})' if named else ''),
         '', 'Previous four FOMC decisions (oldest first):']
    if arm == 'anon_market':
        L.insert(len(L) - 2, f'- Market: the 4-week Treasury bill yield, net of its usual gap to the target, implies an average target of {r.market_implied_after:.2f}% over the next four weeks')
    for j in range(max(0, i - 4), i):
        p = M.iloc[j]
        L.append(f'- {pd.Timestamp(p.meeting_date):%B %Y}: ' if named else f'- {i - j} meeting(s) earlier: ')
        L[-1] += f'{bp(p.delta)}, {int(p.n_against)} dissent(s)'
        if arm == 'anon_dir' and p.n_against:
            dv = V[(V.meeting_date == p.meeting_date) & (V.vote == 'against')].direction.value_counts()
            L[-1] += ' (' + ', '.join(f'{n} wanted {d} policy' for d, n in dv.items()) + ')'
    if i == 0: L.append('- (none on record)')
    voters = V[V.meeting_date == t].sort_values('member_key')
    full = not (t in stated.index and len(voters) < stated[t])
    ids = {}
    if full:
        L += ['', 'Voting members this meeting, with their last four votes on record (oldest first):']
        for k, (_, v) in enumerate(voters.iterrows(), 1):
            H = V[(V.member_key == v.member_key) & (V.meeting_date < t)].sort_values('meeting_date').tail(4)
            hist = [f'against (wanted {d} policy)' if arm == 'anon_dir' and x == 'against' else x for x, d in zip(H.vote, H.direction)]
            vid = v.member_name if named else f'Voter {k}'
            ids[vid] = v.member_key
            L.append(f'- {vid}: ' + (', '.join(hist) if hist else 'no earlier vote on record'))
    else:
        L += ['', f'{int(stated[t])} voting members (names not yet published).']
    L += ['', 'Predict the Committee decision on the federal funds target range and each voting member\'s preferred action'
          ' (a member who prefers a different action than the Committee decides would dissent).',
          'Return JSON {"reasoning": "<three sentences>", "committee_decision": "hike|hold|cut", "target_range_after": "<lower>-<upper>",'
          ' "members": [{"id": "<as listed>", "preferred_action": "hike|hold|cut", "preferred_target_range": "<lower>-<upper>"}]}.']
    return '\n'.join(L), ids

def memory_prompt(i):
    r = M.iloc[i]
    return (f'The FOMC meeting that ended on {pd.Timestamp(r.meeting_end_date):%B %d, %Y}. From memory only, what did the Committee decide about the'
            ' federal funds target range, and how many voting members dissented? If you do not know, give your best guess.\n'
            'Return JSON {"committee_decision": "hike|hold|cut", "target_range_after": "<lower>-<upper>", "dissents": <integer>, "confident": true|false}'), {}

SYS = {'anon': 'You simulate the Federal Open Market Committee. Use only the information given. Reply in JSON.',
       'anon_market': 'You simulate the Federal Open Market Committee. Use only the information given. Reply in JSON.',
       'anon_dir': 'You simulate the Federal Open Market Committee. Use only the information given. Reply in JSON.',
       'named': 'You simulate the Federal Open Market Committee. Use only the information given; do not use knowledge of what was actually decided. Reply in JSON.',
       'memory': 'You answer questions about Federal Reserve history. Reply in JSON.'}
CACHE = 'data/policy_llm_cache.jsonl'; lock = threading.Lock()
cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE):
        o = json.loads(l); cache[o['id']] = o['answer']

def cache_key(i, arm, user):
    return f'{M.iloc[i].meeting_date}|{arm}|{MODEL}|' + hashlib.sha1((SYS[arm] + user).encode()).hexdigest()[:12]

def mid(s):
    try:
        a, b = [float(x) for x in str(s).replace('%', '').replace('–', '-').split('-')[:2]]; return (a + b) / 2
    except Exception:
        try: return float(str(s).replace('%', ''))
        except Exception: return None
def norm(a):
    a = str(a or '').strip().lower()
    return {'raise': 'hike', 'increase': 'hike', 'lower': 'cut', 'decrease': 'cut', 'maintain': 'hold', 'unchanged': 'hold'}.get(a, a) if a else None

def job(args):
    i, arm = args
    user, ids = memory_prompt(i) if arm == 'memory' else prompt(i, arm)
    key = cache_key(i, arm, user)                                 # the prompt's hash is in the key: changed data -> new call
    if key not in cache:
        ans = llm.chat(SYS[arm], user, max_tokens=1500, model=MODEL)
        with lock:
            cache[key] = ans
            with open(CACHE, 'a') as f: f.write(json.dumps({'id': key, 'answer': ans}) + '\n')
    return i, arm, ids, cache[key]

if __name__ == '__main__':
  for arm in ARMS:
    out = []
    with cf.ThreadPoolExecutor(int(os.environ.get('LLM_WORKERS', 12))) as ex:
        for n, (i, a, ids, ans) in enumerate(ex.map(job, [(i, arm) for i in range(len(M))]), 1):
            r = M.iloc[i]
            base = dict(arm=f'AI {a} [{MODEL}]', meeting_date=r.meeting_date, committee_decision=norm(ans.get('committee_decision')),
                        committee_after=mid(ans.get('target_range_after')), dissents=ans.get('dissents'), confident=ans.get('confident'))
            mem = {str(x.get('id')): x for x in ans.get('members', []) or []}
            if not ids: out.append({**base, 'member_key': None, 'preferred_action': None, 'preferred_after': None})
            for vid, mk in ids.items():
                x = mem.get(vid, {})
                out.append({**base, 'member_key': mk, 'preferred_action': norm(x.get('preferred_action')), 'preferred_after': mid(x.get('preferred_target_range'))})
            if n % 40 == 0: print(arm, n, llm.USAGE, flush=True)
    pd.DataFrame(out).to_csv(f'data/policy_llm_{arm}_{MODEL}.csv', index=False)
    print('done', arm, len(out), 'rows; tokens', llm.USAGE, flush=True)
