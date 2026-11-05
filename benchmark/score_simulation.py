"""Score the group's simulation output against every benchmark, in one command.

Reads the multi_agent output files as they are and scores whatever they contain:
  - current pipeline (jobs/make_predictions.py): all_results*.jsonl, one record per member with narratives, projections,
    policy and vote;
  - earlier files: projections_pred*.jsonl (projections per member) and narratives_pred*.jsonl (one answer per record).
Three parts, each on exactly the rows the simulation covers:
  1. dot plot (fed funds rate) -> evaluate_external.py: all statistical baselines B0-B16 on the same rows; plus the other
     four SEP variables against "own previous projection";
  2. narrative answers -> the group's SUBSTANTIVE_ALIGNMENT judge (gpt-5.6-luna, --judge; costs about 1 call per answer)
     next to N0 "copy own previous answer", the AI arms and the other baselines (results/narrative_scores_*.csv);
  3. policy -> preferred action / target range (group rule for votes) or the member's own for/against vote, next to the
     baselines of policy_bench.py. Records that only give selected_alternative need --alternatives (meeting_date,
     alternative, action, target_after).
Run in the benchmark work directory after `run_all.sh` (it needs data/predictions_v6.csv, data/policy_*.csv).
Usage: python score_simulation.py <files or globs ...> --name NAME [--judge] [--alternatives map.csv]"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, threading, concurrent.futures as cf
import numpy as np, pandas as pd
import db

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = ['gdp_growth', 'unemployment_rate', 'pce_inflation', 'core_pce_inflation', 'federal_funds_rate']
HKEY = {'year1': '0', 'year2': '1', 'year3': '2', 'year4': '3', 'long_run': 'LR', 'lr': 'LR', 'longer_run': 'LR'}
ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+'); ap.add_argument('--name', required=True)
ap.add_argument('--judge', action='store_true', help='judge narrative answers with the SUBSTANTIVE_ALIGNMENT prompt (model calls)')
ap.add_argument('--alternatives', help='CSV meeting_date, alternative, action (hike/hold/cut), target_after (midpoint)')
a = ap.parse_args()
NAME = re.sub(r'[^A-Za-z0-9_.-]+', '_', a.name)
for need in ['data/predictions_v6.csv', 'data/projections.csv', 'data/policy_meetings.csv', 'data/policy_votes.csv']:
    if not os.path.exists(need): sys.exit(f'{need} not found: run this in the benchmark work directory after run_all.sh')

# ---------- read the records -------------------------------------------------------------------------------------------
files = sorted({f for p in a.inputs for f in (glob.glob(p, recursive=True) or [p]) if os.path.isfile(f)})
if not files: sys.exit('no input files found')
def date_of(rec, path):
    d = rec.get('meeting_date') or rec.get('current_meeting')
    if not d: m = re.findall(r'\d{4}-\d{2}-\d{2}', path); d = m[-1] if m else None
    return str(pd.Timestamp(d).date()) if d else None
def answer_of(part):
    if isinstance(part, dict) and isinstance(part.get('answer'), dict): return part['answer']
    return part if isinstance(part, dict) else {}
dots, narr, pol, n_rec = [], [], [], 0
for f in files:
    for line in open(f, encoding='utf-8'):
        if not line.strip(): continue
        r = json.loads(line); n_rec += 1
        md = date_of(r, f)
        who = dict(meeting_date=md, name=r.get('canonical_name') or r.get('participant_name') or r.get('name'), number=r.get('participant_number'))
        proj = answer_of(r.get('projections'))
        for v in VARS:
            for hk, val in (proj.get(v) or {}).items() if isinstance(proj.get(v), dict) else []:
                if str(hk).lower() in HKEY and val not in (None, ''):
                    try: dots.append({**who, 'variable': v, 'h': HKEY[str(hk).lower()], 'prediction': float(val)})
                    except ValueError: pass
        items = r.get('narratives') if isinstance(r.get('narratives'), list) else ([r] if 'question_id' in r and ('answer' in r or 'predicted_answer' in r) else [])
        for it in items:
            text = str(it.get('answer') or it.get('predicted_answer') or '').strip()     # narratives_evaluation*.jsonl: predicted_answer
            if text:
                ev = ((it.get('evaluation_results') or {}).get('substantive_alignment')) if isinstance(it.get('evaluation_results'), dict) else None
                narr.append({**who, 'question_code': str(it.get('question_id')), 'answer': text, 'own_eval': ev})
        p, v = answer_of(r.get('policy')), answer_of(r.get('vote'))
        if p or v:
            pol.append({**who, 'preferred_action': p.get('preferred_action'), 'preferred_target_range': p.get('preferred_target_range'),
                        'selected_alternative': p.get('selected_alternative'), 'own_vote': v.get('vote')})
print(f'{len(files)} files, {n_rec} records: {len(dots)} projection values, {len(narr)} narrative answers, {len(pol)} policy/vote records')

# ---------- who is who: name -> person_id (fallback: SEP participant number at that meeting) ---------------------------
STOP = {'chair', 'chairman', 'vice', 'governor', 'president', 'mr', 'ms', 'mrs', 'dr'}
def key(n):
    t = [x for x in re.sub(r'[^a-z ]', ' ', str(n or '').lower()).split() if len(x) > 1 and x not in STOP]
    return f'{t[0]} {t[-1]}' if len(t) >= 2 else (t[0] if t else '')
c = db.connect(); cur = c.cursor()
cur.execute('SELECT person_id::text, canonical_name, display_name FROM fedai_corpus.persons')
by_key = {}
for pid, cn, dn in cur.fetchall():
    for n in {cn, dn}: by_key.setdefault(key(n), set()).add(pid)
by_last = {}
for k, ids in by_key.items(): by_last.setdefault(k.split()[-1] if k else '', set()).update(ids)
cur.execute("""SELECT b.source_meeting_start::text, sp.participant_number, sp.person_id::text FROM fedai_corpus.sep_participants sp
                 JOIN fedai_corpus.sep_batches b USING (batch_id) WHERE sp.person_id IS NOT NULL""")
by_number = {(d, int(n)): pid for d, n, pid in cur.fetchall()}
c.close()
ALIAS = {'powell': 'jerome powell', 'yellen': 'janet yellen', 'bernanke': 'ben bernanke'}       # "Chair Powell" etc.
def person(md, name, number):
    k = key(name); k = ALIAS.get(k, k)
    ids = by_key.get(k) or (by_last.get(k.split()[-1]) if k and len(by_last.get(k.split()[-1], ())) == 1 else None)
    if ids and len(ids) == 1: return next(iter(ids))
    try: return by_number.get((md, int(number)))
    except (TypeError, ValueError): return None
for rows in (dots, narr, pol):
    for x in rows: x['person_id'] = person(x['meeting_date'], x['name'], x['number'])
for label, rows in (('projections', dots), ('narratives', narr), ('policy', pol)):
    bad = {(x['meeting_date'], x['name']) for x in rows if not x['person_id']}
    if bad: print(f'  {label}: {len(bad)} member-meetings not matched to a person, left out, e.g. {sorted(bad)[:3]}')
out = {}

# ---------- 1. dot plot ------------------------------------------------------------------------------------------------
D = pd.DataFrame(dots).dropna(subset=['person_id']) if dots else pd.DataFrame()
if len(D):
    ff = D[D.variable == 'federal_funds_rate'].rename(columns={'person_id': 'participant', 'h': 'horizon'})
    ff = ff.drop_duplicates(['meeting_date', 'participant', 'horizon'])
    ff[['meeting_date', 'participant', 'horizon', 'prediction']].to_csv(f'data/sim_{NAME}_dots.csv', index=False)
    print(f'\n=== 1. Dot plot, fed funds rate: {NAME} against the statistical baselines on the same rows')
    subprocess.run([sys.executable, os.path.join(HERE, 'evaluate_external.py'), f'data/sim_{NAME}_dots.csv', '--name', f'SIM_{NAME}', '--split', 'all'], check=True)
    # by calendar year: 2020 (zero lower bound, everyone at 0.125) is easy and can hide a weak year; committee level = median dot
    Pb = pd.read_csv('data/predictions_v6.csv', parse_dates=['meeting_date']); Pb['h'] = Pb.h.astype(str)
    Mb = Pb.merge(ff.assign(meeting_date=pd.to_datetime(ff.meeting_date)).rename(columns={'participant': 'person_id', 'horizon': 'h', 'prediction': 'sim'})[['meeting_date', 'person_id', 'h', 'sim']],
                  on=['meeting_date', 'person_id', 'h'])
    REFS = {'B0_prior_own': 'B0 own last', 'B1b_no_change': 'B1b no change', 'B5_prior_plus_policy_change': 'B5', 'B14_combo_plus_text_h0': 'B14'}
    yrows = []
    for yr, g in Mb.groupby(Mb.meeting_date.dt.year):
        r = {'year': yr, 'n': len(g), 'simulation': np.mean(np.abs(g.sim - g.federal_funds_rate))}
        r.update({lab: np.mean(np.abs(g[c] - g.federal_funds_rate)) for c, lab in REFS.items()})
        C = g.groupby(['meeting_date', 'h'])[['federal_funds_rate', 'sim', 'B5_prior_plus_policy_change']].median()
        r['committee median: simulation'] = (C.sim - C.federal_funds_rate).abs().mean(); r['committee median: B5'] = (C.B5_prior_plus_policy_change - C.federal_funds_rate).abs().mean()
        yrows.append(r)
    Y = pd.DataFrame(yrows); Y.to_csv(f'data/sim_{NAME}_dots_by_year.csv', index=False)
    print('\nBy year (MAE; committee median = median of the dots):'); print(Y.round(3).to_string(index=False))
    # other variables: against each member's own previous projection (same target year), the dot-plot B0 analogue
    P = pd.read_csv('data/projections.csv', parse_dates=['meeting_date'])
    P['h'] = np.where(P.is_long_run, 'LR', (P.projection_year - P.meeting_date.dt.year).astype('Int64').astype(str))
    rows = []
    T = P.melt(id_vars=['meeting_date', 'person_id', 'h', 'projection_year', 'is_long_run'], value_vars=VARS, var_name='variable', value_name='truth').dropna(subset=['truth'])
    T = T.sort_values('meeting_date')
    T['tkey'] = np.where(T.is_long_run, 'LR', T.projection_year.astype('Int64').astype(str))
    T['prev_own'] = T.groupby(['person_id', 'variable', 'tkey']).truth.shift(1)
    S = D.assign(meeting_date=pd.to_datetime(D.meeting_date)).merge(T, on=['meeting_date', 'person_id', 'variable', 'h'])
    for v, g in S.dropna(subset=['prev_own']).groupby('variable'):
        e1, e0 = (g.prediction - g.truth).abs(), (g.prev_own - g.truth).abs()
        rows.append(dict(variable=v, n=len(g), MAE_simulation=e1.mean(), MAE_own_previous=e0.mean(), simulation_better_by=f'{(e0.mean() - e1.mean()) / e0.mean() * 100:+.0f}%'))
    if rows:
        O = pd.DataFrame(rows); O.to_csv(f'data/sim_{NAME}_other_variables.csv', index=False); out['other_variables'] = O
        print('\nAll five SEP variables, against each member\'s own previous projection for the same year:')
        print(O.round(3).to_string(index=False))

# ---------- 2. narrative answers ---------------------------------------------------------------------------------------
Nn = pd.DataFrame(narr).dropna(subset=['person_id']) if narr else pd.DataFrame()
if len(Nn):
    if not os.path.exists('data/narratives.csv'): subprocess.run([sys.executable, os.path.join(HERE, 'narr_export.py')], check=True)
    REF = pd.read_csv('data/narratives.csv', dtype={'person_id': str})
    REF = REF[(REF.response_status == 'answered') & REF.answer.fillna('').str.strip().astype(bool)]
    REF['meeting_date'] = pd.to_datetime(REF.meeting_date).dt.strftime('%Y-%m-%d')
    REF = REF.drop_duplicates(['meeting_date', 'person_id', 'question_code'])
    X = Nn.merge(REF[['meeting_date', 'person_id', 'question_code', 'question_text', 'family', 'answer']].rename(columns={'answer': 'reference'}),
                 on=['meeting_date', 'person_id', 'question_code'], how='inner').drop_duplicates(['meeting_date', 'person_id', 'question_code'])
    X['key'] = X.meeting_date + '|' + X.person_id + '|' + X.question_code
    print(f'\n=== 2. Narrative answers: {len(X)} of {len(Nn)} matched to a real answer (same meeting, person and question code)')
    sys.path.insert(0, HERE); import narr_prompts, llm
    POS = {'full_match': 1.0, 'partial_match': 0.5, 'mismatch': 0.0}; COV = {'covered': 1.0, 'partially_covered': 0.5, 'not_covered': 0.0}; CON = {'none': 1.0, 'minor': 0.5, 'major': 0.0}
    def final(e):
        if not isinstance(e, dict): return None
        p = POS.get(e.get('position_match')); cn = CON.get((e.get('contradiction') or {}).get('label'))
        cov = [COV[x.get('coverage')] for x in e.get('coverage_labels') or [] if x.get('coverage') in COV]
        return 0.4 * p + 0.4 * float(np.mean(cov)) + 0.2 * cn if p is not None and cn is not None and cov else None
    JM = os.environ.get('JUDGE_MODEL', 'gpt-5.6-luna'); cache_f = 'data/sim_judge.jsonl'; lock = threading.Lock()
    cache = {}
    if os.path.exists(cache_f):
        for line in open(cache_f):
            try: o = json.loads(line); cache[o['id']] = o['eval']
            except json.JSONDecodeError: pass
    X['jid'] = [f'{k}|{hashlib.sha1(t.encode()).hexdigest()[:12]}|{JM}' for k, t in zip(X.key, X.answer)]
    todo = X[~X.jid.isin(cache)]
    if len(todo) and a.judge:
        def one(r):
            p = narr_prompts.SUBSTANTIVE_ALIGNMENT.format(question=r.question_text, reference_answer=r.reference, predicted_answer=r.answer)
            try: e = llm.chat(narr_prompts.SYSTEM_PROMPT, p, max_tokens=900, model=JM)
            except Exception as ex: print('judge error', r.key, str(ex)[:100]); return
            with lock:
                cache[r.jid] = e
                with open(cache_f, 'a') as fh: fh.write(json.dumps({'id': r.jid, 'eval': e}, ensure_ascii=False) + '\n')
        with cf.ThreadPoolExecutor(int(os.environ.get('LLM_WORKERS', 8))) as ex: list(ex.map(one, [r for r in todo.itertuples()]))
    elif len(todo):
        print(f'  {len(todo)} answers not judged yet: add --judge (about {len(todo)} model calls)')
    X['final'] = X.jid.map(lambda j: final(cache.get(j)))
    X['own_judge_final'] = X.own_eval.map(final)
    # baseline scores: the published results folder if present, else this work directory's narr_report.py output
    files = sorted(glob.glob(os.path.join(HERE, '../results/narrative_scores_*.csv'))) or sorted(glob.glob('data/narr_scores_*.csv'))
    if not files: sys.exit('narrative baselines not found: run run_all.sh --with-llm first (it writes data/narr_scores_<period>.csv)')
    B = pd.concat([pd.read_csv(f, dtype={'person_id': str}) for f in files])
    W = B.pivot_table(index='key', columns='method', values='final')
    J = X.dropna(subset=['final']).set_index('key')[['final', 'own_judge_final', 'family']].join(W, how='inner')
    if len(J):
        cols = ['final'] + [m for m in ['N0', 'N1', 'L_ctx', 'L_prev', 'L_named', 'ORACLE'] if m in J]
        both = J.dropna(subset=cols)
        print(f'Scored with the group\'s SUBSTANTIVE_ALIGNMENT prompt ({JM}); {len(both)} answers that every baseline also has:')
        T = both[cols].mean().rename({'final': f'SIM {NAME}'}).to_frame('final score')
        T['note'] = pd.Series({f'SIM {NAME}': 'your simulation', 'N0': 'copy own previous answer', 'N1': 'committee answer, previous SEP', 'L_ctx': 'AI, material only',
                               'L_prev': 'AI, material + own previous answer', 'L_named': 'AI, + name and date', 'ORACLE': 'peers, same meeting (ex post)'})
        print(T.round(3).to_string())
        from scipy import stats
        for ref in ['N0', 'L_prev']:
            if ref in both:
                d = both['final'] - both[ref]; print(f'  simulation - {ref}: {d.mean():+.3f} (t p={stats.ttest_1samp(d, 0).pvalue:.3f}, Wilcoxon p={stats.wilcoxon(d).pvalue:.3f})')
        # longer answers cover more key points: same length adjustment as narr_length.py (one fixed effect per real answer,
        # control for log(answer length / real answer length), standard errors clustered by answer)
        import statsmodels.formula.api as smf
        X['pred_words'] = X.answer.str.split().str.len(); X['ref_words'] = X.reference.str.split().str.len()
        sim_rows = X[X.key.isin(both.index)][['key', 'final', 'pred_words', 'ref_words']].assign(method='SIM')
        ratio = (sim_rows.pred_words / sim_rows.ref_words).median()
        print(f'  length: simulation answers {sim_rows.pred_words.mean():.0f} words on average, real answers {sim_rows.ref_words.mean():.0f} (median ratio {ratio:.1f}x)')
        for ref in ['N0', 'L_prev']:
            if ref not in both: continue
            Dd = pd.concat([sim_rows, B[B.key.isin(both.index) & (B.method == ref)][['key', 'final', 'pred_words', 'ref_words', 'method']]])
            Dd['x'] = (Dd.method == 'SIM').astype(int); Dd['lr'] = np.log(Dd.pred_words / Dd.ref_words)
            fit = smf.ols('final ~ x + lr + C(key)', Dd).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Dd.key)[0]})
            print(f'  simulation - {ref}, length-adjusted: {fit.params["x"]:+.3f} (p={fit.pvalues["x"]:.3f})')
        print('By question:'); print(both.groupby('family')[cols].mean().round(3).rename(columns={'final': 'SIM'}).to_string())
        if J.own_judge_final.notna().any():
            o = J.dropna(subset=['own_judge_final'])
            print(f'  the files also carry the group\'s own judge results: mean {o.own_judge_final.mean():.3f} on {len(o)} answers '
                  f'(ours {o.final.mean():.3f}; correlation {o.own_judge_final.corr(o.final):.2f})')
        both.reset_index().to_csv(f'data/sim_{NAME}_narratives.csv', index=False); out['narratives'] = T

# ---------- 3. policy decisions and votes ------------------------------------------------------------------------------
Pp = pd.DataFrame(pol).dropna(subset=['person_id']) if pol else pd.DataFrame()
if len(Pp):
    def mid(s):
        m = re.findall(r'\d+(?:\.\d+)?', str(s or ''))          # '2.25-2.50': the hyphen is a range, not a sign
        return (float(m[0]) + float(m[1])) / 2 if len(m) >= 2 else (float(m[0]) if m else np.nan)
    ACT = {'raise': 'hike', 'increase': 'hike', 'tighten': 'hike', 'lower': 'cut', 'reduce': 'cut', 'ease': 'cut', 'maintain': 'hold', 'unchanged': 'hold', 'no change': 'hold'}
    Pp['preferred_action'] = Pp.preferred_action.map(lambda x: None if x is None or pd.isna(x) else ACT.get(str(x).strip().lower(), str(x).strip().lower()))
    Pp['preferred_after'] = Pp.preferred_target_range.map(mid)
    if a.alternatives and Pp.selected_alternative.notna().any():
        AM = pd.read_csv(a.alternatives, dtype=str); AM['alternative'] = AM.alternative.str.strip().str.upper()
        Pp['alt'] = Pp.selected_alternative.astype(str).str.strip().str.upper().str[:1]
        Pp = Pp.merge(AM.rename(columns={'alternative': 'alt', 'action': 'alt_action', 'target_after': 'alt_after'}), on=['meeting_date', 'alt'], how='left')
        Pp['preferred_action'] = Pp.preferred_action.fillna(Pp.alt_action); Pp['preferred_after'] = Pp.preferred_after.fillna(pd.to_numeric(Pp.alt_after, errors='coerce'))
    elif Pp.selected_alternative.notna().any() and Pp.preferred_action.isna().all():
        print('\n  policy: records give selected_alternative only; pass --alternatives to score the committee decision (votes are scored either way)')
    VOTE = {'for': 'for', 'voted_for': 'for', 'yes': 'for', 'support': 'for', 'against': 'against', 'voted_against': 'against', 'no': 'against', 'oppose': 'against'}
    Pp['pred_vote'] = Pp.own_vote.map(lambda x: VOTE.get(str(x).strip().lower()) if x is not None and not pd.isna(x) else None)
    Vt = pd.read_csv('data/policy_votes.csv')
    known = set(Vt.member_key)
    Pp['member_key'] = Pp.person_id.where(Pp.person_id.isin(known), 'name:' + Pp.name.astype(str).str.lower().str.strip())
    A = Pp.assign(arm=f'SIM {NAME}', committee_decision=None, committee_after=np.nan, dissents=np.nan, confident=np.nan)
    A[['arm', 'meeting_date', 'committee_decision', 'committee_after', 'dissents', 'confident', 'member_key', 'preferred_action', 'preferred_after', 'pred_vote']] \
        .to_csv(f'data/policy_llm_SIM_{NAME}.csv', index=False)
    print(f'\n=== 3. Policy: {Pp.meeting_date.nunique()} meetings, {len(Pp)} member records '
          f'({Pp.preferred_action.notna().sum()} with a preferred action, {Pp.pred_vote.notna().sum()} with an own vote)')
    open(f'data/sim_{NAME}_meetings.txt', 'w').write('\n'.join(sorted(set(Pp.meeting_date))))
    r = subprocess.run([sys.executable, os.path.join(HERE, 'policy_bench.py')], capture_output=True, text=True,
                       env={**os.environ, 'POLICY_MEETINGS': f'data/sim_{NAME}_meetings.txt'})
    os.remove(f'data/policy_llm_SIM_{NAME}.csv')                      # keep the AI-arm folder clean for later runs
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit('policy_bench.py failed')
    R = pd.read_csv('data/policy_metrics_sim.csv')
    R = R[R.method.str.startswith(('D', 'V', 'G', 'AI', f'SIM {NAME}'))]
    pd.set_option('display.width', 250)
    for lvl, g in R.groupby('level', sort=False):
        cols = ['method', 'n', 'accuracy', 'macro_f1'] + (['midpoint_mae', 'changes_called'] if lvl == 'committee' else ['balanced_accuracy', 'dissent_count_mae', 'dissents_called', 'false_alarms'])
        print(f'\n-- {lvl} level, on the meetings the simulation covers (every method on the same meetings)')
        print(g[cols].round(3).to_string(index=False))
    if os.path.exists('data/policy_tests_sim.csv'):
        Tt = pd.read_csv('data/policy_tests_sim.csv'); Tt = Tt[Tt.method.str.startswith('SIM')]
        if len(Tt): print('\nPaired tests (exact McNemar: right only by the simulation vs only by the baseline):'); print(Tt.drop(columns='period').to_string(index=False))
    R.to_csv(f'data/sim_{NAME}_policy.csv', index=False)
print(f'\nFiles: data/sim_{NAME}_*.csv, data/compare_SIM_{NAME}_all.csv')
