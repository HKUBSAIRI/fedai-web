"""Policy decision and vote benchmark: simple baselines and the group's metrics (multi_agent/evaluation/evaluate_policy.py).
Committee level: decision accuracy and macro F1 over cut/hold/hike; MAE/RMSE of the target midpoint after the meeting.
Member level: vote accuracy, macro F1 over for/against, balanced accuracy, mean per-meeting accuracy, MAE of the number of dissents.
Group rule for a simulated member: vote "for" if the member's preferred action equals the actual decision, else "against".
Input: data/policy_meetings.csv, data/policy_votes.csv (policy_data.py) and, if present, AI predictions data/policy_llm_*.csv.
Writes data/policy_metrics.csv and data/policy_tests.csv. Usage: python policy_bench.py"""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import db

M = pd.read_csv('data/policy_meetings.csv').sort_values('meeting_date').reset_index(drop=True)
V = pd.read_csv('data/policy_votes.csv')
STEP = lambda x: np.round(np.asarray(x, float) / 0.25) * 0.25          # changes come in 25 bp steps
FLOOR = 0.125                                                          # midpoint of 0-0.25
def action(change): return np.where(change > 0, 'hike', np.where(change < 0, 'cut', 'hold'))

# ---- committee baselines: predicted decision and target midpoint after the meeting --------------------------------------
P = {}
P['D0 no change'] = pd.DataFrame({'pred_decision': 'hold', 'pred_after': M.target_before})
prev_delta = M.delta.shift(1).fillna(0)
P['D1 repeat last decision'] = pd.DataFrame({'pred_decision': action(prev_delta), 'pred_after': np.maximum(FLOOR, M.target_before + prev_delta)})
# Taylor (1993) with Okun's law (output gap = -2 x unemployment gap), real-time core PCE inflation and unemployment
# published before the meeting; r* = 2, u* = 4.5 fixed in advance. Partial adjustment: 14% of the gap per meeting
# (0.8 per quarter), rounded to 25 bp, not below the zero lower bound.
mac = pd.read_sql('SELECT meeting_date::text, core_inflation, unrate FROM fedai_corpus.meeting_macro_data', db.connect())
X = M.merge(mac, on='meeting_date', how='left')
taylor = 2 + X.core_inflation + 0.5 * (X.core_inflation - 2) - (X.unrate - 4.5)
chg = STEP(0.14 * (taylor - M.target_before))
P['D2 Taylor rule (inertial)'] = pd.DataFrame({'pred_decision': action(np.maximum(FLOOR, M.target_before + chg) - M.target_before),
                                               'pred_after': np.maximum(FLOOR, M.target_before + chg)})
mchg = STEP(M.market_implied_after - M.target_before)
P['D3 market (4-week T-bill)'] = pd.DataFrame({'pred_decision': action(np.maximum(FLOOR, M.target_before + mchg) - M.target_before),
                                               'pred_after': np.maximum(FLOOR, M.target_before + mchg)})
for k in P: P[k]['meeting_date'] = M.meeting_date.values

# ---- member baselines: predicted vote --------------------------------------------------------------------------------
V = V.merge(M[['meeting_date', 'decision']], on='meeting_date').sort_values(['meeting_date', 'member_key'])
incomplete = set(M[M.stated_voters.notna() & (M.n_voters < M.stated_voters)].meeting_date)     # names of the majority not out yet
VM = {}
VM['V0 everyone votes for'] = V.assign(pred_vote='for')
V['last_vote'] = V.groupby('member_key').vote.shift(1)
VM['V1 own last vote'] = V.assign(pred_vote=V.last_vote.fillna('for'))
# V2: a member who dissented last time dissents again, unless the market (D3) expects the move they wanted
V['last_dir'] = V.groupby('member_key').direction.shift(1)
d3 = V.meeting_date.map(dict(zip(M.meeting_date, P['D3 market (4-week T-bill)'].pred_decision)))
satisfied = ((V.last_dir == 'tighter') & (d3 == 'hike')) | ((V.last_dir == 'easier') & (d3 == 'cut'))
VM['V2 own last vote, unless the market expects their move'] = V.assign(pred_vote=np.where((V.last_vote == 'against') & ~satisfied, 'against', 'for'))
d1 = dict(zip(M.meeting_date, P['D1 repeat last decision'].pred_decision))
VM['G(D1) group rule, repeat last decision'] = V.assign(pred_vote=np.where(V.meeting_date.map(d1) == V.decision, 'for', 'against'))

# ---- AI predictions (policy_llm.py), if present ------------------------------------------------------------------------
MEMORY = None
for f in sorted(glob.glob('data/policy_llm_*.csv')):
    A = pd.read_csv(f); arm = A.arm.iloc[0]
    has_committee = A.committee_decision.notna().any()             # a simulation from score_simulation.py has member answers only
    C = A.drop_duplicates('meeting_date')[['meeting_date', 'committee_decision', 'committee_after']]
    if has_committee: P[f'{arm} (committee answer)'] = C.rename(columns={'committee_decision': 'pred_decision', 'committee_after': 'pred_after'})
    if A.member_key.isna().all():                                   # memory arm: committee answer only
        MEMORY = A.drop_duplicates('meeting_date'); continue
    maj = A.dropna(subset=['preferred_action']).groupby('meeting_date').agg(
        pred_decision=('preferred_action', lambda x: x.value_counts().idxmax()), pred_after=('preferred_after', 'mean')).reset_index()
    if len(maj): P[f'{arm} (majority of members)'] = maj
    W = V.merge(A[['meeting_date', 'member_key', 'preferred_action', 'committee_decision'] + (['pred_vote'] if 'pred_vote' in A else [])].rename(columns={'pred_vote': 'own_vote'}),
                on=['meeting_date', 'member_key'], how='left')
    if W.preferred_action.notna().any():
        VM[f'{arm}: group rule'] = W.assign(pred_vote=np.where(W.preferred_action.isna(), None, np.where(W.preferred_action == W.decision, 'for', 'against')))
    if has_committee:
        VM[f'{arm}: ex ante'] = W.assign(pred_vote=np.where(W.preferred_action.isna(), None, np.where(W.preferred_action == W.committee_decision, 'for', 'against')))
    if 'own_vote' in W and W.own_vote.notna().any():                # the member's own for/against vote (current multi_agent pipeline)
        VM[f'{arm}: own vote'] = W.assign(pred_vote=W.own_vote.where(W.own_vote.notna(), None))

PERIODS = {
    '2007-2020 (corpus)': M.source.eq('corpus'),
    '2019-2020 SEP meetings (group default)': M.meeting_date.between('2019-01-01', '2020-12-31') & M.is_sep_meeting.eq(True),
    '2021-2026 (public)': M.source.eq('public'),
    '2024-07 onward (after model cutoff)': M.meeting_date.ge('2024-07-01'),
}
if os.environ.get('POLICY_MEETINGS'):                                # score_simulation.py: the meetings a simulation covers
    PERIODS = {'simulation meetings': M.meeting_date.isin(open(os.environ['POLICY_MEETINGS']).read().split())}
SUF = '_sim' if os.environ.get('POLICY_MEETINGS') else ''          # keep the published tables untouched
rows = []
for per, mask in PERIODS.items():
    T = M[mask]
    for name, p in P.items():
        Z = T.merge(p, on='meeting_date', how='inner').dropna(subset=['pred_decision'])
        if not len(Z): continue
        e = Z.pred_after - Z.target_after
        rows.append(dict(period=per, level='committee', method=name, n=len(Z),
                         accuracy=accuracy_score(Z.decision, Z.pred_decision),
                         macro_f1=f1_score(Z.decision, Z.pred_decision, labels=['cut', 'hold', 'hike'], average='macro', zero_division=0),
                         midpoint_mae=e.abs().mean(), midpoint_rmse=np.sqrt((e ** 2).mean()),
                         changes_called=f"{int(((Z.decision != 'hold') & (Z.pred_decision == Z.decision)).sum())}/{int((Z.decision != 'hold').sum())}"))
    for name, w in VM.items():
        Z = w[w.meeting_date.isin(T.meeting_date) & ~w.meeting_date.isin(incomplete)].dropna(subset=['pred_vote'])
        if not len(Z): continue
        per_meeting = Z.assign(ok=Z.vote == Z.pred_vote, ra=Z.vote == 'against', pa=Z.pred_vote == 'against').groupby('meeting_date')[['ok', 'ra', 'pa']]
        g = per_meeting.agg({'ok': 'mean', 'ra': 'sum', 'pa': 'sum'})
        rows.append(dict(period=per, level='member', method=name, n=len(Z),
                         accuracy=accuracy_score(Z.vote, Z.pred_vote),
                         macro_f1=f1_score(Z.vote, Z.pred_vote, labels=['for', 'against'], average='macro', zero_division=0),
                         balanced_accuracy=balanced_accuracy_score(Z.vote, Z.pred_vote),
                         meeting_accuracy=g.ok.mean(), dissent_count_mae=(g.pa - g.ra).abs().mean(),
                         dissents_called=f"{int(((Z.vote == 'against') & (Z.pred_vote == 'against')).sum())}/{int((Z.vote == 'against').sum())}",
                         false_alarms=int(((Z.vote == 'for') & (Z.pred_vote == 'against')).sum())))
R = pd.DataFrame(rows)
R.to_csv(f'data/policy_metrics{SUF}.csv', index=False)

# ---- paired tests (exact McNemar on the meetings / votes both methods predicted) ------------------------------------
from scipy.stats import binomtest
def mcnemar(a_ok, b_ok):
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return f'{b} vs {c}, p={binomtest(b, b + c).pvalue:.3f}' if b + c else 'identical'
tests = []
for per, mask in PERIODS.items():
    T = M[mask]
    for name in [k for k in P if k.startswith(('AI', 'SIM'))]:
        for ref in ['D1 repeat last decision', 'D3 market (4-week T-bill)'] + ([name.replace('anon_dir', 'anon')] if 'anon_dir' in name else []):
            Z = T.merge(P[name], on='meeting_date').merge(P[ref], on='meeting_date', suffixes=('', '_r')).dropna(subset=['pred_decision'])
            if len(Z): tests.append(dict(period=per, method=name, vs=ref, n=len(Z), right_only_vs_ref_only=mcnemar(Z.pred_decision == Z.decision, Z.pred_decision_r == Z.decision)))
    for name in [k for k in VM if k.startswith(('AI', 'SIM'))]:
        refs = ['V0 everyone votes for', 'V1 own last vote', 'V2 own last vote, unless the market expects their move']
        if 'anon_dir' in name: refs.append(name.replace('anon_dir', 'anon'))      # does the dissent direction help?
        for ref in refs:
            Z = VM[name][['meeting_date', 'member_key', 'vote', 'pred_vote']].merge(VM[ref][['meeting_date', 'member_key', 'pred_vote']], on=['meeting_date', 'member_key'], suffixes=('', '_r'))
            Z = Z[Z.meeting_date.isin(T.meeting_date) & ~Z.meeting_date.isin(incomplete)].dropna(subset=['pred_vote'])
            if len(Z): tests.append(dict(period=per, method=name, vs=ref, n=len(Z), right_only_vs_ref_only=mcnemar(Z.pred_vote == Z.vote, Z.pred_vote_r == Z.vote)))
if tests:
    pd.DataFrame(tests).to_csv(f'data/policy_tests{SUF}.csv', index=False)
pd.set_option('display.width', 250)
for (per, lvl), g in R.groupby(['period', 'level'], sort=False):
    cols = ['method', 'n', 'accuracy', 'macro_f1'] + (['midpoint_mae', 'midpoint_rmse', 'changes_called'] if lvl == 'committee' else
                                                     ['balanced_accuracy', 'meeting_accuracy', 'dissent_count_mae', 'dissents_called', 'false_alarms'])
    print(f'\n== {per} | {lvl}'); print(g[cols].round(3).to_string(index=False))
if MEMORY is not None and not SUF:                                              # what the model recalls from the date alone
    Z = M.merge(MEMORY, on='meeting_date')
    Z['period'] = np.select([Z.meeting_date < '2024-07-01'], ['before 2024-07'], 'after model cutoff (2024-07 onward)')
    Z['ok'] = Z.committee_decision == Z.decision; Z['range_ok'] = (Z.committee_after - Z.target_after).abs() < 1e-6
    Z['dissents_ok'] = pd.to_numeric(Z.dissents, errors='coerce') == Z.n_against
    print('\nMemory only (date, no data):')
    print(Z.groupby('period').agg(n=('ok', 'size'), decision=('ok', 'mean'), target_range=('range_ok', 'mean'), dissent_count=('dissents_ok', 'mean')).round(3).to_string())
    Z.groupby('period').agg(n=('ok', 'size'), decision=('ok', 'mean'), target_range=('range_ok', 'mean'), dissent_count=('dissents_ok', 'mean')).to_csv('data/policy_memory.csv')
if tests:
    print('\nPaired tests (exact McNemar): correct only by the method vs only by the reference')
    print(pd.DataFrame(tests).to_string(index=False))
