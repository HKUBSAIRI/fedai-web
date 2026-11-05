"""Policy decision and vote benchmark: data.
Ground truth comes from FRED's daily federal funds target (DFEDTAR until 2008-12-15, then the midpoint of DFEDTARU/DFEDTARL):
  before = target on the day before the decision is announced (meeting end date - 1)
  after  = target two days after the announcement (covers changes effective the next day, e.g. 2020-03-03 -> 03-04)
  decision = hike / hold / cut from the sign of after - before.
Meetings and votes: 2007-2020 from the corpus (current version), 2021-2026 from public_fomc_* (Federal Reserve website).
Also compares the ground truth with the group's processed meetings/fomc_decisions_detailed.csv when it is given.
Writes data/policy_meetings.csv, data/policy_votes.csv, data/policy_truth_check.csv.
Usage: python policy_data.py [--group-decisions <path to fomc_decisions_detailed.csv>] [--public-from <docs/public_fomc/data>]
  --public-from reads the 2021+ meetings and votes from fetch_public_fomc.py's files instead of the database (e.g. before a fix is loaded)."""
import io, os, re, sys, urllib.request
import pandas as pd
import db

def fred(series):
    u = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    s = pd.read_csv(io.StringIO(urllib.request.urlopen(u, timeout=60).read().decode()), parse_dates=['observation_date'])
    return pd.to_numeric(s.set_index('observation_date')[series], errors='coerce').dropna()

old, up, lo = fred('DFEDTAR'), fred('DFEDTARU'), fred('DFEDTARL')
target = pd.concat([old[old.index < up.index.min()], (up + lo) / 2]).sort_index().asfreq('D').ffill()   # daily; midpoint from 2008-12-16
def at(day): return float(target.loc[:day].iloc[-1])

c = db.connect()
q = lambda s: pd.read_sql(s, c)
corpus = q("""SELECT meeting_date::text, meeting_end_date::text FROM fedai_corpus.meetings ORDER BY 1""").assign(source='corpus')
public = q("""SELECT meeting_date::text, meeting_end_date::text, votes_for, votes_against FROM fedai_corpus.public_fomc_meetings ORDER BY 1""").assign(source='public')
PUB = sys.argv[sys.argv.index('--public-from') + 1] if '--public-from' in sys.argv else None
if PUB:   # 2021+ meetings and votes from fetch_public_fomc.py output instead of the database (same content once loaded)
    public = pd.read_csv(f'{PUB}/meetings.csv', dtype=str)[['meeting_date', 'meeting_end_date', 'votes_for', 'votes_against']].assign(source='public')
    public[['votes_for', 'votes_against']] = public[['votes_for', 'votes_against']].astype(float)
macro = q('SELECT meeting_date::text, is_sep_meeting FROM fedai_corpus.meeting_macro_data')
public['stated_voters'] = public.votes_for + public.votes_against      # from the statement; a meeting whose names are not all out yet has fewer rows
M = pd.concat([corpus, public.drop(columns=['votes_for', 'votes_against'])], ignore_index=True).merge(macro, on='meeting_date', how='left')
M['meeting_end_date'] = M.meeting_end_date.fillna(M.meeting_date)
end = pd.to_datetime(M.meeting_end_date)
M['target_before'] = [at(d - pd.Timedelta(days=1)) for d in end]
M['target_after'] = [at(d + pd.Timedelta(days=2)) for d in end]
M['delta'] = (M.target_after - M.target_before).round(4)
M['decision'] = ['hike' if d > 0 else 'cut' if d < 0 else 'hold' for d in M.delta]

# Market expectation proxy (no fed funds futures on FRED): the 4-week Treasury bill yield the day before the announcement
# prices the average target over the next 4 weeks. Its usual gap to the target (the "basis") is the median over days 90-50
# before the announcement, a window whose bills mature before the meeting.
bill = fred('DTB4WK')
def implied(d):
    y = bill.loc[:d - pd.Timedelta(days=1)].iloc[-1]
    w = bill.loc[d - pd.Timedelta(days=90): d - pd.Timedelta(days=50)]
    basis = (w - target.reindex(w.index)).median()
    return round(float(y - basis), 4)
M['market_implied_after'] = [implied(d) for d in end]

V1 = q("""SELECT m.meeting_date::text, v.person_id::text AS person_id, p.display_name AS member_name, mp.meeting_role AS role,
                 CASE v.vote_action WHEN 'voted_for' THEN 'for' WHEN 'voted_against' THEN 'against' END AS vote
            FROM fedai_corpus.votes v
            JOIN fedai_corpus.meetings m ON m.active_version_id = v.version_id
            JOIN fedai_corpus.voting_events e ON e.event_id = v.event_id AND e.event_kind = 'monetary_policy'
            JOIN fedai_corpus.persons p ON p.person_id = v.person_id
            LEFT JOIN fedai_corpus.meeting_participants mp ON mp.version_id = v.version_id AND mp.person_id = v.person_id""")
V2 = q("""SELECT meeting_date::text, person_id::text, member_name, role, vote, dissent_preference FROM fedai_corpus.public_fomc_votes""")
if PUB:
    F = pd.read_csv(f'{PUB}/votes.csv', dtype=str)[['meeting_date', 'member_name', 'role', 'vote', 'dissent_preference']]
    V2 = F.merge(V2[['meeting_date', 'member_name', 'person_id']], on=['meeting_date', 'member_name'], how='left')
    V2['person_id'] = V2.person_id.fillna(V2.member_name.map(V2.dropna(subset=['person_id']).drop_duplicates('member_name').set_index('member_name').person_id))
V = pd.concat([V1, V2], ignore_index=True)
V['member_key'] = V.person_id.where(V.person_id.notna(), 'name:' + V.member_name.str.lower().str.strip())
V = V.sort_values(['meeting_date', 'member_name']).reset_index(drop=True)

# Direction of each dissent (tighter / easier / other policy than the Committee chose) and what it was about, read by hand from
# the statements (inputs/dissent_direction.csv). Every dissent must have a label; for 2021 onward the rate dissents are also
# checked against the preferred change parsed from the statement wording, compared with the actual change.
L = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs/dissent_direction.csv'))
V = V.merge(L[['meeting_date', 'member_name', 'direction', 'about']], on=['meeting_date', 'member_name'], how='left')
missing = V[(V.vote == 'against') & V.direction.isna()]
if len(missing): sys.exit('dissents without a direction label (add them to inputs/dissent_direction.csv):\n' + missing[['meeting_date', 'member_name', 'dissent_preference']].to_string())
V.loc[V.vote == 'for', ['direction', 'about']] = None
FRAC = {'1/4': 0.25, '1/2': 0.5, '3/4': 0.75}
def preferred_change(s):
    s = str(s); m = re.search(r'(raise|lower) the target range for the federal funds rate by ([\d./]+) percentage point', s)
    if m: return (1 if m.group(1) == 'raise' else -1) * (FRAC[m.group(2)] if m.group(2) in FRAC else float(m.group(2)))
    return 0.0 if re.search(r'(preferred|prefers) (no change|to maintain)', s) else None
chk = V[(V.vote == 'against') & V.dissent_preference.notna() & (V.about == 'rate')].merge(M[['meeting_date', 'delta']], on='meeting_date')
chk['auto'] = [('tighter' if p > d else 'easier') if p is not None and p != d else None for p, d in zip(chk.dissent_preference.map(preferred_change), chk.delta)]
bad = chk[chk.auto != chk.direction]
if len(bad): sys.exit('direction label disagrees with the statement wording:\n' + bad[['meeting_date', 'member_name', 'direction', 'auto', 'dissent_preference']].to_string())
print(f'dissent directions: {V.direction.value_counts().to_dict()}; {len(chk)} rate dissents since 2021 checked against the wording')
unused = L.merge(V[['meeting_date', 'member_name']], how='left', indicator=True).query('_merge == "left_only"')
if len(unused): print('NOTE labels for dissents not in the database (yet):', unused[['meeting_date', 'member_name']].values.tolist())
n = V.groupby('meeting_date').agg(n_voters=('vote', 'size'), n_against=('vote', lambda x: int((x == 'against').sum())))
M = M.merge(n, left_on='meeting_date', right_index=True, how='left')
M.to_csv('data/policy_meetings.csv', index=False); V.to_csv('data/policy_votes.csv', index=False)
print('meetings', len(M), M.groupby('source').size().to_dict(), 'decisions', M.decision.value_counts().to_dict())
print('votes', len(V), V.vote.value_counts().to_dict(), 'meetings without votes', int(M.n_voters.isna().sum()))

if '--group-decisions' in sys.argv:
    G = pd.read_csv(sys.argv[sys.argv.index('--group-decisions') + 1])
    G = G.rename(columns={'meeting_start': 'meeting_date', 'decision': 'group_decision', 'delta_target': 'group_delta'})
    X = M.merge(G[['meeting_date', 'group_decision', 'group_delta', 'target_mid_after']], on='meeting_date', how='inner')
    bad = X[(X.group_decision != X.decision) | ((X.group_delta - X.delta).abs() > 1e-6) | ((X.target_mid_after - X.target_after).abs() > 1e-6)]
    bad[['meeting_date', 'meeting_end_date', 'target_before', 'target_after', 'delta', 'decision', 'target_mid_after', 'group_delta', 'group_decision']] \
        .to_csv('data/policy_truth_check.csv', index=False)
    print(f'group file: {len(G)} meetings, {len(X)} in common, {len(bad)} differ'); print(bad[['meeting_date', 'decision', 'delta', 'group_decision', 'group_delta']].to_string(index=False))
