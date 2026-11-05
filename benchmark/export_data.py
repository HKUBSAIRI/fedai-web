"""Step 0: export everything the benchmarks read into data/ (read-only database account + public FRED series).
Database (fedai_corpus): SEP projections, meeting roles, real-time macro data, meetings, persons, transcript turns 2011-2020.
FRED (fredgraph.csv, no key): Treasury yields 2011-2020 and 2024-, CPI-U (lifetime inflation experience).
Usage: python export_data.py [--skip-fred]"""
import sys, time, urllib.request, warnings
import pandas as pd
import db
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

Q = {
    'projections': """SELECT meeting_date, participant_number, person_id::text, participant_name, affiliation, projection_horizon, projection_year,
                             is_long_run, gdp_growth, unemployment_rate, pce_inflation, core_pce_inflation, federal_funds_rate
                        FROM fedai_corpus.current_sep_projections ORDER BY meeting_date, participant_number, is_long_run, projection_year""",
    'roles': """SELECT m.meeting_date, m.meeting_id::text, mp.person_id::text, mp.meeting_role, mp.affiliation
                  FROM fedai_corpus.meetings m JOIN fedai_corpus.meeting_participants mp ON mp.version_id = m.active_version_id""",
    'macro': 'SELECT * FROM fedai_corpus.meeting_macro_data ORDER BY meeting_date',
    'meetings': 'SELECT meeting_id::text, meeting_date, meeting_end_date, meeting_type FROM fedai_corpus.meetings ORDER BY meeting_date',
    'persons': 'SELECT person_id::text, canonical_name, display_name FROM fedai_corpus.persons',
}
TURNS = """SELECT t.meeting_date, t.person_id::text, t.sequence_number, t.body_clean
             FROM fedai_corpus.current_turns t
            WHERE t.meeting_date >= '2011-01-01' AND t.person_id IS NOT NULL AND length(t.body_clean) > 200"""
YIELDS = ['DTB3', 'DTB6', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS10']

c = db.connect()
for k, s in Q.items():
    df = pd.read_sql(s, c); df.to_csv(f'data/{k}.csv', index=False); print(k, df.shape)
df = pd.read_sql(TURNS, c); df.to_csv('data/turns_2011_2020.csv.gz', index=False, compression='gzip'); print('turns', df.shape)
c.close()

def fred(series, start, end, out):
    u = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}' + (f'&coed={end}' if end else '')
    for a in range(5):
        try:
            data = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'FedAI-research/0.1'}), timeout=60).read()
            open(out, 'wb').write(data); return
        except Exception as e:
            print('retry', series, e); time.sleep(5 * (a + 1))
    raise RuntimeError(f'FRED download failed: {series}')
if '--skip-fred' not in sys.argv:
    for s in YIELDS:
        fred(s, '2011-01-01', '2020-12-31', f'data/fred_{s}.csv')       # benchmarks_v3 (market forwards, 2012-2020 SEPs)
        fred(s, '2024-01-01', None, f'data/fred2_{s}.csv')              # post_cutoff_eval (2024-09 onward)
    fred('CPIAUCNS', '1913-01-01', None, 'data/fred_CPIAUCNS.csv')      # features_external (lifetime inflation)
    print('FRED series saved')
