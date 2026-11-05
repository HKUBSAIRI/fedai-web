"""Read-only connection to the FedAI database. Settings come from the environment (PGHOST, PGPORT, PGDATABASE, PGUSER,
PGPASSWORD or ~/.pgpass, PGSSLMODE), or from the file named by FEDAI_ENV_FILE (default: the first .env found in this folder
or a parent folder, i.e. the repository's .env wherever the benchmark code sits)."""
import os, psycopg2
def _find_env():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, '.env')): return os.path.join(d, '.env')
        d = os.path.dirname(d)
    return ''
ENV_FILE = os.environ.get('FEDAI_ENV_FILE', _find_env())
def settings():
    env = {}
    if os.path.exists(ENV_FILE):
        env = dict(l.strip().split('=', 1) for l in open(ENV_FILE) if '=' in l and not l.startswith('#'))
    env.update({k: v for k, v in os.environ.items() if k.startswith('PG')})
    return env
def connect():
    env = settings()
    kw = dict(host=env['PGHOST'], port=env.get('PGPORT', '5432'), dbname=env['PGDATABASE'], user=env['PGUSER'],
              sslmode=env.get('PGSSLMODE', 'prefer'), application_name='fedai-benchmark-export', options='-c default_transaction_read_only=on')
    if env.get('PGPASSWORD'): kw['password'] = env['PGPASSWORD']
    return psycopg2.connect(**kw)
