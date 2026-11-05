"""Pre-meeting staff/Beige Book summaries (sep_dual_agent.meeting_macro_contexts) as compact text, one per SEP meeting."""
import json, pandas as pd, db
N = pd.read_csv('data/narratives.csv', parse_dates=['meeting_date'])
c = db.connect(); cur = c.cursor()
cur.execute("SELECT event_start_date, cutoff_date, context_json FROM sep_dual_agent.meeting_macro_contexts")
rows = cur.fetchall(); c.close()
SEC = {'current_conditions': 'Current conditions', 'changes': 'Changes since the last meeting', 'outlook': 'Outlook', 'risks': 'Risks'}
out = {}
for start, cutoff, j in rows:
    if pd.Timestamp(start) not in set(N.meeting_date): continue
    L = []
    for block in ('macro', 'policy'):
        items = (j.get(block) or {}).get('items') or []
        for key, label in list(SEC.items()) + [(None, 'Other')]:
            its = [i for i in items if (i.get('section') == key if key else i.get('section') not in SEC)]
            if not its: continue
            L.append(f"## {'Policy background' if block == 'policy' else 'Economy'}: {label}")
            L += [f"- {i.get('topic') or ''}: {i['claim']}" for i in its]
    out[str(pd.Timestamp(start).date())] = {'cutoff': str(cutoff), 'text': '\n'.join(L)}
json.dump(out, open('data/narr_context.json', 'w'), ensure_ascii=False)
lens = pd.Series({k: len(v['text'].split()) for k, v in out.items()})
print(len(out), 'SEP meetings with context; words min/median/max', lens.min(), lens.median(), lens.max())
print('SEP meetings without context:', sorted(set(N.meeting_date.dt.date.astype(str)) - set(out)))
