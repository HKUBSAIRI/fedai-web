"""Memorization probe: ask the model, with NO data, what a named participant wrote for the fed funds rate in a given SEP.
If recall beats the statistical benchmarks, model predictions on these SEPs are contaminated by memory."""
import json, random, numpy as np, pandas as pd, concurrent.futures as cf
import llm
P = pd.read_csv('data/predictions_v5.csv', parse_dates=['meeting_date'])
random.seed(0)
cells = P[(P.meeting_date >= '2014') & (P.h != 'LR')][['meeting_date', 'participant_name']].drop_duplicates()
sample = cells.sample(80, random_state=0)
SYS = 'You are an expert on Federal Reserve history. Answer from memory only. Reply in JSON.'
def ask(r):
    d = r.meeting_date.strftime('%B %d, %Y'); y = r.meeting_date.year
    q = (f'In the Summary of Economic Projections submitted for the FOMC meeting that began on {d}, what appropriate year-end federal funds rate '
         f'(midpoint, percent) did {r.participant_name} project for the end of {y}, {y + 1}, {y + 2} and {y + 3} (if that year was included)? '
         f'Individual projections are published with a five-year lag. Also give the committee median for the same years. '
         f'Return JSON {{"own": {{"{y}": x, "{y + 1}": x, ...}}, "median": {{...}}, "confidence": "low|medium|high"}}. Use null if you do not know.')
    try: return r, ask_llm(q)
    except Exception as e: return r, {'error': str(e)}
def ask_llm(q): return llm.chat(SYS, q, max_tokens=300)
rows = []
with cf.ThreadPoolExecutor(4) as ex:
    for r, a in ex.map(ask, sample.itertuples()):
        for kind in ('own', 'median'):
            for k, v in (a.get(kind) or {}).items():
                try: rows.append({'meeting_date': r.meeting_date, 'participant_name': r.participant_name, 'kind': kind, 'proj_year': int(k), 'recall': float(v), 'confidence': a.get('confidence')})
                except (TypeError, ValueError): pass
R = pd.DataFrame(rows); R.to_csv('data/lookahead_probe.csv', index=False)
T = P[['meeting_date', 'participant_name', 'proj_year', 'h', 'federal_funds_rate', 'B0_prior_own', 'B14_combo_plus_text_h0']]
own = R[R.kind == 'own'].merge(T, on=['meeting_date', 'participant_name', 'proj_year'])
med = R[R.kind == 'median'].merge(T.groupby(['meeting_date', 'proj_year']).federal_funds_rate.median().rename('true_median').reset_index(), on=['meeting_date', 'proj_year']).drop_duplicates(['meeting_date', 'proj_year'])
print('answered cells', len(own), 'of', len(T.merge(sample, on=['meeting_date', 'participant_name'])), '| confidence', R.drop_duplicates(['meeting_date', 'participant_name']).confidence.value_counts().to_dict())
for c, lab in [('recall', 'LLM recall (no data)'), ('B0_prior_own', 'B0 same rows'), ('B14_combo_plus_text_h0', 'B14 same rows')]:
    print(f'{lab:22s} MAE own dot: {np.mean(np.abs(own[c] - own.federal_funds_rate)):.3f}')
print('exact own-dot matches:', f'{np.mean(np.isclose(own.recall, own.federal_funds_rate)):.1%}', '| B0 exact:', f'{np.mean(np.isclose(own.B0_prior_own, own.federal_funds_rate)):.1%}')
print(f'LLM recall MAE committee median: {np.mean(np.abs(med.recall - med.true_median)):.3f} (n={len(med)})')
print('usage', llm.USAGE)
