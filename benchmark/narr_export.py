"""Export SEP narrative answers with a stable question family (codes change across years, texts are stable)."""
import re, pandas as pd, db
c = db.connect()
A = pd.read_sql("""SELECT b.source_meeting_start AS meeting_date, a.question_code, q.question_text, a.participant_number, sp.person_id::text AS person_id,
                          sp.source_name, a.response_status, a.answer
                     FROM fedai_corpus.sep_answers a JOIN fedai_corpus.sep_batches b USING (batch_id)
                     JOIN fedai_corpus.sep_questions q ON q.batch_id = a.batch_id AND q.question_code = a.question_code
                     LEFT JOIN fedai_corpus.sep_participants sp ON sp.batch_id = a.batch_id AND sp.participant_number = a.participant_number""", c)
c.close()
def family(t):
    t = t.lower()
    rules = [('convergence', 'Q1c longer-run convergence'), ('uncertainty attached', 'Q2a uncertainty'), ('risk weighting', 'Q2b risk balance'),
             ('balance sheet', 'Q3 balance sheet'), ('appropriate path of the federal funds', 'Q3 fed funds path'), ('appropriate monetary policy', 'Q3 fed funds path'),
             ('tealbook', 'Q4c vs Tealbook'), ('greenbook', 'Q4c vs Tealbook'), ('since the previous', 'Q4b changes since last SEP'), ('change since', 'Q4b changes since last SEP'),
             ('central economic outlook', 'Q4a outlook'), ('shaping your', 'Q4a outlook'), ('key factors', 'Q4a outlook')]
    for k, v in rules:
        if k in t: return v
    return 'other'
A['family'] = A.question_text.map(family)
A['meeting_date'] = pd.to_datetime(A.meeting_date)
A.to_csv('data/narratives.csv', index=False)
x = A[A.response_status == 'answered']
print(x.groupby('family').size().to_dict())
print('other examples:', A[A.family == 'other'].question_text.str[:90].unique()[:5])
T = x[x.meeting_date >= '2019-01-01']
print('2019-2020 answered:', len(T), 'SEPs', T.meeting_date.nunique(), 'people', T.person_id.nunique(), T.groupby('family').size().to_dict())
