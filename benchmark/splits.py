"""Test-set definitions shared by evaluate.py and evaluate_external.py.
- 'test 2019-2020', 'test 2014-2020': calendar splits.
- 'person 60/40': as described in sep_dual_agent.experiment_assignments: for each participant, order the SEPs in which he/she
  submitted a fed funds dot; the first one is a cold start and is not scored; of the remaining SEPs the first 60% are for
  learning and the last 40% are scored. Participants with fewer than 3 remaining SEPs contribute nothing to the test."""
import numpy as np, pandas as pd
def person_6040_mask(df, date_col='meeting_date', person_col='person_id'):
    seps = df[[person_col, date_col]].drop_duplicates().sort_values(date_col)
    test = set()
    for pid, g in seps.groupby(person_col):
        d = list(g[date_col])[1:]                       # drop the cold-start SEP
        if len(d) < 3: continue
        k = int(round(0.6 * len(d)))
        test.update((pid, x) for x in d[k:])
    return pd.Series([(p, d) in test for p, d in zip(df[person_col], df[date_col])], index=df.index)
def mask(df, split):
    if split == 'person 60/40': return person_6040_mask(df)
    lo, hi = {'test 2019-2020': ('2019-01-01', '2020-12-31'), 'test 2014-2020': ('2014-01-01', '2020-12-31'), 'all': ('2000-01-01', '2100-01-01')}[split]
    return (df.meeting_date >= lo) & (df.meeting_date <= hi)
