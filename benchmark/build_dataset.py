"""One row per (SEP meeting, participant, horizon) with the fed funds rate dot as target and pre-meeting information as features."""
import pandas as pd, numpy as np
P = pd.read_csv('data/projections.csv', parse_dates=['meeting_date'])
P = P[P.federal_funds_rate.notna()].copy()
P['year'] = P.meeting_date.dt.year
P['h'] = np.where(P.is_long_run, 'LR', (P.projection_year - P.year).astype('Int64').astype(str))
P = P[P.h.isin(['0', '1', '2', '3', 'LR'])]
macro = pd.read_csv('data/macro.csv', parse_dates=['meeting_date'])
macro['target_mid'] = (macro.target_lower + macro.target_upper) / 2
P = P.merge(macro[['meeting_date', 'target_mid', 'inflation', 'core_inflation', 'unrate', 'gdp_growth', 'fedfunds', 'delta_unrate', 'delta_inflation']], on='meeting_date', how='left')
roles = pd.read_csv('data/roles.csv', parse_dates=['meeting_date'])
P = P.merge(roles[['meeting_date', 'person_id', 'meeting_role']], on=['meeting_date', 'person_id'], how='left')
P['voter'] = P.meeting_role.isin(['Chair', 'Vice Chair', 'Voter']).astype(int)
P['chair'] = (P.meeting_role == 'Chair').astype(int)
P['governor'] = P.affiliation.fillna('').str.contains('Board', case=False).astype(int)
meetings = sorted(P.meeting_date.unique())
P['sep_index'] = P.meeting_date.map({m: i for i, m in enumerate(meetings)})
P = P.sort_values(['meeting_date', 'person_id', 'h']).reset_index(drop=True)
P.to_csv('data/dataset.csv', index=False)
print(P.shape, 'meetings', len(meetings), meetings[0], meetings[-1])
print(P.h.value_counts().to_dict()); print('missing macro', P.target_mid.isna().sum(), 'missing role', P.meeting_role.isna().sum())
print(P.groupby('h').federal_funds_rate.describe()[['mean', 'std', 'min', 'max']].round(2))
# participants.csv: one row per person with SEP dots (name, first/last SEP, number of SEPs, usual affiliation); used by wiki_bio and narr_bench
x = P.groupby(['person_id', 'participant_name']).meeting_date.agg(['min', 'max', 'nunique']).reset_index()
aff = roles.merge(P[['meeting_date', 'person_id']].drop_duplicates(), on=['meeting_date', 'person_id']).groupby('person_id').affiliation.agg(
    lambda s: s.dropna().mode().iloc[0] if s.notna().any() else '')
x['aff'] = x.person_id.map(aff); x.to_csv('data/participants.csv', index=False)
