"""Frozen numbers for the pre-registered test on the individual dots of 2021 (released with a five-year lag, expected in early
2027). Run after committee_test_2021.py. Writes to docs/benchmark/results/committee_2021/:
  prereg_2021_committee_shift.csv   the committee-shift forecasts c_hat of the no-parameter rules for the four SEPs of 2021,
                                    by target year (these need no individual data and are final now)
  prereg_2021-03_individual_predictions.csv   the March 2021 dot forecasts of the no-parameter methods: own December 2020
                                    dot (public since 2025, in the database) + c_hat; longer run = own dot; floored at 0.125.
                                    The June-December 2021 forecasts need the own previous dots and are computed by the same
                                    formulas when the dots are released.
Methods: B0 naive; B30b lag + median(rd25 d_spf, rd25 d_fwd, rd25 c_prev/2);
B40 lag + mean(d_spf, d_fwd, c_prev/2); B40g lag + rd25(that mean); B31 lag + S(d_spf, 12.5bp)."""
import os, hashlib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
R = os.path.join(REPO, 'docs', 'benchmark', 'results', 'committee_2021')
C = pd.read_csv(os.path.join(R, 'cells.csv'), parse_dates=['meeting_date'])
COLS = {'C30b median(z1,z2,z3)': 'B30b', 'Ceq mean(d_spf,d_fwd,c_prev/2)': 'B40', 'Ceqg rd25(mean) [post hoc]': 'B40g', 'C31 soft beta=1': 'B31'}
Y = C[(C.meeting_date.dt.year == 2021)][['meeting_date', 'target', 'h', 'm_prev'] + list(COLS)].rename(columns=COLS)
Y.loc[Y.h == 'LR', ['B30b', 'B40', 'B40g', 'B31']] = 0.0
Y.to_csv(os.path.join(R, 'prereg_2021_committee_shift.csv'), index=False)
I = pd.read_csv(os.path.join(REPO, 'docs', 'benchmark', 'work', 'frozen_2015', 'data', 'projections.csv'), parse_dates=['meeting_date'])
D = I[(I.meeting_date == '2020-12-15') & I.federal_funds_rate.notna()].copy()
D['lag'] = np.round(D.federal_funds_rate / 0.125) * 0.125
D['target'] = np.where(D.is_long_run, 'LR', D.projection_year.astype('Int64').astype(str))
k = Y[Y.meeting_date == '2021-03-16'].set_index('target')
rows = []
for _, r in D.iterrows():
    if r.target not in k.index: continue
    c = k.loc[r.target]; f = lambda m: max(0.125, r.lag + (c[m] if pd.notna(c[m]) else 0.0))
    rows.append(dict(meeting_date='2021-03-16', participant=r.person_id, participant_name=r.participant_name, horizon='LR' if r.target == 'LR' else int(r.target) - 2021,
                     lag=r.lag, B0=r.lag, **{m: f(m) for m in ('B30b', 'B40', 'B40g', 'B31')}))
P = pd.DataFrame(rows).sort_values(['participant_name', 'horizon']).reset_index(drop=True)
P.to_csv(os.path.join(R, 'prereg_2021-03_individual_predictions.csv'), index=False)
pd.set_option('display.width', 250)
print(Y.round(4).to_string(index=False)); print(len(P), 'March 2021 rows,', P.participant.nunique(), 'participants')
for f in ['prereg_2021_committee_shift.csv', 'prereg_2021-03_individual_predictions.csv']:
    print(f, 'sha256', hashlib.sha256(open(os.path.join(R, f), 'rb').read()).hexdigest()[:16])
for f in ['benchmarks_v10.py', 'committee_test_2021.py', 'prereg_2021.py']:
    print(f, 'sha256', hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()[:16])
