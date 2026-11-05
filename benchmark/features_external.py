"""Personal features from public biographies and regional data.
- Lifetime inflation experience (Malmendier, Nagel & Yan 2017): weighted average of annual CPI inflation from birth to the
  previous calendar year, weights (age - k)^theta with theta = 3.044 (Malmendier & Nagel 2016), so recent years weigh more.
- PhD school type (Bordo & Istrefi 2018): freshwater / saltwater / other PhD / no PhD.
- District unemployment gap: unemployment of the Reserve Bank's head-office state minus the national rate, both as published
  the day before the meeting (ALFRED vintages); 0 for Board members."""
import re, time, io, urllib.request, numpy as np, pandas as pd
B = pd.read_csv('data/bios_raw.csv')
B.loc[B.participant_name == 'Mary C. Daly', 'birth_year'] = 1963       # Wikipedia: aged 55 as of 2018-10-01
def school_type(e):
    m = re.findall(r'([A-Z][^()]{2,70}?)\s*\(([^)]*)\)', str(e)); phd = ' '.join(s for s, d in m if re.search(r'Ph\.?\s?D|Doctor of Philosophy|DPhil', d))
    if not phd: return 'no_phd'
    if re.search(r'Chicago|Carnegie|Minnesota|Rochester', phd): return 'freshwater'
    if re.search(r'Harvard|Massachusetts Institute|MIT|Yale|Princeton|Berkeley|Stanford|Pennsylvania|Columbia', phd): return 'saltwater'
    return 'other_phd'
B['school'] = B.education_raw.map(school_type)
B.loc[B.wiki_title.isna(), 'school'] = 'unknown'
cpi = pd.read_csv('data/fred_CPIAUCNS.csv', parse_dates=['observation_date'])
ann = cpi.groupby(cpi.observation_date.dt.year).CPIAUCNS.mean(); infl = (ann.pct_change() * 100).dropna()
def experience(birth, year, theta=3.044):
    if pd.isna(birth): return np.nan
    yrs = np.arange(max(int(birth), infl.index.min()), year)          # up to the previous calendar year
    age = year - birth
    k = year - yrs                                                     # lags 1..age
    w = np.clip(age - k, 0, None) ** theta
    return float(np.sum(w * infl.loc[yrs].values) / w.sum()) if w.sum() > 0 else np.nan
D = pd.read_csv('data/dataset.csv', parse_dates=['meeting_date'])
roles = pd.read_csv('data/roles.csv', parse_dates=['meeting_date'])[['meeting_date', 'person_id', 'affiliation']]
F = D[['meeting_date', 'person_id', 'participant_name', 'year']].drop_duplicates().merge(roles, on=['meeting_date', 'person_id'], how='left')
F = F.merge(B[['participant_name', 'birth_year', 'school']], on='participant_name', how='left')
F['age'] = F.year - F.birth_year
F['infl_exp'] = [experience(b, y) for b, y in zip(F.birth_year, F.year)]
STATE = {'Boston': 'MA', 'New York': 'NY', 'Philadelphia': 'PA', 'Cleveland': 'OH', 'Richmond': 'VA', 'Atlanta': 'GA', 'Chicago': 'IL',
         'St. Louis': 'MO', 'Minneapolis': 'MN', 'Kansas City': 'MO', 'Dallas': 'TX', 'San Francisco': 'CA'}
F['state'] = F.affiliation.fillna('').map(lambda a: next((s for k, s in STATE.items() if a.startswith(k)), None))
# state and national unemployment as published the day before each meeting (ALFRED, batched by vintage date)
meet = sorted(F.meeting_date.unique()); vds = [(pd.Timestamp(m) - pd.Timedelta(days=1)).strftime('%Y-%m-%d') for m in meet]
def alfred_last(series, dates):
    out = {}
    for i in range(0, len(dates), 12):
        ch = dates[i:i + 12]; url = f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={','.join([series] * len(ch))}&vintage_date={','.join(ch)}"
        for a in range(4):
            try: df = pd.read_csv(io.StringIO(urllib.request.urlopen(url, timeout=120).read().decode())); break
            except Exception: time.sleep(5 * (a + 1))
        for d in ch:
            col = f"{series}_{d.replace('-', '')}"; s = pd.to_numeric(df[col], errors='coerce').dropna() if col in df else pd.Series(dtype=float)
            out[d] = float(s.iloc[-1]) if len(s) else np.nan
        time.sleep(1)
    return out
nat = alfred_last('UNRATE', vds); st = {s: alfred_last(f'{s}UR', vds) for s in sorted(set(STATE.values()))}
vd_of = dict(zip(meet, vds))
F["district_ugap"] = [st[s][vd_of[m]] - nat[vd_of[m]] if isinstance(s, str) else 0.0 for m, s in zip(F.meeting_date, F.state)]
F.to_csv('data/features_external.csv', index=False)
B.to_csv('data/bios.csv', index=False)
print(F[['infl_exp', 'age', 'district_ugap']].describe().round(2)); print(B.school.value_counts().to_dict())
print(F.groupby('participant_name')[['birth_year', 'infl_exp']].mean().sort_values('infl_exp').round(2).iloc[[0, 1, 2, -3, -2, -1]])
