"""Hawk/dove tone of each participant's remarks per meeting (dictionary method in the spirit of Apel & Blix Grimaldi 2012).
A hawkish hit is an upward/strong modifier within 4 words of a price/rate/activity noun, or an explicit tightening phrase;
dovish hits are the mirror image. Score = (H - D) / (H + D) over the person's turns in the meeting, then demeaned within meeting."""
import re, numpy as np, pandas as pd
T = pd.read_csv('data/turns_2011_2020.csv.gz', parse_dates=['meeting_date'])
NOUN = r'(inflation|inflationary|prices?|wages?|rates?|growth|economy|activity|demand|labor market|employment|expectations)'
UP = r'(increas\w*|higher|rising|rise|rose|strong\w*|tight\w*|accelerat\w*|above|overheat\w*|firm\w*|pick\w* up|upward|upside)'
DOWN = r'(decreas\w*|lower|falling|fall|fell|weak\w*|slack|slow\w*|decelerat\w*|below|soft\w*|subdued|downward|downside|disappoint\w*)'
HAWK_P = r'(tighten\w*|raise (the )?(federal funds |policy )?rates?|rate increases?|remove (some )?accommodation|normaliz\w*|liftoff|lift-off|hike\w*|firming)'
DOVE_P = r'(accommodat\w*|patien\w*|ease|easing|lower (the )?(federal funds |policy )?rates?|rate cuts?|cut rates|stimul\w*|asset purchases|forward guidance|wait)'
def near(a, b, text):
    return len(re.findall(rf'\b{a}\W+(?:\w+\W+){{0,4}}?{b}\b', text)) + len(re.findall(rf'\b{b}\W+(?:\w+\W+){{0,4}}?{a}\b', text))
rows = []
for r in T.itertuples():
    t = str(r.body_clean).lower()
    h = near(UP, NOUN, t) + len(re.findall(rf'\b{HAWK_P}', t))
    d = near(DOWN, NOUN, t) + len(re.findall(rf'\b{DOVE_P}', t))
    rows.append((r.meeting_date, r.person_id, h, d))
S = pd.DataFrame(rows, columns=['meeting_date', 'person_id', 'H', 'D']).groupby(['meeting_date', 'person_id'])[['H', 'D']].sum().reset_index()
S = S[(S.H + S.D) >= 5]
S['tone'] = (S.H - S.D) / (S.H + S.D)
S['tone_rel'] = S.tone - S.groupby('meeting_date').tone.transform('mean')
S.to_csv('data/text_tone.csv', index=False)
print(S.shape, S.tone.describe().round(3).to_dict())
# sanity: who is most hawkish / dovish on average (2012-2020)?
names = pd.read_csv('data/persons.csv').set_index('person_id').display_name
avg = S[S.meeting_date >= '2012'].groupby('person_id').tone_rel.agg(['mean', 'count'])
avg = avg[avg['count'] >= 8].sort_values('mean'); avg.index = avg.index.map(names)
print('most dovish:', avg.head(6)['mean'].round(3).to_dict()); print('most hawkish:', avg.tail(6)['mean'].round(3).to_dict())
