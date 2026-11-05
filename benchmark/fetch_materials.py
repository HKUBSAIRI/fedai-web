"""Public pre-meeting materials for each post-cutoff SEP meeting: the previous FOMC statement, the previous meeting's minutes
(published three weeks after that meeting, i.e. before this one) and the latest Beige Book published before the meeting."""
import re, html, json, time, urllib.request, pandas as pd
FED = 'https://www.federalreserve.gov'
def get(path):
    for a in range(4):
        try: return urllib.request.urlopen(urllib.request.Request(FED + path, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60).read().decode('utf-8', 'ignore')
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(3)
    return None
def text(s):
    s = re.sub(r'<(script|style|nav|header|footer)[^>]*>.*?</\1>', ' ', s, flags=re.S)
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()
# Beige Book editions and publication dates
bb = []
pages = ''.join(get(f'/monetarypolicy/beigebook{y}.htm') or '' for y in (2024, 2025)) + (get('/monetarypolicy/publications/beige-book-default.htm') or '')
if True:
    for code in sorted(c for c in set(re.findall(r'beigebook(\d{6})(?:-summary)?\.htm', pages)) if c >= '202401'):
        s = get(f'/monetarypolicy/beigebook{code}-summary.htm')
        if not s: continue
        t = text(s); m = re.search(r'((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, 20\d\d)', t[t.find('Beige Book'):] if 'Beige Book' in t else t)
        bb.append({'code': code, 'published': pd.Timestamp(m.group(1)) if m else None, 'text': t})
        time.sleep(.5)
BB = pd.DataFrame(bb).dropna(subset=['published']).sort_values('published')
print(BB[['code', 'published']].to_string())
macro = pd.read_csv('data/macro.csv', parse_dates=['meeting_date']); allm = list(macro.meeting_date)
SEPS = ['2024-09-17', '2024-12-17', '2025-03-18', '2025-06-17', '2025-09-16', '2025-12-09', '2026-03-17', '2026-06-16', '2026-09-15']
out = {}
for sd in SEPS:
    t = pd.Timestamp(sd); p = max(m for m in allm if m < t); p2 = (p + pd.Timedelta(days=1)).strftime('%Y%m%d')
    st = get(f'/newsevents/pressreleases/monetary{p2}a.htm'); mi = get(f'/monetarypolicy/fomcminutes{p2}.htm')
    stt = text(st) if st else ''; mit = text(mi) if mi else ''
    a = stt.find('Recent indicators') if 'Recent indicators' in stt else stt.find('Available indicators'); b = stt.find('Voting for')
    statement = stt[a:b] if a >= 0 and b > a else stt[:3000]
    i = mit.find("Participants' Views on Current Conditions"); j = mit.find('Voting for this action')
    minutes = mit[i:j] if i >= 0 and j > i else mit[:20000]
    e = BB[BB.published < t].iloc[-1]
    k = e.text.find('Overall Economic Activity'); beige = e.text[k:k + 8000] if k >= 0 else e.text[:8000]
    out[sd] = {'prev_meeting': str(p.date()), 'statement': statement, 'minutes': minutes, 'beige_code': e.code, 'beige_published': str(e.published.date()), 'beige': beige}
    print(sd, 'prev', p.date(), '| statement', len(statement.split()), 'words | minutes', len(minutes.split()), 'words | Beige', e.code, e.published.date(), len(beige.split()), 'words')
json.dump(out, open('data/materials.json', 'w'))
