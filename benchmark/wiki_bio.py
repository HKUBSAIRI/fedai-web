"""Birth year and education of each SEP participant from the English Wikipedia infobox (source page recorded)."""
import json, re, time, urllib.request, urllib.parse, pandas as pd
P = pd.read_csv('data/participants.csv')
TITLE = {'Ben S. Bernanke': 'Ben Bernanke', 'Janet L. Yellen': 'Janet Yellen', 'Jerome H. Powell': 'Jerome Powell', 'Lael Brainard': 'Lael Brainard',
         'Richard H. Clarida': 'Richard Clarida', 'Randal K. Quarles': 'Randal Quarles', 'Michelle W. Bowman': 'Michelle Bowman', 'Stanley Fischer': 'Stanley Fischer',
         'Daniel K. Tarullo': 'Daniel Tarullo', 'Jeremy C. Stein': 'Jeremy C. Stein', 'Sarah Bloom Raskin': 'Sarah Bloom Raskin', 'Elizabeth A. Duke': 'Elizabeth A. Duke',
         'William C. Dudley': 'William C. Dudley', 'John C. Williams': 'John C. Williams (economist)', 'Esther L. George': 'Esther George', 'Loretta J. Mester': 'Loretta J. Mester',
         'Eric S. Rosengren': 'Eric S. Rosengren', 'Neel Kashkari': 'Neel Kashkari', 'Raphael W. Bostic': 'Raphael Bostic', 'James Bullard': 'James B. Bullard',
         'Charles I. Plosser': 'Charles Plosser', 'Jeffrey M. Lacker': 'Jeffrey M. Lacker', 'Charles L. Evans': 'Charles L. Evans', 'Thomas I. Barkin': 'Thomas Barkin',
         'Narayana Kocherlakota': 'Narayana Kocherlakota', 'Dennis P. Lockhart': 'Dennis Lockhart', 'Sandra Pianalto': 'Sandra Pianalto', 'Richard W. Fisher': 'Richard W. Fisher',
         'Robert S. Kaplan': 'Robert Steven Kaplan', 'Patrick Harker': 'Patrick T. Harker', 'Mary C. Daly': 'Mary C. Daly', 'Mark L. Mullinix': 'Mark Mullinix'}
def page(title):
    u = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode({'action': 'query', 'prop': 'revisions', 'rvprop': 'content', 'rvslots': 'main', 'titles': title, 'redirects': 1, 'format': 'json', 'formatversion': 2})
    for attempt in range(6):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'FedAI-research/0.1 (HKU academic project)'}), timeout=60)); break
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(10 * (attempt + 1))
    else: raise RuntimeError('rate limited')
    p = d['query']['pages'][0]
    return (p.get('title'), p['revisions'][0]['slots']['main']['content']) if 'revisions' in p else (None, '')
rows = []
for name in P.participant_name:
    t = TITLE.get(name, name); title, txt = page(t)
    by = None
    m = re.search(r'\|\s*birth_date\s*=\s*\{\{\s*birth[ _]date(?: and age)?\s*\|(?:[a-z]+=\w+\|)*\s*(\d{4})', txt, re.I) or re.search(r'\|\s*birth_date\s*=[^\n]*?(\d{4})', txt)
    if m: by = int(m.group(1))
    edu = ''
    for f in ('education', 'alma_mater'):
        mm = re.search(rf'\|\s*{f}\s*=(.*?)(?=\n\s*\|\s*\w+\s*=|\n\}}\}})', txt, re.S)
        if mm: edu += ' ' + re.sub(r'\{\{|\}\}|\[\[|\]\]|<[^>]+>|\|', ' ', mm.group(1))
    phd = re.search(r'([A-Z][\w .&\'-]{2,60}?)\s*\(\s*(?:PhD|Ph\.D\.?|DPhil|doctorate)', edu)
    rows.append({'participant_name': name, 'wiki_title': title, 'birth_year': by, 'education_raw': re.sub(r'\s+', ' ', edu).strip()[:300],
                 'phd_school_guess': phd.group(1).strip() if phd else None, 'source': f'https://en.wikipedia.org/wiki/{urllib.parse.quote((title or t).replace(" ", "_"))}' if title else None})
    time.sleep(3)
B = pd.DataFrame(rows); B.to_csv('data/bios_raw.csv', index=False)
print(B[['participant_name', 'wiki_title', 'birth_year', 'phd_school_guess']].to_string()); print(B.education_raw.str[:140].to_string())
