"""Anonymous dot-plot distributions (Figure 2 of each SEP) from federalreserve.gov, for SEPs after the model's training cutoff."""
import re, html, time, urllib.request, pandas as pd
MEET = {'2024-06-11': '20240612', '2024-09-17': '20240918', '2024-12-17': '20241218', '2025-03-18': '20250319', '2025-06-17': '20250618',
        '2025-09-16': '20250917', '2025-12-09': '20251210', '2026-03-17': '20260318', '2026-06-16': '20260617', '2026-09-15': '20260916'}
rows = []
for md, rel in MEET.items():
    url = f'https://www.federalreserve.gov/monetarypolicy/fomcprojtabl{rel}.htm'
    s = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60).read().decode('utf-8', 'ignore')
    i = s.find('Figure 2.'); tbl = s[i: s.find('</table>', i)]
    head = [html.unescape(re.sub(r'<[^>]+>', '', h)).strip() for h in re.findall(r'<th[^>]*>(.*?)</th>', tbl, re.S)]
    years = [h for h in head if re.fullmatch(r'\d{4}|Longer run', h)]
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', tbl, re.S):
        cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', tr, re.S)]
        if not cells or not re.fullmatch(r'\d+\.\d+', cells[0]): continue
        for y, c in zip(years, cells[1:]):
            if c.isdigit(): rows += [{'meeting_date': md, 'target': y, 'rate': float(cells[0])}] * int(c)
    time.sleep(1)
D = pd.DataFrame(rows); D.to_csv('data/post_cutoff_dots.csv', index=False)
S = D.groupby(['meeting_date', 'target']).rate.agg(['count', 'median', 'mean', 'std']).round(3)
print(S.to_string())
