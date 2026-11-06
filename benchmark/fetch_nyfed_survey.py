"""Expert forecasts of the median dot: before every SEP meeting since March 2016, the New York Fed's Survey of Primary
Dealers (SPD) and Survey of Market Participants (SMP) ask "What are your expectations for the (most likely levels of the)
medians of FOMC participants' target federal funds rate projections in the SEP?" (year-end of the SEP year and the next
two or three years, and the longer run). Responses are due about a week before the meeting; the results are published
about three weeks after it. This script downloads the 2016-2020 results (PDF only) and reads the median response per
survey and target into data/nyfed_expected_median.csv; a frozen copy is kept in inputs/nyfed_expected_median.csv
(run_all.sh uses it unless REFRESH_NYFED=1). Needs pdftotext (poppler-utils)."""
import csv, os, re, subprocess, tempfile, urllib.request
BASE = 'https://www.newyorkfed.org/medialibrary/media/markets/survey/'
FILES = {  # (year, month): (SPD results, SMP results); the March 2020 SEP was cancelled
    (2016, 3): ('2016/mar-2016-results.pdf', '2016/mar-results-mp.pdf'),
    (2016, 6): ('2016/jun-2016-results.pdf', '2016/jun-results-mp.pdf'),
    (2016, 9): ('2016/sep-2016-results.pdf', '2016/sep-results-mp.pdf'),
    (2016, 12): ('2016/Dec-2016-SPD-Results-Public-Release.pdf', '2016/Dec-2016-Flash-SMP-Results-Public-Release.pdf'),
    (2017, 3): ('2017/Mar-2017-SPD-Results-Public-Release.pdf', '2017/Mar-2017-SMP-Results-Public-Release.pdf'),
    **{(y, m): (f'{y}/{mon}-{y}-spd-results.pdf', f'{y}/{mon}-{y}-smp-results.pdf')
       for y in (2017, 2018, 2019, 2020) for m, mon in ((3, 'mar'), (6, 'jun'), (9, 'sep'), (12, 'dec'))
       if (y, m) not in ((2017, 3), (2020, 3))},
}
QUESTION = re.compile(r"median[s]? of FOMC participants' target federal funds rate projections", re.I)

def read(pdf_bytes):
    """Year-end targets and the median response of the question above, from the results PDF."""
    with tempfile.TemporaryDirectory() as tmp:
        open(f'{tmp}/r.pdf', 'wb').write(pdf_bytes)
        subprocess.run(['pdftotext', '-layout', f'{tmp}/r.pdf', f'{tmp}/r.txt'], check=True)
        text = open(f'{tmp}/r.txt', encoding='utf-8', errors='replace').read()
    flat = re.sub(r'\s+', ' ', text.replace('’', "'"))
    rest = flat[QUESTION.search(flat).end():][:1500]
    hdr = re.search(r"Year-[Ee]nd(?: Year-[Ee]nd)*(?: Longer)? ((?:20\d\d ?)+)(?:Longer )?Run", rest)
    years = re.findall(r'20\d\d', hdr.group(1))
    values = [float(v) for v in re.findall(r'(-?\d+\.\d+)%', re.search(r"Median ((?:-?\d+\.\d+% ?)+)", rest[hdr.end():]).group(1))]
    assert len(values) == len(years) + 1, (years, values)
    return list(zip(years + ['LR'], values))

rows = []
for (year, month), pair in sorted(FILES.items()):
    for survey, path in zip(('SPD', 'SMP'), pair):
        req = urllib.request.Request(BASE + path, headers={'User-Agent': 'Mozilla/5.0'})
        for target, value in read(urllib.request.urlopen(req, timeout=60).read()):
            rows.append(dict(survey=survey, sep_year=year, sep_month=month, target=target, expected_median=value, source=BASE + path))
os.makedirs('data', exist_ok=True)
with open('data/nyfed_expected_median.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows from', len(FILES), 'SEPs')
