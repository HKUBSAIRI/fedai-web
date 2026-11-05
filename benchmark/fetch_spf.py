"""Survey of Professional Forecasters (Federal Reserve Bank of Philadelphia), median responses: the 3-month Treasury bill
path (TBILL2-6 quarterly, TBILLA-D annual averages for the survey year and the next three years), the 10-year average
bill rate (BILL10, first-quarter surveys only), unemployment and core PCE inflation. Writes data/spf.csv; a frozen copy
is kept in inputs/spf_median.csv (run_all.sh uses it unless REFRESH_SPF=1).
Each survey is released in the middle month of its quarter, before that quarter's SEP meeting."""
import io, urllib.request, pandas as pd
URL = ('https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/'
       'historical-data/medianlevel.xlsx')
raw = urllib.request.urlopen(URL, timeout=60).read()
out = None
for sheet, cols in [('TBILL', ['TBILL2', 'TBILL3', 'TBILL4', 'TBILL5', 'TBILL6', 'TBILLA', 'TBILLB', 'TBILLC', 'TBILLD']),
                    ('BILL10', ['BILL10']), ('UNEMP', ['UNEMPA', 'UNEMPB', 'UNEMPC', 'UNEMPD']),
                    ('COREPCE', ['COREPCEA', 'COREPCEB', 'COREPCEC'])]:
    s = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)[['YEAR', 'QUARTER'] + cols]
    out = s if out is None else out.merge(s, on=['YEAR', 'QUARTER'], how='outer')
out = out[(out.YEAR >= 2010) & (out.YEAR <= 2021)]
for c in out.columns[2:]: out[c] = pd.to_numeric(out[c], errors='coerce')
out.to_csv('data/spf.csv', index=False)
print(len(out), 'surveys', out.YEAR.min(), '-', out.YEAR.max())
