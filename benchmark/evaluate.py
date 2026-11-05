"""Metrics as in multi_agent/evaluation/evaluation_metrics.md: MAE, RMSE, Pearson corr at member and committee level;
paired t-test and Wilcoxon vs a reference method on matched observations."""
import sys, numpy as np, pandas as pd
from scipy import stats
from splits import mask
F = sys.argv[1] if len(sys.argv) > 1 else 'data/predictions_v1.csv'
P = pd.read_csv(F, parse_dates=['meeting_date'])
methods = [c for c in P.columns if c[:1] == 'B']
REF = 'B0_prior_own'
SPLITS = ['test 2019-2020', 'test 2014-2020', 'person 60/40']
def met(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); y, p = y[m], p[m]
    if len(y) < 3: return dict(n=len(y), MAE=np.nan, RMSE=np.nan, Corr=np.nan)
    return dict(n=len(y), MAE=np.mean(np.abs(p - y)), RMSE=np.sqrt(np.mean((p - y) ** 2)), Corr=np.corrcoef(p, y)[0, 1])
def star(p): return '***' if p < .01 else '**' if p < .05 else '*' if p < .10 else ''
rows = []
for sname in SPLITS:
    T = P[mask(P, sname)]
    C = T.groupby(['meeting_date', 'h'])[['federal_funds_rate'] + methods].median().reset_index()
    for m in methods:
        for h in ['all', '0', '1', '2', '3', 'LR']:
            t = T if h == 'all' else T[T.h == h]; c = C if h == 'all' else C[C.h == h]
            r = {'split': sname, 'method': m, 'h': h}
            r.update({f'mem_{k}': v for k, v in met(t.federal_funds_rate.values, t[m].values).items()})
            r.update({f'com_{k}': v for k, v in met(c.federal_funds_rate.values, c[m].values).items() if k != 'n'})
            if m != REF:   # paired tests on absolute errors vs reference, matched rows
                s = t[['federal_funds_rate', m, REF]].dropna()
                e1, e0 = (s[m] - s.federal_funds_rate).abs(), (s[REF] - s.federal_funds_rate).abs()
                if len(s) > 5 and (e1 - e0).abs().sum() > 0:
                    r['t_p'] = stats.ttest_rel(e1, e0).pvalue; r['wil_p'] = stats.wilcoxon(e1, e0).pvalue
                    r['vs_ref'] = f"{(e0.mean() - e1.mean()) / e0.mean() * 100:+.0f}%{star(r['t_p'])}"
            rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(F.replace('predictions', 'metrics'), index=False)
pd.set_option('display.width', 250)
for sname in SPLITS:
    print(f'\n===== {sname}: member-level MAE by horizon (lower is better); vs_ref = MAE improvement over B0 with t-test stars')
    S = R[R.split == sname]
    piv = S.pivot(index='method', columns='h', values='mem_MAE')[['all', '0', '1', '2', '3', 'LR']].round(3)
    piv['n'] = S[S.h == 'all'].set_index('method').mem_n
    piv['RMSE'] = S[S.h == 'all'].set_index('method').mem_RMSE.round(3)
    piv['Corr'] = S[S.h == 'all'].set_index('method').mem_Corr.round(3)
    piv['com_MAE'] = S[S.h == 'all'].set_index('method').com_MAE.round(3)
    piv['vs_B0'] = S[S.h == 'all'].set_index('method').get('vs_ref')
    print(piv.sort_values('all').to_string())
