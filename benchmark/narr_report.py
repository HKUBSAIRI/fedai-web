"""Tables for the narrative benchmark: overall, by question type, paired tests, judge checks. Writes data/narr_scores.csv and data/narr_summary.csv.
Usage: python narr_report.py [--from 2019-01-01] [--to 2020-12-31]"""
import argparse, numpy as np, pandas as pd
from scipy import stats
import narr_bench as nb
ap = argparse.ArgumentParser(); ap.add_argument('--from', dest='date_from', default='2019-01-01'); ap.add_argument('--to', dest='date_to', default='2020-12-31')
a = ap.parse_args(); tag = f'{a.date_from[:4]}-{a.date_to[:4]}'
S = nb.scored(a)
S = S[S.final.notna()]
ORDER = ['N0', 'N1', 'L_ctx', 'L_prev', 'L_named', 'ORACLE']
NAME = {'N0': 'N0 own previous answer', 'N1': 'N1 committee answer at previous SEP', 'L_ctx': 'L_ctx AI: material only (anonymous)',
        'L_prev': 'L_prev AI: material + own previous answer (anonymous)', 'L_named': 'L_named AI: + name and date', 'ORACLE': 'ORACLE peers, same meeting (not available ex ante)'}
W = S.pivot_table(index='key', columns='method', values='final')
common = W.dropna(subset=[m for m in ORDER if m in W]).index          # rows every method answered (N0 needs a previous answer)
C = S[S.key.isin(common)]
print(f'{tag}: {S.key.nunique()} reference answers; {len(common)} with every method (N0 needs an earlier answer to the same question)\n')
T = C.groupby('method')[['final', 'position', 'coverage', 'no_contradiction', 'pred_words']].mean().reindex(ORDER)
T.insert(0, 'n', C.groupby('method').size()); T.index = [NAME[i] for i in T.index]
print(T.round(3).to_string()); print('reference answer words (mean):', round(C.drop_duplicates('key').ref_words.mean()))
def paired(x, y):
    d = (W.loc[common, x] - W.loc[common, y]).dropna()
    return f'{d.mean():+.3f} (t p={stats.ttest_1samp(d, 0).pvalue:.3f}, W p={stats.wilcoxon(d).pvalue:.3f})'
print('\nPaired differences in final score:')
tests = {}
for x, y in [('L_prev', 'N0'), ('L_named', 'N0'), ('L_named', 'L_prev'), ('L_prev', 'L_ctx'), ('L_ctx', 'N1'), ('N0', 'N1'), ('ORACLE', 'N0')]:
    tests[f'{x} - {y}'] = paired(x, y); print(f'  {x:8s} - {y:6s} {tests[f"{x} - {y}"]}')
print('\nFinal score by question type:')
B = C.pivot_table(index='type', columns='method', values='final', aggfunc='mean')[ORDER]
B.insert(0, 'n', C[C.method == 'N0'].groupby('type').size()); print(B.round(3).to_string())
print('\nFinal score by question (family):')
F = C.pivot_table(index='family', columns='method', values='final', aggfunc='mean')[ORDER]; print(F.round(3).to_string())
# judge checks
J = nb.Cache('data/narr_judge.jsonl').d
chk = {}
for m in ('SELF', 'SHUFFLE'):
    v = [nb.score_one(o['eval'])[3] for k, o in J.items() if k.endswith(f'|{m}|{nb.JUDGE_MODEL}') and k[:4] >= a.date_from[:4] and k[:4] <= a.date_to[:4]]
    v = [x for x in v if x is not None]
    if v: chk[m] = (len(v), float(np.mean(v)))
second = 'gpt-5-mini'; pairs = []
for k, o in J.items():
    if k.endswith(f'|{second}') and a.date_from[:4] <= k[:4] <= a.date_to[:4]:
        base = k[: -len(second)] + nb.JUDGE_MODEL
        if base in J: pairs.append((nb.score_one(J[base]['eval'])[3], nb.score_one(o['eval'])[3], J[base]['eval'].get('position_match'), o['eval'].get('position_match')))
print('\nJudge checks:')
for m, (n, v) in chk.items(): print(f'  {m}: n={n}, mean final {v:.3f}')
if pairs:
    P = pd.DataFrame(pairs, columns=['a', 'b', 'pa', 'pb']).dropna()
    print(f'  second judge {second} on {len(P)} answers: Pearson {P.a.corr(P.b):.2f}, Spearman {P.a.corr(P.b, method="spearman"):.2f}, '
          f'same position label {np.mean(P.pa == P.pb):.0%}, mean {P.a.mean():.3f} vs {P.b.mean():.3f}')
    chk['second_judge'] = (len(P), P.a.corr(P.b))
S.to_csv(f'data/narr_scores_{tag}.csv', index=False)
T.round(4).to_csv(f'data/narr_summary_{tag}.csv')
B.round(4).to_csv(f'data/narr_by_type_{tag}.csv')
