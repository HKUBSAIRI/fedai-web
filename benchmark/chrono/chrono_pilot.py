"""ChronoGPT pilot: the model trained only on data up to 2015-12-31 forecasts the committee median dot for each target year at
the four 2016 SEPs (all after its cutoff, so no memory). Two read-outs: free generation, and likelihood choice among
candidates around the previous median (13 values in 0.125 steps). Pre-meeting data only (ALFRED vintages)."""
import re, json, time, numpy as np, pandas as pd, torch
from load import load, tok
YEAR = 2015; TEST_YEAR = YEAR + 1
B = '/home/claude-dev/workspaces/fedai-web/.scratch/bench/data/'
P = pd.read_csv(B + 'predictions_v6.csv', parse_dates=['meeting_date'])
macro = pd.read_csv(B + 'macro.csv', parse_dates=['meeting_date']).set_index('meeting_date')
meetings = sorted(P.meeting_date.unique()); TEST = [m for m in meetings if m.year == TEST_YEAR]
SYS = "You are ChronoGPT, a large language model trained by ManelaLab at WashU.\n    Below is an instruction that describes a task.\n    Write a response that appropriately completes the request."
def wrap(q): return f"\n\n### Instruction:\n{SYS}\n{q}\n\n### Input:\n### Response:\n"
m = load(YEAR)
def logprob(prefix, answer):
    a = tok.encode(prefix); b = tok.encode(answer)
    x = torch.tensor([a + b]); lp = torch.log_softmax(m(x)[0].float(), -1)
    return float(sum(lp[len(a) + i - 1, t] for i, t in enumerate(b)))
def gen(prefix, n=10):
    ids = tok.encode(prefix)
    for _ in range(n):
        nxt = int(torch.argmax(m(torch.tensor([ids]))[0, -1]))
        if nxt == 50256: break
        ids.append(nxt)
    return tok.decode(ids[len(tok.encode(prefix)):])
rows = []
for t in TEST:
    i = meetings.index(t); p = meetings[i - 1]; d = macro.loc[t]
    prev = P[P.meeting_date == p].groupby('proj_year').federal_funds_rate.median()
    true = P[P.meeting_date == t].groupby('proj_year').federal_funds_rate.median()
    for y in [x for x in true.index if x != -1]:
        pm = prev.get(y, prev[prev.index > 0].iloc[-1])
        q = (f"Data available before the FOMC meeting of {t:%B %Y}: the federal funds target range is {d.target_lower:.2f} to {d.target_upper:.2f} percent, "
             f"unemployment is {d.unrate:.1f} percent and core PCE inflation is {d.core_inflation:.1f} percent. At the previous meeting ({p:%B %Y}) the median FOMC "
             f"participant projected the federal funds rate at {pm:.3f} percent for the end of {y}. What will the median projection for the end of {y} be at this meeting? "
             f"Answer with one number in percent.")
        pre = wrap(q); t0 = time.time()
        g = gen(pre, 10); num = re.findall(r'\d+\.\d+|\d+', g)
        cands = [round(pm + k * 0.125, 3) for k in range(-6, 7) if pm + k * 0.125 >= 0.124]
        lps = [logprob(pre, f'{c:.3f}%') for c in cands]
        rows.append({'meeting_date': str(t.date()), 'proj_year': int(y), 'prev_median': pm, 'true_median': true[y],
                     'gen_text': g.strip(), 'gen_value': float(num[0]) if num else np.nan, 'lik_value': cands[int(np.argmax(lps))], 'secs': round(time.time() - t0)})
        print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(f'chrono_pilot_{TEST_YEAR}.csv', index=False)
for c in ['prev_median', 'gen_value', 'lik_value']:
    print(c, 'MAE', round(np.nanmean(np.abs(R[c] - R.true_median)), 3), 'n', int(R[c].notna().sum()))
