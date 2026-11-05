"""Seventh batch: time-series models (agreed at the 11.5 meeting). The participant's previous dot (lag) is his/her own dot
for the same calendar year at the previous SEP. One model per horizon group (Y0, Y1, Y2-Y3, LR) unless noted.
  B17  heterogeneous AR(1): r = (a + a_i) + (rho + rho_i) lag + e. Participant intercepts and slopes are random effects
       (partial pooling, REML); each participant's effects are updated with all his/her dots before the meeting
  B18  heterogeneous ARX: B17 + news since the previous SEP (policy rate, Treasury forward and SPF bill-rate path for the end
       of year H, core inflation, unemployment, GDP growth) + gaps of the lag to the previous committee median and the policy rate
  B19  VAR(p) of the participant's five SEP variables for the same calendar year, pooled over participants, p in {1, 2} by BIC;
       the forecast is the fed funds equation (LR: fed funds, GDP growth and unemployment; LR inflation is always 2)
  B19b VAR across horizons: fed funds dot on the participant's previous dots for year H, year H+1 and the longer run
  B20  panel VAR by GMM (pvar.R, R package panelvar): unit = participant x calendar year, five SEP variables, horizon dummies;
       LR: unit = participant, three variables. Unit effects: past residuals of the unit, shrunk toward the participant's
       average
  B21  panel VARX: B20 + the news of B18
Rows without an own previous dot for the same year keep the naive forecast (B0). Parameters: expanding window, or fixed
after FEDAI_TRAIN_END (protocol.py); participant and unit effects keep updating. Forecasts are floored at 0.125."""
import os, shutil, subprocess, tempfile, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from protocol import TRAIN_END, fitset
import news
warnings.filterwarnings('ignore')
V6 = pd.read_csv('data/predictions_v6.csv', parse_dates=['meeting_date'])
D = news.panel(V6); ZLB = news.ZLB
meetings = sorted(D.meeting_date.unique())
GROUPS = (['0'], ['1'], ['2', '3'], ['LR'])
X18 = news.NEWS + ['gap_med', 'gap_now']
preds = {}
def put(n, idx, v): preds.setdefault(n, pd.Series(np.nan, index=D.index)).loc[idx] = v

# ---------- B17, B18: random-coefficient (mixed) models with frozen hyperparameters and updated participant effects
def design(df, cols): return np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
def independent(tr, cols):
    """Columns that add rank to the design (with a constant), in order. In 2012-2015 the policy rate never moved, so its
    change is constant and its gap to the lag is collinear with the lag; such columns are dropped."""
    keep, X = [], np.ones((len(tr), 1))
    for c in cols:
        Z = np.column_stack([X, tr[c].values])
        if np.linalg.matrix_rank(Z, tol=1e-8 * len(tr) * np.abs(Z).max()) > X.shape[1]: keep.append(c); X = Z
    return keep
def mixed_fit(tr, X):
    keep = independent(tr, X)
    for q in (2, 1):                                               # random intercept and slope; else random intercept only
        try:                                                       # variances near zero sit on the boundary: optimizers may
            r = sm.MixedLM(tr.ffr.values, design(tr, keep), groups=tr.person_id.values, exog_re=design(tr, ['lag'])[:, :q]) \
                  .fit(reml=True, method=['lbfgs', 'bfgs', 'powell', 'nm'])  # report no convergence; accept finite, valid estimates
            G = np.atleast_2d(np.asarray(r.cov_re))
            if np.all(np.isfinite(r.fe_params)) and np.all(np.isfinite(G)) and np.linalg.eigvalsh(G).min() > -1e-8 and r.scale > 0:
                return dict(keep=keep, beta=np.asarray(r.fe_params), G=G, s2=r.scale, q=q)
        except Exception: pass
def mixed_predict(f, hist, cur):
    out = design(cur, f['keep']) @ f['beta']
    res = hist.ffr.values - design(hist, f['keep']) @ f['beta']
    Z, Zc, Gi = design(hist, ['lag'])[:, :f['q']], design(cur, ['lag'])[:, :f['q']], np.linalg.pinv(f['G'])
    for p in cur.person_id.unique():                                # posterior mean of the participant's effects
        m = (hist.person_id == p).values
        if m.any():
            b = np.linalg.solve(Z[m].T @ Z[m] / f['s2'] + Gi, Z[m].T @ res[m] / f['s2'])
            c = (cur.person_id == p).values; out[c] += Zc[c] @ b
    return out

# ---------- B19, B19b: pooled VAR equations for the fed funds dot
V5, VLR = ['ffr', 'gdp', 'ur', 'pce', 'core'], ['ffr', 'gdp', 'ur']
def ols(tr, cols):
    X = design(tr, cols); b = np.linalg.lstsq(X, tr.ffr.values, rcond=None)[0]
    rss = ((tr.ffr.values - X @ b) ** 2).sum(); n = len(tr)
    return b, n * np.log(rss / n) + X.shape[1] * np.log(n)
D['next_f'] = D.next_l1.fillna(D.ffr_l1); D['lr_f'] = D.lr_l1.fillna(D.ffr_l1); D['last_f'] = D.last_l1.fillna(D.ffr_l1)

cache = {}
for t in meetings:
    cur = D[D.meeting_date == t]; hist = D[(D.meeting_date < t) & D.has_lag]
    if hist.meeting_date.nunique() < 6: continue
    for hs in GROUPS:
        g = tuple(hs); hg = hist[hist.h.isin(hs)]; tr = fitset(hg); te = cur[cur.h.isin(hs)]
        if len(tr) < 30 or te.empty: continue
        base = te.lag.values
        for name, X in [('B17_hetero_ar1', ['lag']), ('B18_hetero_arx', ['lag'] + X18)]:
            key = (name, g, len(tr))
            if key not in cache: cache[key] = mixed_fit(tr, X)
            f = cache[key]
            if f is not None:
                p = mixed_predict(f, hg, te.fillna({c: 0.0 for c in X18}))
                put(name, te.index, np.maximum(ZLB, np.where(te.has_lag, p, base)))
        vs = VLR if hs == ['LR'] else V5
        key = ('B19', g, len(tr))
        if key not in cache:
            l1 = [v + '_l1' for v in vs]; l2 = l1 + [v + '_l2' for v in vs]
            both = tr.dropna(subset=l2)
            p = 2 if len(both) >= 30 and ols(both, l2)[1] < ols(both, l1)[1] else 1
            cache[key] = (p, l1, l2, ols(tr, l1)[0], ols(both, l2)[0] if p == 2 else None)
        p, l1, l2, b1, b2 = cache[key]
        f1 = design(te.fillna({c: 0.0 for c in l1}), l1) @ b1
        if p == 2:
            ok2 = te[l2].notna().all(axis=1).values
            f1 = np.where(ok2, design(te.fillna({c: 0.0 for c in l2}), l2) @ b2, f1)
        put('B19_var_sep_variables', te.index, np.maximum(ZLB, np.where(te.has_lag, f1, base)))
        cols = ['ffr_l1', 'last_f'] if hs == ['LR'] else ['ffr_l1', 'next_f', 'lr_f']
        key = ('B19b', g, len(tr))
        if key not in cache: cache[key] = ols(tr.dropna(subset=cols), cols)[0]
        fb = design(te.fillna({c: 0.0 for c in cols}), cols) @ cache[key]
        put('B19b_var_horizons', te.index, np.maximum(ZLB, np.where(te.has_lag, fb, base)))
    print(t.date(), end=' ', flush=True)
print()

# ---------- B20, B21: panel VAR / VARX by GMM in R
RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')
CODE = os.path.dirname(os.path.abspath(__file__))
D['time'] = D.sep_index + 1
NEWSX = ['d_pol', 'd_fwd', 'd_spf', 'd_core_inflation', 'd_unrate', 'd_gdp_growth_y']
for k in ['1', '2', '3']: D['h' + k] = (D.h == k).astype(float)
cal = D[D.h != 'LR'].copy(); cal['unit'] = pd.factorize(cal.person_id + '_' + cal.proj_year.astype(str))[0] + 1
lr = D[D.h == 'LR'].copy(); lr['unit'] = pd.factorize(lr.person_id)[0] + 1
n_seps = len(meetings)
frozen = None if TRAIN_END is None else int(D[D.meeting_date <= TRAIN_END].time.max())
cuts = list(range(6, (frozen or n_seps - 1) + 1))                 # model for SEP time s+1: rows with time <= s (fixed after TRAIN_END)
def pvar(panel, dep, ex, name, tag):
    with tempfile.TemporaryDirectory() as tmp:
        cols = ['unit', 'time'] + dep + ex
        panel[cols].fillna(0.0).to_csv(f'{tmp}/p.csv', index=False)
        subprocess.run([RSCRIPT, f'{CODE}/pvar.R', f'{tmp}/p.csv', f'{tmp}/c.csv', ','.join(dep), ','.join(ex) or '-'] + [str(c) for c in cuts],
                       check=True, capture_output=True)
        C = pd.read_csv(f'{tmp}/c.csv')
    C.assign(model=name, system=tag).to_csv(f'data/pvar_coef_{name}_{tag}.csv', index=False)
    C = C[C.equation == 'fod_ffr']
    out = pd.Series(np.nan, index=panel.index)
    for s in range(1, n_seps):
        b = C[C.cutoff == (min(s, frozen) if frozen else s)].set_index('term').value
        if b.empty: continue
        def fit(df):
            f = b.get('const', 0.0) + sum(b[f'fod_lag1_{v}'] * df[f'{v}_l1'].fillna(0.0) for v in dep)
            return f + sum(b.get(f'fod_{x}', 0.0) * df[x].fillna(0.0) for x in ex)
        te = panel[panel.time == s + 1]; hi = panel[(panel.time <= s) & panel.has_lag]
        if te.empty: continue
        res = hi.ffr - fit(hi)
        pp = res.groupby(hi.person_id).agg(['sum', 'count']); mu_p = (pp['sum'] / (pp['count'] + 3))
        uu = res.groupby(hi.unit).agg(['sum', 'count'])
        mp = te.person_id.map(mu_p).fillna(0.0)
        mu = (te.unit.map(uu['sum']).fillna(0.0) + 3 * mp) / (te.unit.map(uu['count']).fillna(0.0) + 3)
        out.loc[te.index] = np.where(te.has_lag, fit(te) + mu, te.lag)
    return out
if shutil.which(RSCRIPT) is None: print(f'B20/B21 skipped: {RSCRIPT} not found (R with the panelvar package; set RSCRIPT)')
for name, ex in ([('B20_panel_var', []), ('B21_panel_varx', NEWSX)] if shutil.which(RSCRIPT) else []):
    a = pvar(cal, V5, ['h1', 'h2', 'h3'] + ex, name, 'calendar'); b = pvar(lr, VLR, ex, name, 'LR')
    put(name, pd.concat([a, b]).index, np.maximum(ZLB, pd.concat([a, b]).values))
    print(name, 'done', flush=True)

out = V6.copy()
for k, v in preds.items(): out[k] = v
out.to_csv('data/predictions_v7.csv', index=False)
print({k: int(v.notna().sum()) for k, v in preds.items()})
