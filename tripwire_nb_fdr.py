
# (same content as previously created, with stable NB/Poisson fallbacks)
from __future__ import annotations
import argparse, os, math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd, numpy as np
try:
    import mpmath as mp
    _HAS_MPMATH = True
except Exception:
    _HAS_MPMATH = False

def _auto_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key.lower() in cols:
            return cols[key.lower()]
    for c in df.columns:
        cl = c.lower()
        if any(key in cl for key in candidates):
            return c
    return None

def _parse_period(series: pd.Series) -> pd.PeriodIndex:
    s = series.copy()
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.astype('period[Q]')
    def to_period(x):
        if pd.isna(x):
            return pd.NaT
        xs = str(x).strip()
        import re
        m = re.match(r'^\s*(\d{4})[\-\s]*Q?([1-4])\s*$', xs, flags=re.IGNORECASE)
        if m:
            y, q = int(m.group(1)), int(m.group(2))
            return pd.Period(freq='Q', year=y, quarter=q)
        m = re.match(r'^\s*Q([1-4])[\-\s]*(\d{4})\s*$', xs, flags=re.IGNORECASE)
        if m:
            q, y = int(m.group(1)), int(m.group(2))
            return pd.Period(freq='Q', year=y, quarter=q)
        try:
            dt = pd.to_datetime(xs, errors='raise')
            return dt.to_period('Q')
        except Exception:
            return pd.NaT
    periods = s.map(to_period)
    if periods.isna().any():
        try:
            tmp = pd.PeriodIndex(s.astype(str), freq='Q')
            return tmp
        except Exception:
            pass
    return periods.astype('period[Q]')

def _ensure_period(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if isinstance(df[date_col].dtype, pd.PeriodDtype):
        return df
    lower = {c.lower(): c for c in df.columns}
    if 'year' in lower and ('quarter' in lower or 'q' in lower):
        ycol = lower['year']; qcol = lower.get('quarter', lower.get('q'))
        per = pd.PeriodIndex(year=df[ycol].astype(int), quarter=df[qcol].astype(int), freq='Q')
        df = df.copy(); df['__period__'] = per
        return df.rename(columns={'__period__': date_col})
    df = df.copy(); df[date_col] = _parse_period(df[date_col]); return df

def _poisson_tail_by_sum(x: int, mu: float, max_terms: int = 200000) -> float:
    if x <= 0: return 1.0
    def log_pmf(k): return -mu + k * math.log(mu) - math.lgamma(k + 1)
    pmf = math.exp(log_pmf(x)); total = pmf; k = x
    for _ in range(max_terms):
        pmf = pmf * mu / (k + 1); total += pmf; k += 1
        if pmf < 1e-15 * total: break
    return float(total)

def _poisson_tail_sf(x: int, mu: float) -> float:
    if x <= 0: return 1.0
    if mu <= 0: return 0.0 if x > 0 else 1.0
    if _HAS_MPMATH:
        try:
            num = mp.gammainc(x, mu, mp.inf); den = mp.gamma(x)
            val = float(num / den); 
            return 0.0 if val < 0 else val
        except Exception: pass
    return _poisson_tail_by_sum(x, mu)

def _nb_tail_by_sum(x: int, mu: float, k: float, max_terms: int = 100000) -> float:
    r = k; p = mu / (mu + k)
    if p <= 0.0: return 0.0 if x > 0 else 1.0
    if p >= 1.0: return 1.0 if x <= 0 else 0.0
    def log_pmf(j):
        return (math.lgamma(j + r) - math.lgamma(r) - math.lgamma(j + 1)
                + r * math.log(1 - p) + j * math.log(p))
    pmf = math.exp(log_pmf(x)); total = pmf; j = x
    for _ in range(max_terms):
        pmf = pmf * p * (j + r) / (j + 1); total += pmf; j += 1
        if pmf < 1e-15 * total: break
    return float(total)

def _nb_tail_sf(x: int, mu: float, k: float) -> float:
    if x <= 0: return 1.0
    if mu <= 0: return 0.0 if x > 0 else 1.0
    if not np.isfinite(k) or k <= 0: return _poisson_tail_sf(x, mu)
    if not _HAS_MPMATH: return _nb_tail_by_sum(x, mu, k)
    r = k; p = mu / (mu + k)
    if p <= 0.0: return 0.0 if x > 0 else 1.0
    if p >= 1.0: return 1.0 if x <= 0 else 0.0
    z = 1 - p
    try:
        I = mp.betainc(r, x, 0, z, regularized=True)
        sf = 1.0 - float(I)
        return 0.0 if sf < 0 else sf
    except Exception:
        return _nb_tail_by_sum(x, mu, k)

def _bh_fdr(pvals: pd.Series, alpha: float):
    s = pvals.copy()
    valid = s.notna()
    m = valid.sum()
    if m == 0:
        return pd.Series(index=s.index, dtype=float), pd.Series(False, index=s.index)
    order = s.loc[valid].sort_values().index
    ranks = pd.Series(range(1, m + 1), index=order, dtype=float)
    q = pd.Series(np.nan, index=s.index, dtype=float)
    q.loc[order] = (s.loc[order] * m / ranks)
    q_sorted = q.loc[order][::-1].cummin()[::-1]
    q.loc[order] = q_sorted
    flags = (q <= alpha) & valid
    return q, flags

from dataclasses import dataclass
@dataclass
class Config:
    alpha: float = 0.10
    lookback: int = 12
    min_history: int = 8
    min_count: Optional[int] = None
    front_col: Optional[str] = None
    date_col: Optional[str] = None
    count_col: Optional[str] = None

def _fit_baseline_and_pvalue(hist: np.ndarray, x_obs: int):
    eps = 1e-12; n = len(hist)
    if n == 0: return (np.nan, np.nan, np.nan, 'NA', np.nan)
    mu = float(np.mean(hist))
    if mu <= eps: 
        pval = 1.0 if x_obs <= 0 else 0.0
        return (mu, np.inf, pval, 'Zero', 0.0)
    var = float(np.var(hist, ddof=1)) if n >= 2 else float(np.var(hist, ddof=0))
    if var <= mu * (1 + 1e-9):
        pval = _poisson_tail_sf(int(x_obs), mu)
        return (mu, np.inf, pval, 'Poisson', var)
    k = mu * mu / max(var - mu, eps)
    pval = _nb_tail_sf(int(x_obs), mu, k)
    return (mu, k, pval, 'NB', var)

def run_tripwire(timeseries_path: str, out_path: str, cfg: Config) -> pd.DataFrame:
    ext = os.path.splitext(timeseries_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(timeseries_path)
    elif ext in ('.parquet', '.pq'):
        import pyarrow.parquet as pq; df = pq.read_table(timeseries_path).to_pandas()
    else:
        raise ValueError("Unsupported file type. Use CSV or Parquet.")
    front_col = cfg.front_col or _auto_col(df, ['lineage_id','front_id','front','community_id','community','front_uid'])
    date_col  = cfg.date_col  or _auto_col(df, ['period','quarter','date','dt','time','year_q'])
    count_col = cfg.count_col or _auto_col(df, ['new_works','new_count','count_new','n_new','new','works_new','count'])
    if not front_col or not date_col or not count_col:
        raise ValueError(f"Column detection failed. Found front_col={front_col}, date_col={date_col}, count_col={count_col}")

    # Store original names for output consistency
    orig_front_col = front_col
    orig_date_col = date_col

    df = df[[front_col, date_col, count_col]].copy()
    df.rename(columns={front_col:'front_id', date_col:'period', count_col:'count'}, inplace=True)
    df = _ensure_period(df, 'period')
    if not isinstance(df['period'].dtype, pd.PeriodDtype):
        raise ValueError("Could not parse 'period' to quarterly Period dtype.")
    idx = pd.MultiIndex.from_product([df['front_id'].unique(),
                                      pd.period_range(df['period'].min(), df['period'].max(), freq='Q')],
                                     names=['front_id','period'])
    df = df.set_index(['front_id','period']).reindex(idx).sort_index()
    if 'count' not in df: df['count'] = np.nan
    df['count'] = df['count'].fillna(0).astype(int); df = df.reset_index()
    rows = []; L = int(cfg.lookback)
    for fid, g in df.groupby('front_id', sort=False):
        g = g.sort_values('period').reset_index(drop=True)
        y = g['count'].values.astype(int); periods = g['period'].values
        for i in range(len(g)):
            start = max(0, i - L)
            hist_window = y[start:i].astype(int)
            if cfg.min_count is not None:
                hist_filtered = hist_window[hist_window >= cfg.min_count]
            else:
                hist_filtered = hist_window
            n_hist = int(hist_filtered.size)
            x_obs = int(y[i])

            if cfg.min_count is not None and x_obs < cfg.min_count:
                mu_hat = k_hat = pval = var_hat = np.nan
                model_type = f'<min({cfg.min_count})'
            elif n_hist >= cfg.min_history:
                mu_hat, k_hat, pval, model_type, var_hat = _fit_baseline_and_pvalue(hist_filtered, x_obs)
            else:
                mu_hat = k_hat = pval = var_hat = np.nan
                model_type = 'NA'

            rr = (x_obs / mu_hat) if (mu_hat and mu_hat > 0 and math.isfinite(mu_hat)) else np.nan
            excess = (x_obs - mu_hat) if (mu_hat and math.isfinite(mu_hat)) else np.nan
            rows.append({'front_id': fid,'period': periods[i],'count': x_obs,'n_history': n_hist,
                         'mu_hat': mu_hat,'var_hat': var_hat,'k_hat': k_hat,'model': model_type,
                         'p_value': pval,'rr_obs_over_mu': rr,'excess_obs_minus_mu': excess})
    out = pd.DataFrame(rows)
    out['q_value'] = np.nan; out['alert'] = False
    for per, gg in out.groupby('period'):
        q, flags = _bh_fdr(gg['p_value'], cfg.alpha)
        out.loc[gg.index, 'q_value'] = q.values; out.loc[gg.index, 'alert'] = flags.values
    out['alpha'] = cfg.alpha; out['lookback'] = cfg.lookback; out['min_history'] = cfg.min_history

    model_str = out['model'].astype(str)
    has_model = (
        out['model'].notna()
        & ~model_str.isin(['NA', 'Zero'])
        & ~model_str.str.startswith('<min', na=False)
    )
    meaningful = has_model
    filtered = out.loc[meaningful].copy()
    filtered['period'] = filtered['period'].astype(str)

    # Rename back to original column names for output consistency
    filtered.rename(columns={'front_id': orig_front_col, 'period': orig_date_col}, inplace=True)

    filtered.to_csv(out_path, index=False)
    return filtered

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeseries', required=True)
    ap.add_argument('--out', default='alerts_tripwire.csv')
    ap.add_argument('--alpha', type=float, default=0.10)
    ap.add_argument('--lookback', type=int, default=12)
    ap.add_argument('--min-history', dest='min_history', type=int, default=8)
    ap.add_argument('--min-count', dest='min_count', type=int, default=None)
    ap.add_argument('--front-col', dest='front_col', default=None)
    ap.add_argument('--date-col', dest='date_col', default=None)
    ap.add_argument('--count-col', dest='count_col', default=None)
    args = ap.parse_args()
    cfg = Config(alpha=args.alpha, lookback=args.lookback, min_history=args.min_history, min_count=args.min_count,
                 front_col=args.front_col, date_col=args.date_col, count_col=args.count_col)
    out = run_tripwire(args.timeseries, args.out, cfg)
    print(f"Wrote {args.out} with {len(out)} rows and {out['alert'].sum()} alerts at alpha={args.alpha}.")
if __name__ == '__main__':
    main()

