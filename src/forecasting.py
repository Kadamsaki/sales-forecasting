"""Reusable forecasting code: data prep, decomposition, SARIMA, Prophet, seasonal-naive baseline, backtesting."""
import itertools, logging, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

PERIODS = {"D": 7, "W": 52, "MS": 12, "QS": 4}
LABELS = {"D": "daily", "W": "weekly", "MS": "monthly", "QS": "quarterly"}


# ---------------------------------------------------------------- data prep
def prepare(df, date_col, value_col):
    """Return (series, info). Parses dates, detects frequency, aggregates duplicates, fills small gaps."""
    d = df[[date_col, value_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    bad = int(d.isna().any(axis=1).sum())
    d = d.dropna().sort_values(date_col)
    if len(d) < 24:
        raise ValueError("Need at least 24 valid rows to forecast.")
    gap = d[date_col].drop_duplicates().diff().dt.days.median()
    freq = "D" if gap <= 1.5 else "W" if gap <= 8 else "MS" if gap <= 32 else "QS" if gap <= 95 else None
    if freq is None:
        raise ValueError("Could not detect a daily/weekly/monthly/quarterly frequency.")
    s = d.set_index(date_col)[value_col].resample(freq).sum(min_count=1)   # sums duplicates / sub-period rows
    filled = int(s.isna().sum())
    s = s.interpolate(limit_direction="both")
    s.index.freq = pd.tseries.frequencies.to_offset(freq) if s.index.freq is None else s.index.freq
    info = dict(freq=freq, freq_label=LABELS[freq], period=PERIODS[freq], n=len(s), filled_gaps=filled,
                dropped_rows=bad, start=s.index[0], end=s.index[-1], can_log=bool((s > 0).all()))
    return s, info


def decompose(s, period):
    """Robust STL decomposition -> DataFrame(trend, seasonal, resid) + seasonal strength (0-1)."""
    r = STL(s, period=period, robust=True).fit()
    out = pd.DataFrame({"observed": s, "trend": r.trend, "seasonal": r.seasonal, "resid": r.resid})
    strength = max(0.0, 1 - out.resid.var() / (out.seasonal + out.resid).var())
    return out, float(strength)


def adf_report(s):
    """Augmented Dickey-Fuller on the series and its first difference."""
    out = {}
    for name, x in [("level", s), ("first difference", s.diff().dropna())]:
        p = adfuller(x, autolag="AIC")[1]
        out[name] = dict(p_value=round(float(p), 4), stationary=bool(p < 0.05))
    return out


# ---------------------------------------------------------------- metrics
def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return dict(MAPE=float(np.mean(np.abs((a - p) / a)) * 100), RMSE=float(np.sqrt(np.mean((a - p) ** 2))),
                MAE=float(np.mean(np.abs(a - p))))


# ---------------------------------------------------------------- models
def _future_index(s, h):
    return pd.date_range(s.index[-1], periods=h + 1, freq=s.index.freq)[1:]


def seasonal_naive(s, h, period, alpha=0.05, **_):
    """Baseline: repeat the last observed season. Interval from in-sample seasonal-naive errors (normal approx)."""
    reps = int(np.ceil(h / period)); vals = np.tile(s.values[-period:], reps)[:h]
    err = (s.values[period:] - s.values[:-period]); sd = np.std(err)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2); k = np.sqrt(np.ceil(np.arange(1, h + 1) / period))
    idx = _future_index(s, h)
    return pd.DataFrame({"yhat": vals, "lower": vals - z * sd * k, "upper": vals + z * sd * k}, index=idx)


# small, fast candidate grid (selected by AIC)
_ORDERS = [(1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 1, 0)]
_SEAS = [(0, 1, 1), (1, 1, 1)]


def sarima(s, h, period, alpha=0.05, use_log=True, return_order=False, **_):
    """SARIMA with order chosen by AIC over a small grid. Optional log transform (stabilises growing seasonality)."""
    y = np.log(s) if use_log else s
    seas = _SEAS if period > 1 else [(0, 0, 0)]
    best = None
    for o, so in itertools.product(_ORDERS, seas):
        try:
            m = SARIMAX(y, order=o, seasonal_order=so + (period,) if period > 1 else (0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
            if best is None or m.aic < best[0]: best = (m.aic, m, o, so)
        except Exception:
            continue
    _, m, o, so = best
    fc = m.get_forecast(h); ci = fc.conf_int(alpha=alpha)
    f = pd.DataFrame({"yhat": fc.predicted_mean.values, "lower": ci.iloc[:, 0].values, "upper": ci.iloc[:, 1].values},
                     index=_future_index(s, h))
    if use_log: f = np.exp(f)
    return (f, {"order": o, "seasonal_order": so + (period,), "aic": round(m.aic, 1)}) if return_order else f


def prophet_fc(s, h, period, alpha=0.05, **_):
    from prophet import Prophet
    freq = s.index.freq.name if hasattr(s.index.freq, "name") else "MS"
    freq = {"MS": "MS", "QS-JAN": "QS", "W-SUN": "W"}.get(freq, freq)
    m = Prophet(seasonality_mode="multiplicative", interval_width=1 - alpha, yearly_seasonality=bool((s.index[-1] - s.index[0]).days >= 730),
                weekly_seasonality=(freq == "D"), daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": s.index, "y": s.values}))
    fut = m.make_future_dataframe(h, freq=freq, include_history=False)
    p = m.predict(fut).set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
    p.columns = ["yhat", "lower", "upper"]
    return p


MODELS = {"Seasonal Naive": seasonal_naive, "SARIMA": sarima, "Prophet": prophet_fc}


# ---------------------------------------------------------------- backtest
def backtest(s, period, horizon=12, folds=3, alpha=0.05, models=None):
    """Rolling-origin (expanding window) backtest. Returns long DataFrame of per-fold metrics + interval coverage."""
    rows = []
    for k in range(folds, 0, -1):
        cut = len(s) - k * horizon
        train, test = s.iloc[:cut], s.iloc[cut:cut + horizon]
        train = train.copy(); train.index.freq = s.index.freq
        for name in (models or MODELS):
            f = MODELS[name](train, len(test), period, alpha)
            m = metrics(test.values, f.yhat.values)
            m.update(model=name, fold=folds - k + 1, test_start=test.index[0].date(), test_end=test.index[-1].date(),
                     coverage=float(((test.values >= f.lower.values) & (test.values <= f.upper.values)).mean() * 100))
            rows.append(m)
    return pd.DataFrame(rows)
