"""Sales Demand Forecasting Web App.  Run: streamlit run app.py"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import forecasting as F

st.set_page_config(page_title="Sales Demand Forecasting", layout="wide")
st.title("Sales Demand Forecasting")
st.caption("Upload a sales time series (date + value), pick a model, and get a forecast with confidence intervals.")

SAMPLE = Path(__file__).resolve().parent / "data" / "retail_sales.csv"

# ---------------- sidebar: data + settings
with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload CSV", type="csv")
    use_sample = st.checkbox("Use sample data (US monthly retail sales)", value=up is None)
    st.header("2. Forecast settings")
    model_name = st.selectbox("Model", ["SARIMA", "Prophet", "Seasonal Naive"])
    horizon = st.slider("Forecast horizon (periods)", 3, 36, 12)
    conf = st.select_slider("Confidence level", options=[80, 90, 95, 99], value=95)

raw = pd.read_csv(up) if up is not None else (pd.read_csv(SAMPLE) if use_sample else None)
if raw is None:
    st.info("Upload a CSV or tick *Use sample data* in the sidebar to begin."); st.stop()

c1, c2 = st.columns(2)
guess_d = next((c for c in raw.columns if "date" in c.lower() or c.lower() in ("ds", "month", "time")), raw.columns[0])
num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
if not num_cols: st.error("No numeric column found for the sales values."); st.stop()
date_col = c1.selectbox("Date column", raw.columns, index=list(raw.columns).index(guess_d))
val_col = c2.selectbox("Value column", num_cols, index=len(num_cols) - 1 if "y" in num_cols else 0)

try:
    s, info = F.prepare(raw, date_col, val_col)
except Exception as e:
    st.error(f"Could not prepare the data: {e}"); st.stop()

st.success(f"{info['n']} {info['freq_label']} observations, {info['start']:%Y-%m-%d} to {info['end']:%Y-%m-%d}"
           + (f"; {info['filled_gaps']} missing period(s) filled by interpolation" if info["filled_gaps"] else ""))
if info["n"] < 2 * info["period"]:
    st.warning(f"Only {info['n']} points for a seasonal period of {info['period']}: seasonal models need at least 2 full cycles to be reliable.")

alpha = 1 - conf / 100

@st.cache_data(show_spinner="Fitting model...")
def run_forecast(series, model, h, period, a):
    fn = F.MODELS[model]
    return fn(series, h, period, a)

@st.cache_data(show_spinner="Backtesting all models (about 30 seconds)...")
def run_backtest(series, period):
    horizon = min(12 if period >= 12 else 2 * period, max(len(series) // 6, period))
    return F.backtest(series, period, horizon=horizon, folds=3), horizon

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Trend & seasonality", "Model accuracy", "Data"])

with tab1:
    fc = run_forecast(s, model_name, horizon, info["period"], alpha)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    tail = s.iloc[-min(len(s), max(60, 3 * horizon)):]
    ax.plot(tail, color="k", lw=1.2, label="Actual"); ax.plot(fc.yhat, color="#4f46e5", label=f"{model_name} forecast")
    ax.fill_between(fc.index, fc.lower, fc.upper, color="#4f46e5", alpha=.18, label=f"{conf}% interval")
    ax.legend(); ax.set_title(f"{horizon}-period forecast"); st.pyplot(fig)
    out = fc.round(2).rename(columns={"yhat": "forecast", "lower": f"lower_{conf}", "upper": f"upper_{conf}"})
    out.index.name = "date"
    st.dataframe(out, width="stretch")
    st.download_button("Download forecast (CSV)", out.to_csv().encode(), "forecast.csv", "text/csv")

with tab2:
    if info["n"] >= 2 * info["period"]:
        dec, strength = F.decompose(s, info["period"])
        fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
        for a, c in zip(ax, ["observed", "trend", "seasonal", "resid"]): a.plot(dec[c], lw=1); a.set_ylabel(c)
        plt.tight_layout(); st.pyplot(fig)
        st.metric("Seasonal strength (0 to 1)", f"{strength:.2f}")
        adf = F.adf_report(s)
        st.write("**Stationarity (ADF test, p < 0.05 means stationary)**")
        st.table(pd.DataFrame(adf).T)
    else:
        st.info("Not enough data for a seasonal decomposition.")

with tab3:
    st.write("Rolling-origin backtest: each model is trained on past data only and scored on the next period, repeated 3 times.")
    if st.button("Run backtest for all models"):
        bt, hz = run_backtest(s, info["period"])
        summ = bt.groupby("model").agg(MAPE=("MAPE", "mean"), RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
                                       interval_coverage=("coverage", "mean")).round(2).sort_values("MAPE")
        st.caption(f"3 folds, horizon {hz} periods, 95% intervals. MAPE and coverage are in %.")
        st.dataframe(summ, width="stretch")
        st.caption("Lower MAPE/RMSE is better. Interval coverage should be close to 95%.")

with tab4:
    st.dataframe(s.rename("value").to_frame(), width="stretch")
