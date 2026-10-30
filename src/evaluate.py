"""Run decomposition, stationarity tests, rolling-origin backtest and final forecast on the sample data."""
import json, sys, time
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import forecasting as F

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "outputs"
s, info = F.prepare(pd.read_csv(ROOT / "data" / "retail_sales.csv"), "ds", "y")
print(info)

# 1. Decomposition + stationarity
dec, strength = F.decompose(s, info["period"])
adf = F.adf_report(s)
fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
for a, c in zip(ax, ["observed", "trend", "seasonal", "resid"]): a.plot(dec[c], lw=1); a.set_ylabel(c)
ax[0].set_title("STL decomposition"); plt.tight_layout(); plt.savefig(OUT / "decomposition.png", dpi=120); plt.close()

# 2. Backtest: 4 folds x 12 months (expanding window)
t = time.time()
bt = F.backtest(s, info["period"], horizon=12, folds=4)
bt.to_csv(OUT / "backtest_folds.csv", index=False)
summary = bt.groupby("model").agg(MAPE=("MAPE", "mean"), RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
                                  coverage_95=("coverage", "mean")).round(2).sort_values("MAPE")
summary.to_csv(OUT / "model_comparison.csv"); print(summary); print("backtest secs", round(time.time() - t))
print(bt.pivot(index="fold", columns="model", values="MAPE").round(2))

# 3. Final 12-month forecast from every model
H = 12; fcs = {}
for name, fn in F.MODELS.items():
    fcs[name] = fn(s, H, info["period"], 0.05)
    fcs[name].to_csv(OUT / f"forecast_{name.lower().replace(' ', '_')}.csv")
_, order = F.sarima(s, H, info["period"], return_order=True)
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.plot(s["2012":], color="k", lw=1.2, label="Actual")
for name, c in [("SARIMA", "#4f46e5"), ("Prophet", "#d9480f")]:
    f = fcs[name]; ax.plot(f.yhat, color=c, label=name); ax.fill_between(f.index, f.lower, f.upper, color=c, alpha=.15)
ax.set_title("12-month forecast with 95% intervals"); ax.legend(); plt.tight_layout(); plt.savefig(OUT / "forecast.png", dpi=120); plt.close()

json.dump(dict(n_obs=info["n"], start=str(info["start"].date()), end=str(info["end"].date()), seasonal_strength=round(strength, 3),
               adf=adf, sarima_order=order, best_model=summary.index[0], backtest_folds=4, horizon=12),
          open(OUT / "summary.json", "w"), indent=2, default=str)
print(json.dumps(dict(strength=round(strength, 3), adf=adf, order=order), default=str))
