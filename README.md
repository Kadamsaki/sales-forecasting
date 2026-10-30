# Sales Demand Forecasting Web App

Forecast retail sales with trend/seasonality decomposition, three competing models (Seasonal Naive baseline, SARIMA, Prophet), honest rolling-origin backtesting (MAPE, RMSE), and a Streamlit app where users upload their own data and download forecasts with confidence intervals.

**Stack:** Python, statsmodels (SARIMA, STL, ADF), Prophet, pandas, Streamlit.

## Data
US monthly retail sales, Jan 1992 - May 2016 (293 observations; the dataset shipped with Prophet's examples; values look like millions of USD but the file does not state units).

## Findings
- **Strong, stable seasonality:** STL seasonal strength 0.94 (0 to 1). December spikes, January/February dips, on top of a steady upward trend.
- **Non-stationary level:** ADF p = 0.86 on the raw series, p = 0.01 after first differencing, so one difference (d = 1) is used. Seasonal differencing (D = 1) handles the yearly cycle.
- **Seasonality grows with the level**, so SARIMA is fitted on log sales (multiplicative effects); Prophet uses multiplicative seasonality.

## Model comparison
Rolling-origin backtest: 4 folds x 12-month horizon, expanding training window (test periods cover Jun 2012 - May 2016). Models only ever see data before each test period.

| Model | MAPE | RMSE | MAE | 95% interval coverage |
|---|---|---|---|---|
| **SARIMA** (log, order picked by AIC) | **1.28%** | 6,598 | 5,449 | 100% |
| Prophet (default settings) | 3.08% | 14,156 | 13,007 | ~65% |
| Seasonal Naive (baseline) | 3.29% | 15,475 | 14,157 | 96% |

- SARIMA cuts the error by about 60% versus the baseline; Prophet barely beats the baseline here.
- Prophet's nominal 95% intervals contained only about two-thirds of actual values (too narrow). SARIMA's were conservative (every point inside), which with only 48 test points means "slightly too wide" rather than "perfectly calibrated".

## App features
Upload a CSV or use the sample data; pick date and value columns; automatic frequency detection (daily / weekly / monthly / quarterly) and gap filling; choose model, horizon (3-36 periods) and confidence level (80-99%); forecast chart with intervals, decomposition, ADF test, one-click backtest of all models, and CSV download.

## Limitations
- One dataset, 4 backtest folds: results show the method works, not that SARIMA always beats Prophet.
- No COVID-era or other structural breaks in this data; forecasts assume past patterns continue. No holidays, promotions or price effects are modelled.
- Prophet runs with default settings (not tuned); SARIMA chooses among only 8 candidate orders.
- MAPE is undefined when actual values are zero.
- Uploaded data is aggregated by summing values within each period.

## Run
```
pip install -r requirements.txt
python src/evaluate.py          # decomposition, backtest, forecasts -> outputs/
streamlit run app.py            # the web app
```
Deploy free on Streamlit Community Cloud (connect the GitHub repo, main file `app.py`).
