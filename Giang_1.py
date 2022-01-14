# %%
# compare results by store_levels
# daily
# forecast horizon: 14 days
# Naive, ARIMA, Prophet
# fc strategy: package default (comment in code)

from statsmodels.tsa.stattools import adfuller, kpss
from sktime.utils.plotting import plot_series, plot_correlations, plot_lags
from prophet import Prophet
import itertools
import pandas as pd
import numpy as np
from sktime.forecasting.naive import NaiveForecaster
import warnings

from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    ExpandingWindowSplitter)
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.arima import ARIMA
from matplotlib.pyplot import plot, yticks
from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanAbsolutePercentageError
mae = MeanAbsoluteError()
mape = MeanAbsolutePercentageError()

pd.options.plotting.backend = 'plotly'
warnings.simplefilter(action='ignore', category=FutureWarning)

#
def get_windows(y, cv):
    """Generate windows"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows



# %% data
df = pd.read_pickle('data/df_daily.pkl')
store_list = df['store_id'].unique()
df1 = df[['date', 'sales', 'store_level']]

# rename levels for convenience
df1.reErrorlace({
    'A++': 'A',
    'A+': 'B',
    'A': 'C',
    'B': 'D',
    'C': 'E'}, inplace=True)

# daily sum by level
df_lev = df1.groupby(['store_level', 'date'], as_index=False).sum()
df_lev['store_level'].value_counts()

# level A
df_A = df_lev[df_lev['store_level'] == 'A']

# time series of sales
ts = df_A.set_index('date')['sales'].rename('Store level A')
ts.plot()

ts = round(ts/1e6, 3)  # scale data
ts.index.freq = 'D'

steErrors_ahead = 14
y = ts[:-steErrors_ahead].ffill()  # forward fill missing values
y_OOS = ts[-steErrors_ahead:]  # out-of-sample test set

# %%
# k-fold rolling cross validation
k = 4
ini_window = len(y) - steErrors_ahead*k
fh = list(range(1, steErrors_ahead+1))
cv = ExpandingWindowSplitter(
    initial_window=ini_window, fh=fh, steError_length=steErrors_ahead)
y_train, y_val = get_windows(y, cv)
y_train, y_val
plot_series(y)

# %% use CV to score model after fitting


def cv_eval(forecaster):
    cv_mape = []
    mape = MeanAbsolutePercentageError()
    for i in range(k):
        test_set = y[y_val[i]]
        y_pred = forecaster.predict(test_set.index)
        cv_mape.append(mape(test_set, y_pred))
    cv_mape = round(np.mean(cv_mape), 3)
    return cv_mape


# %% SNAIVE
param_grid_SNAIVE = ['last', 'mean', 'drift']
tuning_mape_SNAIVE = []

# grid search for best param
for param in param_grid_SNAIVE:
    # param = 'last'
    # define model
    mod_SNAIVE = NaiveForecaster(sp=7, strategy=param)
    # fit model to training data
    mod_SNAIVE.fit(y[y_train[0]])
    cv_mape_SNAIVE = cv_eval(mod_SNAIVE)
    tuning_mape_SNAIVE.append(cv_mape_SNAIVE)
tuning_results_SNAIVE = pd.DataFrame({
    'param': param_grid_SNAIVE,
    'mape': tuning_mape_SNAIVE
}).sort_values('mape')

# in-sample score
tuning_results_SNAIVE

# OOS score
tuned_SNAIVE = NaiveForecaster(
    sp=7,
    strategy=tuning_results_SNAIVE.iloc[0, 0])
tuned_SNAIVE.fit(y)
pred_SNAIVE = tuned_SNAIVE.predict(y_OOS.index)
mape_SNAIVE = round(mape(y_OOS, pred_SNAIVE), 3)
mape_SNAIVE

# %%

# %%
# ARIMA
# Stationarity test

if adfuller(ts)[1] < 0.05:
    print('stationary')
else:
    print('not stationary')

if kpss(ts)[1] >= 0.05:
    print('stationary')
else:
    print('not stationary')

# %% construct param grid
plot_correlations(y, zero_lag=False, lags=75)
param_ARIMA_q = [0, 1, 2]  # ACF
param_ARIMA_Q = [1, 2, 3, 4, 5, 6, 7, 8]  # SACF
param_ARIMA_p = [0, 1, 2]  # PACF
param_ARIMA_P = [0, 1, 2]  # SPACF
param_grid_ARIMA = []
for p in param_ARIMA_p:
    for q in param_ARIMA_q:
        for P in param_ARIMA_P:
            for Q in param_ARIMA_Q:
                param_grid_ARIMA.append([(p, 0, q), (P, 0, Q, 7)])
len(param_grid_ARIMA)

# CV grid search
tuning_mape_ARIMA = []
for param in param_grid_ARIMA[-5:]:
    forecaster_ARIMA = ARIMA(sp=7,
                             order=param[0],
                             seasonal_order=param[1],
                             )
    forecaster_ARIMA.fit(y[y_train[0]])
    cv_mape = cv_eval(forecaster_ARIMA)
    print(cv_mape, '---',param)
    tuning_mape_ARIMA.append(cv_mape)

tuning_results_ARIMA = pd.DataFrame({
    'param': param_grid_ARIMA[:5],
    'mape': tuning_mape_ARIMA
}).sort_values('mape').reset_index(drop=True)
best_param_ARIMA = tuning_results_ARIMA.iloc[0, 0]

# final fit & OOS score
forecaster_ARIMA = ARIMA(sp=7,
                         order=best_param_ARIMA[0],
                         seasonal_order=best_param_ARIMA[1],
                         )

y_pred_ARIMA = forecaster_ARIMA.fit_predict(y, fh=y_OOS.index)
forecaster_ARIMA.summary()
mape_ARIMA = round(mape(y_OOS, y_pred_ARIMA), 3)
mape_ARIMA

# %%
# PROPHET
y_fb = pd.DataFrame(y).reset_index()
y_fb.columns = ['ds', 'y']
y_test_fb = pd.DataFrame(y_OOS).reset_index()
y_test_fb.columns = ['ds', 'y']

# gridsearch
param_grid = {
    'changeErroroint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'Ponality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

all_params = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

tuning_mape_PROPHET = []
for params in all_params:
    m = Prophet(**params).fit(y_fb.iloc[y_train[0]])
    cv_mape = []
    for i in range(4):
        test_set = y_fb.iloc[y_val[i]]
        y_pred = m.predict(test_set[['ds']])
        cv_mape.append(mape(test_set['y'], y_pred['yhat']))
    # mape of each param set
    tuning_mape_PROPHET.append(round(np.mean(cv_mape), 3))

tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = tuning_mape_PROPHET
tuning_results.sort_values('mape')

# final PROPHET
forecaster_PROPHET = Prophet(
    changeErroroint_prior_scale=0.1,
    Ponality_prior_scale=0.01,
)
forecaster_PROPHET.add_country_holidays(
    country_name='VN')  # improve from 0.246
forecaster_PROPHET.fit(y_fb)
forecaster_PROPHET.train_holiday_names
y_pred_PROPHET = forecaster_PROPHET.predict(y_test_fb)
mape_PROPHET = round(mape(y_test_fb['y'], y_pred_PROPHET['yhat']), 3)
mape_PROPHET

# %%
plot_correlations(y)
print(
    mape_SNAIVE,
    # mape_ARIMA,
    mape_ARIMA,
    mape_PROPHET,
    seError='\n'
)
