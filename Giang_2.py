# %% next steps:
# try data transformation: Box-Cox, log...

# COMPARE:
# use 1 model by level to forecast individual stores
# individual models for individual stores

# Giang: level C
# Phuc-Thuong: level D

import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series, plot_correlations, plot_lags
from prophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
mape = MeanAbsolutePercentageError()
from statsmodels.tsa.stattools import adfuller, kpss

# %% data
df_lev = pd.read_pickle('data/df_lev.pkl').set_index('date')
ts = df_lev[df_lev['store_level'] == 'C']['sales']
ts = round(ts/1e6, 3)
ts.index.freq = 'D'
ts.plot()

steps_ahead = 14
# in-sample/train set
ts.isna().sum()
y_train = ts[:-steps_ahead]

# out-of-sample/test set
y_test = ts[-steps_ahead:]  # out-of-sample test set



# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# k-fold rolling cross validation
k = 4
# y_train = ts[-steps_ahead*10:-steps_ahead]
ini_window = len(y_train) - steps_ahead*k
fh = list(range(1, steps_ahead+1))
cv = ExpandingWindowSplitter(
    initial_window=ini_window, fh=fh, step_length=steps_ahead)
get_windows(y_train, cv)


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NAIVE
param_grid_NAIVE = ['last', 'mean']
cv_res_NAIVE = []

# grid search for best param
for param in param_grid_NAIVE:
    # param = 'mean'
    # define model
    cv_mod_NAIVE = NaiveForecaster(sp=2, strategy=param)
    # fit model to training data
    cv_mape_NAIVE = evaluate(cv_mod_NAIVE, cv, y_train, return_data=True)
    cv_res_NAIVE.append(round(np.mean(cv_mape_NAIVE['test_MeanAbsolutePercentageError']), 3))

cv_res_NAIVE = pd.DataFrame({'param': param_grid_NAIVE,'mape': cv_res_NAIVE})
cv_res_NAIVE



# %%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# plot_correlations(y_train, zero_lag=False)

# order
param_ARIMA_p = [5]  # PACF
param_ARIMA_d = [1]  # stationarity test
param_ARIMA_q = [0]  # ACF

# seasonal order
param_ARIMA_P = [2]  # SPACF
param_ARIMA_D = [0]  # stationarity test
param_ARIMA_Q = [0]  # SACF

param_grid_ARIMA = []
param_grid_ARIMA_seas = []
for p in param_ARIMA_p:
    for d in param_ARIMA_d:
        for q in param_ARIMA_q:
            for P in param_ARIMA_P:
                for D in param_ARIMA_D:
                    for Q in param_ARIMA_Q:
                        param_grid_ARIMA.append((p, d, q))
                        param_grid_ARIMA_seas.append((P, D, Q, 7))

param_grid_ARIMA, param_grid_ARIMA_seas


#%% 
# grid search
tuning_mape_ARIMA = []
tuning_pred_ARIMA = []
for param in param_grid_ARIMA:
    forecaster_ARIMA = ARIMA(sp=7,
                             order=param[0],
                             seasonal_order=param[1],
                             )
    # fit model once on 1st fold train set
    cv_mape_ARIMA = []
    fold_0_train = y_train[get_windows(y_train, cv)[0][0]]
    fold_0_test = y_train[get_windows(y_train, cv)[0][1]]
    forecaster_ARIMA.fit(fold_0_train)
    # 1st fold pred & mape
    y_pred = forecaster_ARIMA.predict(fh=fh)
    cv_mape_ARIMA.append(round(mape(fold_0_test, y_pred),3))
    # CV evaluation (step-forward update without refitting)
    for i in range(1,k):
        fold_i_train = y_train[get_windows(y_train, cv)[i][0]]
        fold_i_test = y_train[get_windows(y_train, cv)[i][1]]
        forecaster_ARIMA.update(fold_i_train, update_params=False)
        y_pred = forecaster_ARIMA.predict(fh=fh)
        cv_mape_ARIMA.append(round(mape(fold_i_test, y_pred),3))
        
    tuning_mape_ARIMA.append(round(np.mean(cv_mape_ARIMA), 3))

tuning_mape_ARIMA

#%% 
# AutoARIMA
plot_correlations(y_train, zero_lag=False)
tuning_mape_AutoARIMA = []
forecaster_AutoARIMA = AutoARIMA(sp=7)

# fit model once on 1st fold train set
cv_mape_AutoARIMA = []
fold_0_train = y_train[get_windows(y_train, cv)[0][0]]
fold_0_test = y_train[get_windows(y_train, cv)[0][1]]
forecaster_AutoARIMA.fit(fold_0_train)
# pred & mape of 1st fold
y_pred = forecaster_AutoARIMA.predict(fh=fh)
cv_mape_AutoARIMA.append(round(mape(fold_0_test, y_pred),3))

### ???
forecaster_AutoARIMA.update_predict(y_train, cv, update_params=False)
forecaster_AutoARIMA.update_predict(y_train, cv)#, update_params=False)
forecaster_AutoARIMA.summary()

cv_mape_AutoARIMA.append(round(mape(fold_i_test, y_pred),3))
tuning_mape_AutoARIMA
