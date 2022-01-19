# %% next steps:
# try data transformation: Box-Cox, log...

# COMPARE:
# use 1 model by level to forecast individual stores
# individual models for individual stores

# Giang: level C
# Phuc-Thuong: level D

from sktime.forecasting.theta import ThetaForecaster
import itertools
from sktime.forecasting.ets import AutoETS
from tkinter.messagebox import NO
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series, plot_correlations, plot_lags
from prophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError, MeanAbsoluteError, MeanAbsoluteScaledError


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
y_test_final = ts[-steps_ahead:]  # out-of-sample test set


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# k-fold rolling cross validation
k = 4  # number of folds
sp = 7  # seasonal period
ini_window = len(y_train) - steps_ahead*k  # initial train set to fit a model
fh = list(range(1, steps_ahead+1))  # forecast horizon
cv = ExpandingWindowSplitter(
    initial_window=ini_window, fh=fh, step_length=steps_ahead)

scoring_MAPE = MeanAbsolutePercentageError()
scoring_MAE = MeanAbsoluteError()
scoring_MASE = MeanAbsoluteScaledError()

# function to evaluate model
def cv_eval(y_train, forecaster, cv, fh):
    results = pd.DataFrame()
    for i, (train, test) in enumerate(cv.split(y_train)):
        # split data
        y_train_cv, y_test_cv = y_train.iloc[train,], y_train.iloc[test,]


        # fit/update
        if i == 0:  # initial fit
            forecaster.fit(y_train_cv)
        else:
            forecaster.update(y_train_cv, update_params=False)

        # predict
        y_pred_cv = forecaster.predict(fh=fh)

        # score
        score_MAE = scoring_MAE(y_test_cv, y_pred_cv)
        score_MAPE = scoring_MAPE(y_test_cv, y_pred_cv)
        score_MASE = scoring_MASE(y_test_cv, y_pred_cv, y_train=y_train_cv)

        # save results
        results = results.append({
            'y_test': y_test_cv,
            'y_pred_cv': y_pred_cv,
            'MAE': score_MAE,
            'MAPE': score_MAPE,
            'MASE': score_MASE,
        },
            ignore_index=True,
        )
    return results, forecaster


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NAIVE
param_grid_NAIVE = ['last', 'mean', 'drift']
cv_res_NAIVE = []

# grid search for best param
for param in param_grid_NAIVE:
    # define model
    cv_mod_NAIVE = NaiveForecaster(sp=sp, strategy=param)
    # fit model to training data
    cv_mape_NAIVE = evaluate(cv_mod_NAIVE, cv, y_train, return_data=True)
    cv_res_NAIVE.append(
        round(np.mean(cv_mape_NAIVE['test_MeanAbsolutePercentageError']), 3))

cv_res_NAIVE = pd.DataFrame(
    {'param': param_grid_NAIVE, 'mape': cv_res_NAIVE}).sort_values('mape')
cv_res_NAIVE


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AutoARIMA
# specify model & param grid
forecaster_AutoARIMA = AutoARIMA(sp=sp,
                                 max_order=None,
                                 random=True,
                                 random_state=123,
                                 n_fits=10)

# fit & evaluate model
cv_results_AutoARIMA, forecaster_AutoARIMA = cv_eval(
    y_train,
    forecaster_AutoARIMA,
    cv, fh)

forecaster_AutoARIMA.summary()
cv_results_AutoARIMA


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Prophet
# data prep
y_train_Prophet = pd.DataFrame(y_train.reset_index()).rename(
    columns={'index': 'ds', 'sales': 'y'})

# specify model & param grid
param_grid_Prophet = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}
param_grid_Prophet = [dict(zip(param_grid_Prophet.keys(), v))
                      for v in itertools.product(*param_grid_Prophet.values())]

tuning_mape_PROPHET = []
for params in param_grid_Prophet:
    forecaster_Prophet = Prophet(**params)
    cv_results_Prophet = pd.DataFrame()
    for i, (train, test) in enumerate(cv.split(y_train_Prophet)):
        # split data
        y_train_cv, y_test_cv = y_train_Prophet.iloc[train,], y_train_Prophet.iloc[test,]

        # fit/update
        if i == 0:  # initial fit
            forecaster_Prophet.fit(y_train_cv)
        else:
            forecaster_Prophet.update(y_train_cv, update_params=False)

        # predict
        y_pred_cv = forecaster_Prophet.predict(y_test_cv[['ds']])

        # score
        score_MAE = scoring_MAE(y_test_cv['y'], y_pred_cv['yhat'])
        score_MAPE = scoring_MAPE(y_test_cv['y'], y_pred_cv['yhat'])
        score_MASE = scoring_MASE(y_test_cv['y'], y_pred_cv['yhat'], y_train=y_train_cv['y'])

        # save results
        cv_results_Prophet.append({
            'y_test': y_test_cv,
            'y_pred_cv': y_pred_cv,
            'MAE': score_MAE,
            'MAPE': score_MAPE,
            'MASE': score_MASE,
        }, ignore_index=True)
    cv_results_Prophet

tuning_results = pd.DataFrame(param_grid_Prophet)
tuning_results['mape'] = tuning_mape_PROPHET
tuning_results.sort_values('mape')


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AutoETS
# data prep
y_train_ETS = y_train.replace(0, np.nan).ffill()

# specify model & param grid
forecaster_AutoETS = AutoETS(sp=sp, auto=True, n_jobs=-1)

# fit & evaluate model
cv_results_AutoETS, forecaster_AutoETS = cv_eval(
    y_train_ETS,
    forecaster_AutoETS,
    cv, fh
)

forecaster_AutoETS.summary()
cv_results_AutoETS

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Theta
# data prep
y_train_Theta = y_train.replace(0, np.nan).ffill()

# specify model & param grid
forecaster_Theta = ThetaForecaster(sp=sp)

# fit & evaluate model
cv_results_Theta, forecaster_Theta = cv_eval(
    y_train_Theta,
    forecaster_Theta,
    cv, fh
)

# %%
cv_results_AutoARIMA.iloc[:, :3].mean()
cv_results_AutoETS.iloc[:, :3].mean()
cv_results_Theta.iloc[:, :3].mean()

#%%
from prophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
