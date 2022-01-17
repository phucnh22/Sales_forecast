# %% next steps:
# try data transformation: Box-Cox, log...

# COMPARE:
# use 1 model by level to forecast individual stores
# individual models for individual stores

# Giang: level C
# Phuc-Thuong: level D

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
k = 4 # number of folds
sp = 7 # seasonal period
ini_window = len(y_train) - steps_ahead*k # initial train set to fit a model
fh = list(range(1, steps_ahead+1)) # forecast horizon
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
        y_train_cv, y_test_cv = y_train[train], y_train[test]

        # fit/update
        if i == 0:  # initial fit
            forecaster.fit(y_train_cv, fh=fh)
        else:
            forecaster.update(y_train_cv, update_params=False)

        # predict
        y_pred_cv = forecaster.predict(fh)

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
forecaster_AutoARIMA = AutoARIMA(sp=7,
                                 max_order=None,
                                 random=True,
                                 random_state=123,
                                 n_fits=10)

# fit & evaluate model
cv_results_AutoARIMA, forecaster_AutoARIMA = cv_eval(
    y_train,
    forecaster_AutoARIMA, 
    cv, fh
    )

forecaster_AutoARIMA.summary()
cv_results_AutoARIMA

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AutoARIMA
# specify model & param grid
forecaster_AutoARIMA = AutoARIMA(sp=7,
                                 max_order=None,
                                 random=True,
                                 random_state=123,
                                 n_fits=10)

# fit & evaluate model
cv_results_AutoARIMA, forecaster_AutoARIMA = cv_eval(
    y_train,
    forecaster_AutoARIMA, 
    cv, fh
    )

forecaster_AutoARIMA.summary()
cv_results_AutoARIMA
