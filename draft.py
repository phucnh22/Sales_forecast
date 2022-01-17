#%% draft
from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series, plot_correlations, plot_lags


#%% NAIVE - last
y_train_short = y_train[-20:]
steps = 7
forecaster = NaiveForecaster(strategy="last", sp=steps)
forecaster.fit(y_train_short)
fh=list(range(1,steps+1))
y_pred = forecaster.predict(fh)
plot_series(y_train_short, y_pred)



#%% NAIVE - mean
y_train_short = y_train[-12:]
fh=list(range(1,4))
forecaster = NaiveForecaster(strategy="mean", sp=3)
forecaster.fit(y_train_short)
y_pred = forecaster.predict(fh)
plot_series(y_train_short, y_pred)

#%%
print(y_pred,
    y_train_short[[7,4,1]].mean(),
    y_train_short[[8,5,2]].mean(),
    y_train_short[[9,6,3,0]].mean(),
    sep='\n'
)

round(y_pred[0] - y_train_short[[8,5,2]].mean())
round(y_pred[1] - y_train_short[[9,6,3,0]].mean())
round(y_pred[2] - y_train_short[[10,7,4,1]].mean())


#%% NAIVE - drift
y_train_short = y_train[-50:]
fh=list(range(1,4))
forecaster = NaiveForecaster(strategy="drift")
forecaster.fit(y_train_short)
y_pred = forecaster.predict(fh)
plot_series(y_train_short, y_pred)



# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# k-fold rolling cross validation
y_train = ts[-steps_ahead*7:-steps_ahead]
k = 4
ini_window = len(y_train) - steps_ahead*k
fh = list(range(1, steps_ahead+1))

cv = ExpandingWindowSplitter(
    initial_window=ini_window, fh=fh, step_length=steps_ahead)
cv_folds = get_windows(y_train, cv)


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NAIVE
param_grid_NAIVE = ['last']#, 'mean']
cv_res_NAIVE = []

# grid search for best param
for param in param_grid_NAIVE:
    # param = 'mean'
    # define model
    cv_mod_NAIVE = NaiveForecaster(sp=2, strategy=param)
    # fit model to training data
    cv_mape_NAIVE = evaluate(cv_mod_NAIVE, cv, y_train, return_data=True)#, strategy='update')
    cv_res_NAIVE.append(round(np.mean(cv_mape_NAIVE['test_MeanAbsolutePercentageError']), 3))

cv_res_NAIVE

#%%
cv_train = cv_folds[0][0]
cv_test = cv_folds[0][1]
cv_y_train = y_train.iloc[cv_train]
cv_y_test = y_train.iloc[cv_test]

cv_mape_NAIVE['y_pred'][0]



#%% if refit for each fold, fold's params will differ
y_pred_re1 = forecaster_ARIMA.fit_predict(y_train[get_windows(y_train, cv)[0][0]], fh=fh)
y_pred_re2 = forecaster_ARIMA.fit_predict(y_train[get_windows(y_train, cv)[1][0]], fh=fh)
y_pred_re1, y_pred_re2


#%%

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
    results = results.append(
        {"y_test": y_test_cv,
        "y_pred_cv": y_pred_cv,
        'MAE': score_MAE,
        'MAPE': score_MAPE,
        'MASE': score_MASE,
        },
        ignore_index=True,
    )
return results, forecaster
